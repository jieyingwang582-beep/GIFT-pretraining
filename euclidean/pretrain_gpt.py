# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

# Capture the true program start time BEFORE any heavy imports.
import time
_PROGRAM_START_TIME = time.time()

import json

# Suppress warnings on all ranks but rank 0.
import os
import warnings
rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import List, Optional, Tuple

import torch

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model, get_thd_batch_on_this_cp_rank, get_batch_on_this_hybrid_cp_rank, StragglerDetector
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank, get_mtp_ranks
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()

# -----------------------------
# FP8 baseline imports (NO Fisher)
# -----------------------------
import torch.nn as nn
import torch.distributed as dist
from megatron.core.optimizer.baseline_fp8_manager import BaselineFP8Manager

from megatron.core.distributed.finalize_model_grads import (
    finalize_model_grads as _orig_finalize_model_grads
)
import megatron.training.training as _mtt


# ------------------------------------------
# FP8 baseline global state + logging tag
# ------------------------------------------
_CURRENT_FP8_TAG = "BASELINE"
global_fp8_manager: Optional[BaselineFP8Manager] = None
_SKIP_DP_SYNC_THIS_STEP = False

_FP8_LOG_EVERY = int(os.getenv("FP8_BASELINE_LOG_EVERY", "100"))  # set to 1 for immediate prints


def _set_fp8_tag(tag: str):
    global _CURRENT_FP8_TAG
    _CURRENT_FP8_TAG = tag


# Tag rank0 prints in this file only (optional; safe)
_orig_print_rank_0 = print_rank_0


def print_rank_0(*args, **kwargs):
    prefix = f"[{_CURRENT_FP8_TAG}]"
    if len(args) == 1 and isinstance(args[0], str):
        return _orig_print_rank_0(prefix + " " + args[0], **kwargs)
    return _orig_print_rank_0(prefix, *args, **kwargs)


def _step_from_args() -> int:
    args = get_args()
    if hasattr(args, "curr_iteration"):
        return int(args.curr_iteration)
    return int(getattr(args, "iteration", 0))


def _real_model(m):
    m = m[0] if isinstance(m, (list, tuple)) else m
    return m.module if hasattr(m, "module") else m


def _get_param_grad(p: torch.nn.Parameter):
    if hasattr(p, "main_grad") and (p.main_grad is not None):
        return p.main_grad
    if getattr(p, "grad", None) is not None:
        return p.grad
    return None


@torch.no_grad()
def allreduce_remaining_grads_fp32(model: nn.Module, group: dist.ProcessGroup, already_synced_param_ids: set):
    """Sync any grads NOT already handled by FP8 manager. Keeps training correct."""
    if group is None:
        return
    ws = dist.get_world_size(group=group)

    grads = []
    meta = []  # (grad_ref, shape, dtype)

    for p in model.parameters():
        if id(p) in already_synced_param_ids:
            continue
        g = _get_param_grad(p)
        if g is None:
            continue
        if getattr(g, "is_sparse", False):
            continue
        grads.append(g.detach())
        meta.append((g, g.shape, g.dtype))

    if not grads:
        return

    flat = torch.cat([g.float().contiguous().view(-1) for g in grads], dim=0)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    flat.div_(float(ws))

    offset = 0
    for (g, shape, dtype) in meta:
        n = g.numel()
        chunk = flat[offset:offset + n].view(shape)
        g.copy_(chunk.to(dtype=dtype))
        offset += n


# ---------------------------------------------------------
# Patch: finalize_model_grads to SKIP Megatron DP sync
# ---------------------------------------------------------
def finalize_model_grads_skip_dp_sync(model, num_tokens=None, pg_collection=None,force_all_reduce=False):
    # If we are NOT skipping, run Megatron normal DP sync.
    if not _SKIP_DP_SYNC_THIS_STEP:
        return _orig_finalize_model_grads(model, num_tokens=num_tokens, pg_collection=pg_collection)

    # Otherwise, prevent Megatron from doing DP allreduce here.
    saved = []
    for m in model:
        if hasattr(m, "finish_grad_sync"):
            saved.append((m, m.finish_grad_sync))
            m.finish_grad_sync = (lambda *a, **k: None)
    try:
        return _orig_finalize_model_grads(model, num_tokens=num_tokens, pg_collection=pg_collection)
    finally:
        for m, fn in saved:
            m.finish_grad_sync = fn


# ---------------------------------------------------------
# Wrap optimizer.step: do FP8 baseline + FP32 remainder sync
# ---------------------------------------------------------
def wrap_optimizer_step_once(model, optimizer, manager: BaselineFP8Manager):
    if getattr(optimizer, "_fp8_baseline_wrapped", False):
        return
    optimizer._fp8_baseline_wrapped = True
    _orig_step = optimizer.step

    def step_wrapper(*args, **kwargs):
        step = _step_from_args()
        real = _real_model(model)
        group = manager.group

        # 1) FP8 sync on allowed modules only
        did_apply, synced_ids = manager.sync_gradients_fp8(real, step)

        # 2) FP32 sync on everything else for correctness
        allreduce_remaining_grads_fp32(real, group=group, already_synced_param_ids=synced_ids)

        # 3) Optional: print coverage on rank0
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            if (_FP8_LOG_EVERY > 0) and (step % _FP8_LOG_EVERY == 0):
                # estimate pct by counting synced grad elements vs total grad elements
                total = 0
                for p in real.parameters():
                    g = _get_param_grad(p)
                    if g is not None:
                        total += g.numel()

                fp8_numel = 0
                # synced_ids includes weights/bias objects we wrote back; count their grads
                for name, module in real.named_modules():
                    # manager internally chooses by allowed module names; but we can approximate:
                    # count any module weight/bias grad whose id is in synced_ids
                    w = getattr(module, "weight", None)
                    if w is not None and id(w) in synced_ids:
                        g = _get_param_grad(w)
                        if g is not None:
                            fp8_numel += g.numel()
                    b = getattr(module, "bias", None)
                    if b is not None and id(b) in synced_ids:
                        g = _get_param_grad(b)
                        if g is not None:
                            fp8_numel += g.numel()

                pct = 100.0 * (fp8_numel / float(max(1, total)))
                print(f"[FP8_BASELINE] step={step} fp8={pct:.2f}%  fp8_numel={fp8_numel}  total_grad_numel={total}  did_apply={did_apply}")

        return _orig_step(*args, **kwargs)

    optimizer.step = step_wrapper


# ---------------------------------------------------------
# Patch train_step: install skip-dp-sync + init manager
# ---------------------------------------------------------
_original_train_step = _mtt.train_step


def train_step_with_fp8_baseline(
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    opt_param_scheduler,
    config,
    forward_backward_func,
    iteration=None
):
    global global_fp8_manager, _SKIP_DP_SYNC_THIS_STEP

    step = _step_from_args()

    # Patch finalize_model_grads once
    if not getattr(config, "_fp8_baseline_patched", False):
        config.finalize_model_grads_func = finalize_model_grads_skip_dp_sync
        config._fp8_baseline_patched = True
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print("[FP8_BASELINE] patched config.finalize_model_grads_func -> finalize_model_grads_skip_dp_sync")

    # Init FP8 manager once
    if global_fp8_manager is None:
        dp_group = parallel_state.get_data_parallel_group()
        global_fp8_manager = BaselineFP8Manager(
            model=_real_model(model),
            comm_group=dp_group,
            warmup_steps=0,
        )
        global_fp8_manager.set_tag_callback(_set_fp8_tag)

        wrap_optimizer_step_once(
            model=model,
            optimizer=optimizer,
            manager=global_fp8_manager,
        )

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print("[FP8_BASELINE] BaselineFP8Manager initialized and optimizer.step wrapped")

    # Always skip Megatron DP sync; our optimizer.step wrapper will sync
    _SKIP_DP_SYNC_THIS_STEP = True

    return _original_train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
    )


# Activate patch globally
_mtt.train_step = train_step_with_fp8_baseline
print_rank_0("[FP8_BASELINE] patched megatron.training.training.train_step -> train_step_with_fp8_baseline")


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage) and (
    (not mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage))):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(
        data_iterator,
        mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
        )

    cu_seqlens = batch.pop('cu_seqlens', None)
    cu_seqlens_padded = batch.pop('cu_seqlens_padded', None)
    max_seqlen = batch.pop('max_seqlen', None)
    local_cp_size = batch.pop('local_cp_size', None)
    if local_cp_size is not None:
        local_cp_size = int(local_cp_size.item())

    if cu_seqlens is None and local_cp_size is None:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)  # The implementation of this function is in MCore
        packed_seq_params = None
    elif local_cp_size is None:  # Packed THD format
        assert max_seqlen.dim() == 1
        batch, packed_seq_params = get_thd_batch_on_this_cp_rank(batch, cu_seqlens, cu_seqlens_padded, max_seqlen)
    else: # Hybrid CP format
        batch, packed_seq_params = get_batch_on_this_hybrid_cp_rank(batch, local_cp_size)
    
    return (*batch.values(), packed_seq_params)


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, packed_seq_params=packed_seq_params
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    args = get_args()
    config = core_transformer_config_from_args(args)
    return (
        is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "context_parallel_size": args.context_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "sequence_parallel_size": args.tensor_model_parallel_size*args.sequence_parallel,
        "hybrid_context_parallel": args.hybrid_context_parallel,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    is_dataset_built = partial(is_dataset_built_on_rank, vp_stage=vp_stage)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
