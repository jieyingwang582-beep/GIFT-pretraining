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
from megatron.core.optimizer.fisher_manager import (FisherManager, wrap_optimizer_step_once, _real_model,)
from megatron.core.distributed.finalize_model_grads import (
    finalize_model_grads as _orig_finalize_model_grads
)
import megatron.training.training
import megatron
import glob
import torch.distributed as dist
import megatron.core.optimizer.fisher_manager as fisher_manager_module
import megatron.training.training as _mtt


global_fisher_manager = None
global_step_counter = 0

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()

def _ig_step() -> int:
    args = get_args()
    if hasattr(args, "curr_iteration"):
        return int(args.curr_iteration)
    return int(getattr(args, "iteration", 0))


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage) and (
        (not mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage))
    ):
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
    else:  # Hybrid CP format
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
        "sequence_parallel_size": args.tensor_model_parallel_size * args.sequence_parallel,
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


original_train_step = megatron.training.training.train_step


def _latest_iter_dir(base_dir: str) -> Optional[str]:
    if (base_dir is None) or (not os.path.isdir(base_dir)):
        return None
    cands = sorted(glob.glob(os.path.join(base_dir, "iter_*")))
    if not cands:
        return None
    return cands[-1]


def _maybe_save_ig_state(save_dir: str, iteration: int):
    """Save IG/KFAC state into the same iter_xxxxxxx directory as Megatron checkpoints."""
    global global_fisher_manager, global_step_counter

    if global_fisher_manager is None:
        return
    if save_dir is None:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    iter_dir = os.path.join(save_dir, f"iter_{int(iteration):07d}")
    if not os.path.isdir(iter_dir):
        iter_dir = os.path.join(save_dir, f"iter_{int(iteration)-1:07d}")
        if not os.path.isdir(iter_dir):
            return

    path = os.path.join(iter_dir, "ig_state.pt")
    tmp = path + ".tmp"

    payload = {
        "global_step_counter": int(global_step_counter),
        "fisher_manager": global_fisher_manager.state_dict(),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"[IG] saved state -> {path}")


def _maybe_load_ig_state(load_dir: str):
    """Load IG state from load_dir (can be base dir or iter dir). Prefer matching iteration."""
    global global_fisher_manager, global_step_counter
    if global_fisher_manager is None or not load_dir:
        return

    p = load_dir.rstrip("/")
    base = os.path.basename(p)

    cand_dirs = []

    if base.startswith("iter_") and os.path.isdir(p):
        cand_dirs.append(p)
    else:
        it = int(getattr(get_args(), "iteration", 0))
        cand_dirs.append(os.path.join(p, f"iter_{it:07d}"))
        if it > 0:
            cand_dirs.append(os.path.join(p, f"iter_{it-1:07d}"))
        latest = _latest_iter_dir(p)
        if latest is not None:
            cand_dirs.append(latest)

    for d in cand_dirs:
        path = os.path.join(d, "ig_state.pt")
        if not os.path.isfile(path):
            continue

        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and ("fisher_manager" in ckpt):
            global_fisher_manager.load_state_dict(ckpt["fisher_manager"])

            global_step_counter = _ig_step()

            if (not dist.is_initialized()) or dist.get_rank() == 0:
                print(f"[IG] restored state <- {path} (now_global_step_counter={global_step_counter})")
            return


def finalize_model_grads_skip_dp_sync(model, num_tokens=None, pg_collection=None, force_all_reduce=False):
    # If we are NOT explicitly allowed to skip, run Megatron's normal DP grad sync.
    if not fisher_manager_module._SKIP_DP_SYNC_THIS_STEP:
        return _orig_finalize_model_grads(
            model,
            num_tokens=num_tokens,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce
        )

    # Otherwise, prevent Megatron from doing the DP allreduce here.
    # (Your optimizer.step wrapper must perform the full gradient sync itself for correctness.)
    saved = []
    for m in model:
        if hasattr(m, "finish_grad_sync"):
            saved.append((m, m.finish_grad_sync))
            m.finish_grad_sync = (lambda *a, **k: None)
    try:
        return _orig_finalize_model_grads(
            model,
            num_tokens=num_tokens,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce
        )
    finally:
        for m, fn in saved:
            m.finish_grad_sync = fn


def train_step_with_kfac_injection(
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    opt_param_scheduler,
    config,
    forward_backward_func,
    **kwargs,
):
    global global_fisher_manager, global_step_counter
    global_step_counter = _ig_step()

    # 1) patch once
    if not getattr(config, "_skip_dp_sync_patched", False):
        config.finalize_model_grads_func = finalize_model_grads_skip_dp_sync
        config._skip_dp_sync_patched = True

    # 2) init fisher manager once
    if global_fisher_manager is None:
        args = get_args()
        global_fisher_manager = FisherManager(
            _real_model(model),
            stats_decay=0.95,
            damping=0.001,
            update_freq=args.ig_update_freq,
            warmup_steps=0,  # no warmup
            factor_decay=0.95,
            comm_group=parallel_state.get_data_parallel_group(),
        )

        wrap_optimizer_step_once(
            model=model,
            optimizer=optimizer,
            manager=global_fisher_manager,
            get_step=_ig_step,
        )

        # ---- restore IG state once (on init) ----
        args = get_args()
        candidate_dirs = []
        if getattr(args, "load", None):
            candidate_dirs.append(args.load)
        if getattr(args, "save", None):
            candidate_dirs.append(args.save)

        if dist.is_initialized():
            dist.barrier()
        for d in candidate_dirs:
            _maybe_load_ig_state(d)

        if dist.is_initialized():
            dist.barrier()

        # start from clean state: no hooks resident between steps
        global_fisher_manager._remove_hooks()

    # 3) From warmup_end onward, ALWAYS skip Megatron DP sync.
    #    Our optimizer.step wrapper will do full sync (FP32 or FP8+FP32) every step.
    mgr = global_fisher_manager
    step = _ig_step()
    fisher_manager_module._SKIP_DP_SYNC_THIS_STEP = (
        (mgr is not None) and (step >= int(mgr.warmup_steps))
    )

    # 4) Only register hooks on collect steps: 1, 51, 101, ...
    should_collect_this_step = False
    if mgr is not None:
        should_collect_this_step = mgr._is_collect_step(mgr._hook_step)
        if should_collect_this_step:
            mgr._register_hooks()
        else:
            mgr._remove_hooks()

    try:
        # 5) run original megatron train_step
        out = original_train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func,
            **kwargs,
        )
    finally:
        # ensure hooks do not stay resident after this step
        if mgr is not None and should_collect_this_step:
            mgr._remove_hooks()

    # 6) after Megatron potentially saved checkpoint, dump IG state into iter_xxxxxxx
    args = get_args()
    save_interval = getattr(args, "save_interval", None)
    save_dir = getattr(args, "save", None)

    if (save_interval is not None) and (save_dir is not None):
        ig_it = _ig_step()
        if (ig_it % int(save_interval)) == 0:
            _maybe_save_ig_state(save_dir, ig_it)
    return out


megatron.training.training.train_step = train_step_with_kfac_injection
_mtt.train_step = train_step_with_kfac_injection  # ensure patched in the imported module too

def add_extra_args(parser):
    parser.add_argument("--ig-update-freq", type=int, default=50)
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)
    return parser
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
        extra_args_provider=add_extra_args,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
