# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import functools
import logging
import math
import os
import time
import warnings
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import Dict, List, Optional

import torch
from torch.distributed import _coalescing_manager

import megatron.core.nccl_allocator as nccl_allocator
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import log_single_rank

from ..fp8_utils import (
    is_float8tensor,
    is_mxfp8tensor,
    modify_underlying_storage,
    post_all_gather_processing,
)
from ..utils import is_torch_min_version, log_on_each_pipeline_stage
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .reduce_scatter_with_fp32_accumulation import reduce_scatter_with_fp32_accumulation

logger = logging.getLogger(__name__)

try:
    if is_torch_min_version("1.13.0"):
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
    else:
        dist_all_gather_func = torch.distributed._all_gather_base
        dist_reduce_scatter_func = torch.distributed._reduce_scatter_base
except Exception:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base

import megatron.core.nccl_allocator as nccl_allocator


class _BaselineGradCommProfiler:
    """
    Measure baseline grad communication from launch to completion.

    This is intentionally a single aggregated metric:
      baseline_grad_comm
    """

    def __init__(self):
        self.total_ms = 0.0
        self.count = 0
        self._cuda_spans = []
        self._cpu_ms = 0.0
        self.print_freq = int(os.getenv("BASELINE_GRAD_COMM_PRINT_FREQ", "10"))
        self.reset_after_print = os.getenv("BASELINE_GRAD_COMM_RESET_AFTER_PRINT", "1") == "1"
        self._global_count = 0

    def _rank0(self) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def add_cuda_span(self, start_ev, end_ev):
        self._cuda_spans.append((start_ev, end_ev))
        self.count += 1
        self._global_count += 1
        self.maybe_print()

    def add_cpu_ms(self, ms: float):
        self._cpu_ms += float(ms)
        self.count += 1
        self._global_count += 1
        self.maybe_print()

    def flush(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self._cuda_spans:
            for s, e in self._cuda_spans:
                self.total_ms += float(s.elapsed_time(e))
            self._cuda_spans.clear()

        if self._cpu_ms > 0.0:
            self.total_ms += float(self._cpu_ms)
            self._cpu_ms = 0.0

    def maybe_print(self):
        if self.count == 0:
            return
        if (self._global_count % self.print_freq) != 0:
            return
        if not self._rank0():
            return

        self.flush()
        avg_ms = self.total_ms / max(1, self.count)
        print(
#             f"[BASELINE_GRAD_COMM_PROFILE] step={self._global_count}\n"
            f"  {'baseline_grad_comm':<36} total_ms={self.total_ms:10.3f}  "
            f"avg_ms={avg_ms:10.3f}  count={self.count}",
            flush=True,
        )

        if self.reset_after_print:
            self.total_ms = 0.0
            self.count = 0
            self._cuda_spans.clear()
            self._cpu_ms = 0.0


_BASELINE_GRAD_COMM_PROFILER = _BaselineGradCommProfiler()


class BufferType(Enum):
    """
    Enumeration for buffer type.
    """

    PARAM = 1
    GRAD = 2


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into data_parallel_world_size chunks of equal size.
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer


class _ParamAndGradBucket:
    """
    Bucket to keep track of a subset of the model's parameters and gradients.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        gradient_scaling_factor: float,
        bucket_id: int,
    ):
        self.params_list = params
        self.params = set(params)
        assert len(self.params_list) == len(self.params)
        self.param_data = param_data
        self.grad_data = grad_data
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.gradient_scaling_factor = gradient_scaling_factor
        self.bucket_id = bucket_id
        self.param_to_index = {}
        offset = 0
        for param in params:
            self.param_to_index[param] = (offset, offset + param.numel())
            offset += param.numel()


class _ParamAndGradBucketGroup:
    """
    Put multiple buckets into a group so that their communications can be aggregated together.
    """

    def __init__(
        self,
        buckets: List[_ParamAndGradBucket],
        ddp_config: DistributedDataParallelConfig,
        collective_group: torch.distributed.ProcessGroup,
        collective_group_size: int,
    ):
        self.buckets = buckets
        self.ddp_config = ddp_config

        if self.ddp_config.use_distributed_optimizer:
            self.intra_distributed_optimizer_instance_group = collective_group
            self.intra_distributed_optimizer_instance_size = collective_group_size
            self.intra_distributed_optimizer_instance_rank = collective_group.rank()
        else:
            self.data_parallel_group = collective_group

        self.param_to_bucket = {}
        self.params = set()
        for bucket in self.buckets:
            for param in bucket.params_list:
                self.param_to_bucket[param] = bucket
                self.params.add(param)

        self.next_param_gather_bucket_group = None

        if self.ddp_config.num_distributed_optimizer_instances > 1:
            self.inter_distributed_optimizer_instance_group = None
            self.communication_stream = None
            assert (
                not self.ddp_config.reduce_scatter_with_fp32_accumulation
            ), "RS w/ FP32 accumulation not supported with num_distributed_optimizer_instances > 1"

        global dist_reduce_scatter_func
        if self.ddp_config.reduce_scatter_with_fp32_accumulation:
            dist_reduce_scatter_func = reduce_scatter_with_fp32_accumulation
            log_single_rank(
                logger,
                logging.INFO,
                "Using reduce_scatter_with_fp32_accumulation as reduce-scatter implementation",
            )

        self.golden_per_param_grad_ready_counts = {}
        self.per_param_grad_ready_counts = {}
        self.is_last_microbatch = True
        self.is_first_batch = True

        self.param_gather_handle = None
        self.param_gather_dispatched = False
        self.grad_reduce_handle = None

        self.cached_param_buffer_shard_list = [None] * len(self.buckets)
        self.cached_grad_buffer_shard_list = [None] * len(self.buckets)

        # Timing state for full grad communication span: launch -> completion.
        self._grad_comm_start_event = None
        self._grad_comm_cpu_t0 = None
        self._grad_comm_timing_active = False

    def _start_grad_comm_timing(self, use_comm_stream: bool):
        if self._grad_comm_timing_active:
            return
        self._grad_comm_timing_active = True

        if torch.cuda.is_available():
            start_ev = torch.cuda.Event(enable_timing=True)
            if use_comm_stream:
                with torch.cuda.stream(self.communication_stream):
                    start_ev.record()
            else:
                start_ev.record()
            self._grad_comm_start_event = start_ev
            self._grad_comm_cpu_t0 = None
        else:
            self._grad_comm_cpu_t0 = time.perf_counter()
            self._grad_comm_start_event = None

    def _finish_grad_comm_timing(self, use_comm_stream: bool = False):
        if not self._grad_comm_timing_active:
            return

        if torch.cuda.is_available():
            end_ev = torch.cuda.Event(enable_timing=True)
            if use_comm_stream:
                with torch.cuda.stream(self.communication_stream):
                    end_ev.record()
            else:
                end_ev.record()
            _BASELINE_GRAD_COMM_PROFILER.add_cuda_span(self._grad_comm_start_event, end_ev)
        else:
            assert self._grad_comm_cpu_t0 is not None
            _BASELINE_GRAD_COMM_PROFILER.add_cpu_ms(
                (time.perf_counter() - self._grad_comm_cpu_t0) * 1000.0
            )

        self._grad_comm_start_event = None
        self._grad_comm_cpu_t0 = None
        self._grad_comm_timing_active = False

    def reset(self):
        """
        Reset metadata in bucket group in preparation for the next iteration of training.
        """
        if self.is_first_batch and len(self.per_param_grad_ready_counts) > 0:
            assert len(self.per_param_grad_ready_counts) == len(self.params)
            self.golden_per_param_grad_ready_counts = self.per_param_grad_ready_counts
            self.is_first_batch = False
        self.per_param_grad_ready_counts = {}
        self.is_last_microbatch = True

    def check_grads(self, check_for_nan_or_inf, check_for_large):
        """
        Make sure norm of grads in bucket are not NaN prior to data-parallel
        all-reduce / reduce-scatter.
        """
        rerun_state_machine = get_rerun_state_machine()
        for i in range(len(self.buckets)):
            grad_norm = self.buckets[i].grad_data.norm(p=2)
            if check_for_nan_or_inf:
                rerun_state_machine.validate_result(
                    result=grad_norm,
                    rejection_func=torch.isnan,
                    message=f"found NaN in local grad norm for bucket #{i} "
                    f"in backward pass before data-parallel communication collective",
                    tolerance=0.001,
                    fatal=True,
                )
                rerun_state_machine.validate_result(
                    result=grad_norm,
                    rejection_func=torch.isinf,
                    message=f"found Inf in local grad norm for bucket #{i} "
                    f"in backward pass before data-parallel communication collective",
                    tolerance=0.001,
                    fatal=True,
                )
            if check_for_large:
                rerun_state_machine.validate_result(
                    result=grad_norm,
                    rejection_func=partial(
                        rerun_state_machine.is_unexpectedly_large, threshold=10, context="grads"
                    ),
                    message=f"found unexpected large grads in bucket #{i} "
                    f"in backward pass before data-parallel communication collective",
                    tolerance=0.001,
                    fatal=False,
                )

    def start_param_sync(self, force_sync: bool = False):
        """
        Initiates all necessary param all-gathers for this bucket.
        """
        assert self.ddp_config.use_distributed_optimizer

        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                return
        else:
            assert self.param_gather_handle is None

        async_op = self.ddp_config.overlap_param_gather and not force_sync
        with _coalescing_manager(
            self.intra_distributed_optimizer_instance_group, async_ops=async_op
        ) as cm:
            for idx, bucket in enumerate(self.buckets):
                if self.cached_param_buffer_shard_list[idx] is None:
                    self.cached_param_buffer_shard_list[idx] = shard_buffer(
                        bucket.param_data, self.intra_distributed_optimizer_instance_size
                    )
                local_data_view = self.cached_param_buffer_shard_list[idx][
                    self.intra_distributed_optimizer_instance_rank
                ]
                dist_all_gather_func(
                    bucket.param_data,
                    local_data_view,
                    group=self.intra_distributed_optimizer_instance_group,
                    async_op=async_op,
                )
        if async_op:
            self.param_gather_handle = cm
        else:
            self.param_gather_handle = None
        self.param_gather_dispatched = True

    def finish_param_sync(self, skip_next_bucket_dispatch: bool = False):
        """
        Finishes param sync communication operation for this bucket.
        """
        assert self.ddp_config.use_distributed_optimizer
        assert self.ddp_config.overlap_param_gather

        if not self.param_gather_dispatched:
            self.start_param_sync()

        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            if self.next_param_gather_bucket_group is not None and not skip_next_bucket_dispatch:
                if self.next_param_gather_bucket_group.param_gather_dispatched:
                    warnings.warn(
                        "The next bucket's parameter all-gather operation has already been "
                        "dispatched. This may be caused by a mismatch between the order of "
                        "parameter registration and forward pass execution, which will "
                        "hurt the communication-computation overlap performance."
                    )
                else:
                    self.next_param_gather_bucket_group.start_param_sync()

            if self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
                for bucket in self.buckets:
                    for param in bucket.params:
                        param_start, param_end = bucket.param_to_index[param]
                        param_slice = bucket.param_data.view(-1)[param_start:param_end]
                        param.data.copy_(param_slice.view(param.data.shape))
                    bucket.param_data.zero_()
            else:
                fp8_params = []
                for bucket in self.buckets:
                    for param in bucket.params:
                        if is_float8tensor(param):
                            fp8_params.append(param)
                if len(fp8_params) > 0:
                    post_all_gather_processing(fp8_params)

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        This function now starts timing the full grad-communication span.
        """
        if self.is_first_batch and self.grad_reduce_handle is not None:
            return

        assert (
            self.grad_reduce_handle is None
        ), "Should not have multiple communication calls outstanding at once"

        if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
            self.check_grads(
                check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
                check_for_large=self.ddp_config.check_for_large_grads,
            )

        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor

        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        async_op = (
            self.ddp_config.overlap_grad_reduce
            and self.ddp_config.num_distributed_optimizer_instances == 1
        )

        use_comm_stream = (
            self.ddp_config.num_distributed_optimizer_instances > 1
            and self.ddp_config.overlap_grad_reduce
        )
        if use_comm_stream:
            stream_context = torch.cuda.stream(self.communication_stream)
            self.communication_stream.wait_stream(torch.cuda.default_stream())
        else:
            stream_context = nullcontext()

        if self.ddp_config.use_distributed_optimizer:
            communication_group = self.intra_distributed_optimizer_instance_group
        else:
            communication_group = self.data_parallel_group

        # Start timing right before the real communication launch.
        self._start_grad_comm_timing(use_comm_stream=use_comm_stream)

        grad_reduce_handle = None
        with stream_context, _coalescing_manager(communication_group, async_ops=async_op) as cm:
            for idx, bucket in enumerate(self.buckets):
                if self.ddp_config.use_distributed_optimizer and not force_all_reduce:
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data, self.intra_distributed_optimizer_instance_size
                        )
                    local_data_view = self.cached_grad_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]
                    grad_reduce_handle = dist_reduce_scatter_func(
                        local_data_view,
                        bucket.grad_data,
                        op=reduce_op,
                        group=communication_group,
                        async_op=async_op,
                    )
                else:
                    if torch.distributed.get_rank() == 0 and force_all_reduce:
                        logger.info(
                            f"Performing reduction using all_reduce because {force_all_reduce=}"
                        )
                    torch.distributed.all_reduce(
                        bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op
                    )

        if (
            self.ddp_config.use_distributed_optimizer
            and self.ddp_config.num_distributed_optimizer_instances > 1
        ):
            assert self.inter_distributed_optimizer_instance_group is not None
            with (
                stream_context,
                _coalescing_manager(
                    self.inter_distributed_optimizer_instance_group, async_ops=async_op
                ) as cm,
            ):
                for idx, bucket in enumerate(self.buckets):
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data, self.intra_distributed_optimizer_instance_size
                        )
                    local_data_view = self.cached_grad_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]

                    torch.distributed.all_reduce(
                        local_data_view,
                        op=reduce_op,
                        group=self.inter_distributed_optimizer_instance_group,
                        async_op=async_op,
                    )

        if async_op:
            if self.ddp_config.reduce_scatter_with_fp32_accumulation and not force_all_reduce:
                assert (
                    len(self.buckets) == 1
                ), "Only 1 bucket supported with reduce_scatter_with_fp32_accumulation=True"
                assert grad_reduce_handle is not None
                self.grad_reduce_handle = grad_reduce_handle
            else:
                self.grad_reduce_handle = cm
        else:
            self.grad_reduce_handle = None
            # Synchronous path: communication already completed here, so finish timing now.
            self._finish_grad_comm_timing(use_comm_stream=use_comm_stream)

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        This function now closes timing for async overlap path.
        """
        self.param_gather_dispatched = False

        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            return

        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)

        if self.ddp_config.num_distributed_optimizer_instances > 1:
            torch.cuda.default_stream().wait_stream(self.communication_stream)
            # Multi-instance overlap path uses comm stream launch and default-stream wait.
            self._finish_grad_comm_timing(use_comm_stream=False)
            return

        assert self.grad_reduce_handle is not None, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
            "params have grad available)"
        )
        self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None
        self._finish_grad_comm_timing(use_comm_stream=False)

    def register_grad_ready(
        self, param: torch.nn.Parameter, force_all_reduce: Optional[bool] = False
    ):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.
        """
        assert (
            self.ddp_config.overlap_grad_reduce
        ), "register_grad_ready() should only be called when overlap_grad_reduce is True"
        if self.is_last_microbatch:
            assert param in self.param_to_bucket, "Param is not in the bucket group"
            if param not in self.per_param_grad_ready_counts:
                self.per_param_grad_ready_counts[param] = 0
            self.per_param_grad_ready_counts[param] += 1
            if not self.is_first_batch:
                if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
                    assert len(self.per_param_grad_ready_counts) == len(self.params)
                    self.start_grad_sync(force_all_reduce=force_all_reduce)


class _ParamAndGradBuffer:
    """
    Groups parameters and gradients into a contiguous buffer, and then breaks the buffer into
    buckets with roughly `bucket_size` parameters each.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
        param_indices: List[int],
        nccl_ub: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):

        if pg_collection is None:
            self.dp_cp_group = parallel_state.get_data_and_context_parallel_group(
                with_context_parallel=True
            )
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            assert hasattr(pg_collection, 'tp') and hasattr(pg_collection, 'dp_cp')
            self.dp_cp_group = pg_collection.dp_cp
            self.tp_group = pg_collection.tp

        self.ddp_config = ddp_config
        self.params = params
        self.param_indices = param_indices

        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = self.data_parallel_group.size()
        self.gradient_scaling_factor = gradient_scaling_factor
        self.nccl_ub = nccl_ub

        self.buckets = []
        self.param_to_bucket = {}
        self.param_index_map = {}

        def _pad(number_to_be_padded: int, divisor: int) -> int:
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_end_of_bucket_if_needed(bucket_end_index: int) -> int:
            if self.ddp_config.use_distributed_optimizer:
                if self.ddp_config.pad_buckets_for_high_nccl_busbw:
                    bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128, 2**16)
                else:
                    bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128)
                return _pad(bucket_end_index, bucket_size_divisor)
            return bucket_end_index

        def _pad_start_of_param_if_needed(param_start_index: int) -> int:
            if self.ddp_config.use_distributed_optimizer:
                return _pad(param_start_index, 64)
            return param_start_index

        param_start_index = 0
        bucket_start_index = param_start_index
        bucket_params = set()
        self.bucket_indices = []
        per_bucket_numel_unpadded = []
        bucket_id = 0

        def _update_bucket_metadata(param_end_index: int) -> int:
            nonlocal bucket_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
            bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)

            self.bucket_indices.append((bucket_start_index, bucket_end_index))
            bucket_start_index = bucket_end_index
            bucket_params = set()
            bucket_id += 1
            return bucket_end_index

        def _does_param_require_new_bucket(param):
            return (
                getattr(param, "shared_embedding", False)
                and self.ddp_config.use_distributed_optimizer
            )

        for param in params[::-1]:
            this_numel = param.data.nelement()
            param_start_index = _pad_start_of_param_if_needed(param_start_index)

            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
                param_start_index = _update_bucket_metadata(param_start_index)

            param_end_index = param_start_index + this_numel
            self.param_index_map[param] = (param_start_index, param_end_index, bucket_id)
            bucket_params.add(param)

            if (
                bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
            ) or _does_param_require_new_bucket(param):
                bucket_end_index = _update_bucket_metadata(param_end_index)
                param_start_index = bucket_end_index
            else:
                param_start_index = param_end_index

        if len(bucket_params) > 0:
            bucket_end_index = _update_bucket_metadata(param_end_index)

        self.numel = bucket_end_index
        self.numel_unpadded = sum(per_bucket_numel_unpadded)
        assert self.numel_unpadded <= self.numel
        if self.ddp_config.use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded

        self.param_data = None

        if self.nccl_ub:
            nccl_allocator.init()
            pool = nccl_allocator.create_nccl_mem_pool(
                symmetric=not self.ddp_config.disable_symmetric_registration
            )
            mem_alloc_context = functools.partial(
                nccl_allocator.nccl_mem,
                pool,
                group=self.data_parallel_group,
                symmetric=not self.ddp_config.disable_symmetric_registration,
            )
            torch.distributed.barrier()
            tmp_warmup_tensor = torch.zeros([1], device="cuda")
            torch.distributed.all_reduce(tmp_warmup_tensor, group=self.data_parallel_group)
            torch.distributed.barrier()
        else:
            mem_alloc_context = nullcontext

        with mem_alloc_context():
            if self.ddp_config.use_distributed_optimizer and any(is_mxfp8tensor(p) for p in params):
                self.shared_buffer = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                if self.grad_dtype == torch.float32:
                    self.param_data = self.shared_buffer[: math.ceil(self.numel / 2)].view(
                        torch.bfloat16
                    )
                else:
                    self.param_data = self.shared_buffer
                self.grad_data = self.shared_buffer
            else:
                if self.ddp_config.use_distributed_optimizer:
                    self.param_data = torch.zeros(
                        self.numel,
                        dtype=self.param_dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                self.grad_data = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )

        self.grad_data_size = 0
        self.param_data_size = 0
        self.param_data_cpu = None

        bucket_params = []
        bucket_start_index = 0
        cur_bucket_id = 0
        for param in params[::-1]:
            param_start_index, param_end_index, bucket_id = self.param_index_map[param]
            if not self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
                if self.param_data is not None:
                    new_param_data = self._get(
                        param.data.shape, param_start_index, buffer_type=BufferType.PARAM
                    )
                    if is_float8tensor(param):
                        modify_underlying_storage(param, new_param_data)
                    else:
                        old_param_data = param.data
                        param.data = new_param_data
                        assert old_param_data._base is None
                        param.data.detach().copy_(old_param_data)
                        del old_param_data

            param.main_grad = self._get(
                param.data.shape, param_start_index, buffer_type=BufferType.GRAD
            )
            if bucket_id != cur_bucket_id:
                bucket_end_index = _pad_end_of_bucket_if_needed(param_start_index)
                self.buckets.append(
                    self._new_bucket(
                        bucket_params=bucket_params,
                        start_index=bucket_start_index,
                        end_index=bucket_end_index,
                        numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                        bucket_id=cur_bucket_id,
                    )
                )
                bucket_start_index = bucket_end_index
                bucket_params = []
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            bucket_params.append(param)

        if len(bucket_params) > 0:
            bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)
            self.buckets.append(
                self._new_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_start_index,
                    end_index=bucket_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                )
            )

        log_strs = []
        log_strs.append(
            f"Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}"
        )
        for index, bucket in enumerate(self.buckets):
            numel = 0
            for param in bucket.params:
                numel += param.data.nelement()
            log_strs.append(
                f"Params for bucket {index + 1} ({numel} elements, "
                f"{bucket.grad_data.nelement()} padded size):"
            )
            for param in bucket.params:
                log_strs.append(f"\t{param_to_name[param]}")
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            "\n".join(log_strs),
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
        )

    def scale_gradients(self, scaling_factor: float) -> None:
        self.grad_data *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, "Requested tensor is out of buffer range"
        if buffer_type == BufferType.PARAM:
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:
            buffer_tensor = self.grad_data[start_index:end_index]
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def _new_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
    ) -> _ParamAndGradBucket:
        if self.ddp_config.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        bucketed_param_data = None
        if self.param_data is not None:
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
        )
        bucket = _ParamAndGradBucket(
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            gradient_scaling_factor=self.gradient_scaling_factor,
            bucket_id=bucket_id,
        )
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

        return bucket

    def reset(self):
        self.grad_data.zero_()

    def offload_to_cpu(self, move_params: bool = True, move_grads: bool = True) -> None:
        if move_grads and self.grad_data is not None and self.grad_data.storage().size() > 0:
            self.grad_data_size = self.grad_data.storage().size()
            self.grad_data.storage().resize_(0)
        if move_params and self.param_data is not None and self.param_data.storage().size() > 0:
            self.param_data_size = self.param_data.storage().size()
            if self.param_data_cpu is not None:
                self.param_data_cpu.copy_(self.param_data, non_blocking=True)
            else:
                self.param_data_cpu = self.param_data.cpu().pin_memory()
            self.param_data.storage().resize_(0)

    def reload_from_cpu(self, move_params: bool = True, move_grads: bool = True):
        if (
            move_params
            and self.param_data is not None
            and self.param_data_cpu is not None
            and self.param_data.storage().size() == 0
        ):
            self.param_data.storage().resize_(self.param_data_size)
            self.param_data.copy_(self.param_data_cpu, non_blocking=True)
        if move_grads and self.grad_data is not None and self.grad_data_size > 0:
            self.grad_data.storage().resize_(self.grad_data_size)
            self.grad_data.zero_()
            self.grad_data_size = 0


def partition_buckets(
    buffers: List[_ParamAndGradBuffer], force_single_bucket_group: bool = False
) -> List[_ParamAndGradBucketGroup]:
    """
    Automatically regroup the buckets of input buffers and return a list of bucket groups.
    """

    if len(buffers) == 0:
        return []

    dtype_to_buffer_map = {}
    for buffer in buffers:
        dtype = buffer.param_dtype
        assert dtype not in dtype_to_buffer_map
        dtype_to_buffer_map[dtype] = buffer

    if force_single_bucket_group:
        buckets = []
        ddp_config = buffers[0].ddp_config
        data_parallel_group = buffers[0].data_parallel_group
        data_parallel_world_size = buffers[0].data_parallel_world_size
        for buffer in buffers:
            assert ddp_config == buffer.ddp_config
            assert data_parallel_group == buffer.data_parallel_group
            assert data_parallel_world_size == buffer.data_parallel_world_size
            buckets.extend(buffer.buckets)

        bucket_group = _ParamAndGradBucketGroup(
            buckets, ddp_config, data_parallel_group, data_parallel_world_size
        )
        return [bucket_group]

    if torch.uint8 not in dtype_to_buffer_map:
        bucket_groups = []
        for buffer in buffers:
            for bucket in buffer.buckets:
                bucket_groups.append(
                    _ParamAndGradBucketGroup(
                        [bucket],
                        buffer.ddp_config,
                        buffer.data_parallel_group,
                        buffer.data_parallel_world_size,
                    )
                )
        return bucket_groups
    else:
        non_fp8_buckets = []
        for buffer in buffers:
            if buffer.param_dtype != torch.uint8:
                for bucket in buffer.buckets:
                    non_fp8_buckets.append(bucket)

        bucket_groups = []
        fp8_buffer = dtype_to_buffer_map[torch.uint8]
        for bucket in fp8_buffer.buckets:
            if len(bucket_groups) == len(fp8_buffer.buckets) - 1:
                group_buckets = [bucket] + non_fp8_buckets
            else:
                group_buckets = [bucket]
            bucket_groups.append(
                _ParamAndGradBucketGroup(
                    group_buckets,
                    buffer.ddp_config,
                    buffer.data_parallel_group,
                    buffer.data_parallel_world_size,
                )
            )
        return bucket_groups
