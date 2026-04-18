import os
import re
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Set, List, Dict


FP8_MAX_E5M2 = 57344.0


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def native_fp8_allreduce_q_inplace_(
    q_fp8: torch.Tensor,
    group: Optional[dist.ProcessGroup],
):
    if group is None:
        return None
    if not q_fp8.is_cuda:
        raise RuntimeError("expects CUDA tensor")
    if q_fp8.dtype != torch.float8_e5m2:
        raise RuntimeError(f"Q must be torch.float8_e5m2, got {q_fp8.dtype}")
    if not q_fp8.is_contiguous():
        q_fp8 = q_fp8.contiguous()
    work = dist.all_reduce(q_fp8, op=dist.ReduceOp.SUM, group=group, async_op=True)
    return work


def fp8_layer_scale(tensor: torch.Tensor, clip_factor: float = 1.0):
    EPS = 1e-8
    amax = tensor.abs().amax()
    scaled_max = amax * clip_factor
    scale = (scaled_max / FP8_MAX_E5M2).clamp(min=EPS)
    return scale


def fp8_encode_layer(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e5m2,
) -> torch.Tensor:
    return (tensor / scale).to(fp8_dtype)


def fp8_decode_layer(
    q_fp8: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return q_fp8.to(out_dtype) * scale


def _split_infos_balanced(module_infos: List[dict], num_buckets: int) -> List[List[dict]]:
    if len(module_infos) == 0:
        return []
    num_buckets = max(1, min(num_buckets, len(module_infos)))

    buckets = [[] for _ in range(num_buckets)]
    bucket_sizes = [0 for _ in range(num_buckets)]

    infos_sorted = sorted(module_infos, key=lambda x: int(x["numel"]), reverse=True)
    for info in infos_sorted:
        idx = min(range(num_buckets), key=lambda i: bucket_sizes[i])
        buckets[idx].append(info)
        bucket_sizes[idx] += int(info["numel"])

    return [b for b in buckets if len(b) > 0]


class BaselineFP8Manager:
    def __init__(
        self,
        model: nn.Module,
        comm_group: Optional[dist.ProcessGroup],
        warmup_steps: int = 0,
        allowed_module_names: Optional[Set[str]] = None,
        clip_factor: float = 1.0,
        fp8_dtype: torch.dtype = torch.float8_e5m2,
        comm_num_buckets: int = 1,
    ):
        self.model = model
        self.group = comm_group
        self.warmup_steps = warmup_steps
        self.allowed_module_names = allowed_module_names
        self.clip_factor = clip_factor
        self.fp8_dtype = fp8_dtype
        self.comm_num_buckets = max(1, int(comm_num_buckets))
        self._tag_callback = None

        self._module_name_map: Dict[int, str] = {}
        for name, module in self.model.named_modules():
            self._module_name_map[id(module)] = name

        self._avg_limit = 100
        self._counted_steps = 0

        self._layer_stats: Dict[str, Dict[str, float]] = {}
        self._fc_pattern = re.compile(r"module\.decoder\.layers\.(\d+)\.mlp\.(linear_fc1|linear_fc2)$")

    def set_tag_callback(self, callback):
        self._tag_callback = callback

    def update_tag(self, tag: str):
        if self._tag_callback:
            self._tag_callback(tag)

    def step_and_update(self, global_step: int):
        pass

    def _param_grad(self, p: torch.nn.Parameter):
        if hasattr(p, "main_grad") and (p.main_grad is not None):
            return p.main_grad
        return getattr(p, "grad", None)

    def _write_back_grad(self, p: torch.nn.Parameter, new_grad: torch.Tensor):
        if hasattr(p, "main_grad") and (p.main_grad is not None):
            p.main_grad.copy_(new_grad)
        elif getattr(p, "grad", None) is not None:
            p.grad.copy_(new_grad)

    def _module_name(self, module: nn.Module) -> str:
        return self._module_name_map.get(id(module), "")

    def _iter_all_modules_with_direct_params(self, model: nn.Module):
        for _, module in model.named_modules():
            has_direct_param = False
            for _, p in module.named_parameters(recurse=False):
                g = self._param_grad(p)
                if g is None:
                    continue
                if not g.is_cuda:
                    continue
                if getattr(g, "is_sparse", False):
                    continue
                has_direct_param = True
                break
            if has_direct_param:
                yield module

    def _collect_module_grads(
        self, module: nn.Module
    ) -> Tuple[List[torch.nn.Parameter], List[torch.Tensor], List[torch.Size], int]:
        params: List[torch.nn.Parameter] = []
        grads: List[torch.Tensor] = []
        shapes: List[torch.Size] = []
        total_numel = 0

        for _, p in module.named_parameters(recurse=False):
            g = self._param_grad(p)
            if g is None:
                continue
            if not g.is_cuda:
                continue
            if getattr(g, "is_sparse", False):
                continue
            params.append(p)
            grads.append(g)
            shapes.append(g.shape)
            total_numel += g.numel()

        return params, grads, shapes, total_numel

    def _sync_scale_bucket(self, infos: List[dict], group):
        if len(infos) == 0:
            return
        scale_bucket = torch.stack([info["scale"].reshape(1) for info in infos], dim=0)
        if group is not None:
            dist.all_reduce(scale_bucket, op=dist.ReduceOp.MAX, group=group)
        for i, info in enumerate(infos):
            info["scale"] = scale_bucket[i].reshape(())

    def _parse_fc_layer(self, module_name: str) -> Optional[str]:
        m = self._fc_pattern.match(module_name)
        if m is None:
            return None
        layer_id = int(m.group(1))
        fc_kind = m.group(2)
        if fc_kind == "linear_fc1":
            return f"layer{layer_id}.fc1"
        if fc_kind == "linear_fc2":
            return f"layer{layer_id}.fc2"
        return None

    def _ensure_layer_stat(self, key: str):
        if key not in self._layer_stats:
            self._layer_stats[key] = {
                "upper_sum": 0.0,
                "lower_sum": 0.0,
                "max_upper": 0.0,
                "max_lower": 0.0,
                "steps_seen": 0.0,
            }

    def _empty_step_stats(self):
        stats = {}
        for i in range(32):
            stats[f"layer{i}.fc1"] = {"upper_hits": 0, "lower_hits": 0, "numel": 0}
            stats[f"layer{i}.fc2"] = {"upper_hits": 0, "lower_hits": 0, "numel": 0}
        stats["fc1_all"] = {"upper_hits": 0, "lower_hits": 0, "numel": 0}
        stats["fc2_all"] = {"upper_hits": 0, "lower_hits": 0, "numel": 0}
        return stats

    def _ratio_from_stat(self, stat: dict) -> Tuple[float, float]:
        if stat["numel"] == 0:
            return 0.0, 0.0
        upper_ratio = float(stat["upper_hits"]) / float(stat["numel"])
        lower_ratio = float(stat["lower_hits"]) / float(stat["numel"])
        return upper_ratio, lower_ratio

    def _accumulate_boundary_stats(self, q_fp8_flat: torch.Tensor, info: dict, step_stats: dict):
        layer_fc_key = info["layer_fc_key"]
        if layer_fc_key is None:
            return

        q_f32 = q_fp8_flat.to(torch.float32)
        upper_hits = (q_f32 == FP8_MAX_E5M2).sum().item()
        lower_hits = (q_f32 == -FP8_MAX_E5M2).sum().item()
        numel = q_f32.numel()

        step_stats[layer_fc_key]["upper_hits"] += int(upper_hits)
        step_stats[layer_fc_key]["lower_hits"] += int(lower_hits)
        step_stats[layer_fc_key]["numel"] += int(numel)

        if layer_fc_key.endswith(".fc1"):
            agg_key = "fc1_all"
        else:
            agg_key = "fc2_all"

        step_stats[agg_key]["upper_hits"] += int(upper_hits)
        step_stats[agg_key]["lower_hits"] += int(lower_hits)
        step_stats[agg_key]["numel"] += int(numel)

    def _print_step_stats(self, global_step: int, step_stats: dict):
        if not _rank0() or not _env_flag_enabled("FP8_MLP_BOUNDARY_STEP_LOG", default=False):
            return

        fc1_u, fc1_l = self._ratio_from_stat(step_stats["fc1_all"])
        fc2_u, fc2_l = self._ratio_from_stat(step_stats["fc2_all"])

        print(
            "[FP8_MLP_BOUNDARY_STEP] "
            f"step={global_step} "
            f"fc1_upper={fc1_u:.8e} "
            f"fc1_lower={fc1_l:.8e} "
            f"fc2_upper={fc2_u:.8e} "
            f"fc2_lower={fc2_l:.8e}",
            flush=True,
        )

    def _update_running_avg_and_maybe_print_table(self, step_stats: dict):
        self._counted_steps += 1

        for i in range(32):
            for kind in ("fc1", "fc2"):
                key = f"layer{i}.{kind}"
                upper_ratio, lower_ratio = self._ratio_from_stat(step_stats[key])

                self._ensure_layer_stat(key)
                self._layer_stats[key]["upper_sum"] += upper_ratio
                self._layer_stats[key]["lower_sum"] += lower_ratio
                self._layer_stats[key]["max_upper"] = max(self._layer_stats[key]["max_upper"], upper_ratio)
                self._layer_stats[key]["max_lower"] = max(self._layer_stats[key]["max_lower"], lower_ratio)
                self._layer_stats[key]["steps_seen"] += 1.0

        if self._counted_steps == self._avg_limit and _rank0():
            rows = []
            for i in range(32):
                for kind in ("fc1", "fc2"):
                    key = f"layer{i}.{kind}"
                    s = self._layer_stats[key]
                    n = max(1.0, s["steps_seen"])
                    upper_avg = s["upper_sum"] / n
                    lower_avg = s["lower_sum"] / n
                    total_avg = upper_avg + lower_avg
                    max_upper = s["max_upper"]
                    max_lower = s["max_lower"]
                    max_total = max_upper + max_lower
                    rows.append({
                        "name": key,
                        "upper_avg": upper_avg,
                        "lower_avg": lower_avg,
                        "total_avg": total_avg,
                        "max_upper": max_upper,
                        "max_lower": max_lower,
                        "max_total": max_total,
                    })

            rows.sort(key=lambda x: x["total_avg"], reverse=True)

            print(
                "[FP8_MLP_LAYER_BOUNDARY_TABLE_AVG100] "
                "sorted_by=total_avg_desc",
                flush=True,
            )
            print(
                "rank | layer       | upper_avg    | lower_avg    | total_avg    | max_upper    | max_lower    | max_total",
                flush=True,
            )
            for idx, row in enumerate(rows, start=1):
                print(
                    f"{idx:>4d} | "
                    f"{row['name']:<11s} | "
                    f"{row['upper_avg']:.8e} | "
                    f"{row['lower_avg']:.8e} | "
                    f"{row['total_avg']:.8e} | "
                    f"{row['max_upper']:.8e} | "
                    f"{row['max_lower']:.8e} | "
                    f"{row['max_total']:.8e}",
                    flush=True,
                )

            top_k = 10
            print(
                f"[FP8_MLP_LAYER_BOUNDARY_TOP{top_k}_AVG100]",
                flush=True,
            )
            for idx, row in enumerate(rows[:top_k], start=1):
                print(
                    f"top{idx} "
                    f"name={row['name']} "
                    f"upper_avg={row['upper_avg']:.8e} "
                    f"lower_avg={row['lower_avg']:.8e} "
                    f"total_avg={row['total_avg']:.8e} "
                    f"max_upper={row['max_upper']:.8e} "
                    f"max_lower={row['max_lower']:.8e} "
                    f"max_total={row['max_total']:.8e}",
                    flush=True,
                )

    def _sync_q_bucket(
        self,
        infos: List[dict],
        group,
        world_size: int,
        synced_param_ids: Set[int],
        step_stats: dict,
    ) -> bool:
        if len(infos) == 0:
            return False

        q_chunks = []
        q_numels = []

        for info in infos:
            q = fp8_encode_layer(info["g_flat"], info["scale"], fp8_dtype=self.fp8_dtype)
            q = q.contiguous().view(-1)

            self._accumulate_boundary_stats(q, info, step_stats)

            q_chunks.append(q)
            q_numels.append(q.numel())

        q_bucket = torch.cat(q_chunks, dim=0).view(1, -1)

        work = None
        if group is not None:
            work = native_fp8_allreduce_q_inplace_(q_bucket, group=group)
            if work is not None:
                work.wait()

        q_bucket_flat = q_bucket.view(-1)
        q_offset = 0
        did_apply = False

        for info, qn in zip(infos, q_numels):
            q_part = q_bucket_flat[q_offset:q_offset + qn].view(1, -1)
            gsum = fp8_decode_layer(q_part, info["scale"], out_dtype=torch.float32)
            gavg = (gsum / float(world_size)).view(-1)

            offset = 0
            for p, g, shape in zip(info["params"], info["grads"], info["shapes"]):
                n = g.numel()
                gnew = gavg[offset:offset + n].view(shape).to(dtype=g.dtype)
                self._write_back_grad(p, gnew)
                synced_param_ids.add(id(p))
                offset += n

            q_offset += qn
            did_apply = True

        return did_apply

    def sync_gradients_fp8(self, model: nn.Module, global_step: int) -> Tuple[bool, Set[int]]:
        synced_param_ids: Set[int] = set()

        if global_step < self.warmup_steps:
            self.update_tag("BASELINE")
            return False, synced_param_ids

        group = self.group
        world_size = dist.get_world_size(group=group) if group is not None else 1
        did_apply = False

        module_infos = []

        for module in self._iter_all_modules_with_direct_params(model):
            module_name = self._module_name(module)

            if self.allowed_module_names is not None and module_name not in self.allowed_module_names:
                continue

            params, grads, shapes, module_numel = self._collect_module_grads(module)
            if module_numel == 0:
                continue

            flat_grads = [g.detach().to(torch.float32).contiguous().view(-1) for g in grads]
            g_flat = torch.cat(flat_grads, dim=0).view(1, -1)
            local_scale = fp8_layer_scale(g_flat, clip_factor=self.clip_factor)
            layer_fc_key = self._parse_fc_layer(module_name)

            module_infos.append({
                "module_name": module_name,
                "layer_fc_key": layer_fc_key,
                "params": params,
                "grads": grads,
                "shapes": shapes,
                "g_flat": g_flat,
                "scale": local_scale,
                "numel": g_flat.numel(),
            })

        if len(module_infos) == 0:
            self.update_tag("BASELINE_WAIT")
            return False, synced_param_ids

        step_stats = self._empty_step_stats()
        buckets = _split_infos_balanced(module_infos, self.comm_num_buckets)

        for bucket_infos in buckets:
            self._sync_scale_bucket(bucket_infos, group)
            bucket_applied = self._sync_q_bucket(
                bucket_infos,
                group,
                world_size,
                synced_param_ids,
                step_stats,
            )
            did_apply = did_apply or bucket_applied

        self._print_step_stats(global_step, step_stats)
        self._update_running_avg_and_maybe_print_table(step_stats)

        self.update_tag("FP8_BASELINE_ALL" if did_apply else "BASELINE_WAIT")
        return did_apply, synced_param_ids


def precondition_gradients_with_ef(model: nn.Module, manager: BaselineFP8Manager, global_step: int = None):
    return manager.sync_gradients_fp8(model, global_step)
