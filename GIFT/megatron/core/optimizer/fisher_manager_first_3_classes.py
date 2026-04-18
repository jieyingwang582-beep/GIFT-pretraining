import time
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

try:
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
    _TE_TYPES = (TEColumnParallelLinear, TERowParallelLinear)
except Exception:
    _TE_TYPES = tuple()

_LINEAR_TYPES = (nn.Linear, ColumnParallelLinear, RowParallelLinear) + _TE_TYPES

_SKIP_DP_SYNC_THIS_STEP = False
_IG_LOWRANK_R_A = 32

_IG_TARGET_LAYER_NAMES = {
    "module.decoder.layers.17.mlp.linear_fc2",
    "module.decoder.layers.20.mlp.linear_fc2",
    "module.decoder.layers.14.mlp.linear_fc2",
    "module.decoder.layers.30.mlp.linear_fc2",
    "module.decoder.layers.13.mlp.linear_fc2",

    "module.decoder.layers.2.mlp.linear_fc2",
    "module.decoder.layers.12.mlp.linear_fc2",
    "module.decoder.layers.7.mlp.linear_fc2",
    "module.decoder.layers.15.mlp.linear_fc2",
    "module.decoder.layers.0.mlp.linear_fc2",
    "module.decoder.layers.4.mlp.linear_fc2",
    "module.decoder.layers.9.mlp.linear_fc2",
    "module.decoder.layers.24.mlp.linear_fc2",

    "module.decoder.layers.27.mlp.linear_fc1",
    "module.decoder.layers.19.mlp.linear_fc2",
    "module.decoder.layers.23.mlp.linear_fc2",
    "module.decoder.layers.5.mlp.linear_fc2",
    "module.decoder.layers.29.mlp.linear_fc2",
    "module.decoder.layers.0.mlp.linear_fc1",
    "module.decoder.layers.11.mlp.linear_fc2",
    "module.decoder.layers.18.mlp.linear_fc2",
    "module.decoder.layers.21.mlp.linear_fc2",
    "module.decoder.layers.27.mlp.linear_fc2",
    "module.decoder.layers.23.mlp.linear_fc1",
    "module.decoder.layers.25.mlp.linear_fc1",
    "module.decoder.layers.29.mlp.linear_fc1",
}


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class _CudaTimer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self._cpu_t0 = None
        self._cpu_t1 = None
        self._start = None
        self._end = None

    def __enter__(self):
        if self.enabled:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._cpu_t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self._end.record()
        else:
            self._cpu_t1 = time.perf_counter()

    def ms(self) -> float:
        if self.enabled:
            torch.cuda.synchronize()
            return float(self._start.elapsed_time(self._end))
        return float((self._cpu_t1 - self._cpu_t0) * 1000.0)


class LayerStats:
    def __init__(self, name: str, layer_id: int, a_dim: int):
        self.name = name
        self.layer_id = layer_id
        self.a_dim = a_dim
        self.A: Optional[torch.Tensor] = None
        self.A_accum: Optional[torch.Tensor] = None
        self.A_count: int = 0


class LayerFactors:
    def __init__(self, L_A: torch.Tensor):
        self.L_A = L_A


class FisherManager:
    def __init__(
        self,
        model: nn.Module,
        stats_decay: float = 0.95,
        damping: float = 0.001,
        update_freq: int = 50,
        warmup_steps: int = 0,
        factor_decay: float = 0.95,
        comm_group=None,
        grow_start: int = 200000,
        grow_interval: int = 10000,
        grow_mult: float = 1.0,
        max_update_freq: int = 50,
    ):
        self.model = model
        self.stats_decay = stats_decay
        self.damping = damping
        self.update_freq = update_freq
        self.warmup_steps = warmup_steps
        self.factor_decay = factor_decay
        self.group = comm_group
        self.grow_start = grow_start
        self.grow_interval = grow_interval
        self.grow_mult = grow_mult
        self.max_update_freq = max_update_freq

        self.stats: Dict[str, LayerStats] = {}
        self.factors: Dict[str, LayerFactors] = {}
        self.error_buffers_u: Dict[str, torch.Tensor] = {}

        self.rank = dist.get_rank(group=self.group) if (_dist_ready() and self.group is not None) else (dist.get_rank() if _dist_ready() else 0)
        self.world_size = dist.get_world_size(group=self.group) if (_dist_ready() and self.group is not None) else (dist.get_world_size() if _dist_ready() else 1)

        self._hook_step = 1
        self.fp8_sync_freq = 50

        self._modules = dict(model.named_modules())
        self._hook_handles = []
        self._hooks_registered = False
        self._build_layer_stats()

    def _is_collect_step(self, step: int) -> bool:
        return step >= 1 and ((step - 1) % self.update_freq == 0)

    def _is_fp8_sync_step(self, step: int) -> bool:
        return step >= 1 and ((step - 1) % self.fp8_sync_freq == 0)

    def _is_target_ig_module(self, name: str, module: nn.Module) -> bool:
        lname = name.lower()
        if "embedding" in lname:
            return False
        if not isinstance(module, _LINEAR_TYPES):
            return False
        w = getattr(module, "weight", None)
        if w is None or getattr(w, "ndim", None) != 2:
            return False
        return name in _IG_TARGET_LAYER_NAMES

    def _build_layer_stats(self):
        layer_id = 0
        for name, module in self.model.named_modules():
            if not self._is_target_ig_module(name, module):
                continue
            w = getattr(module, "weight", None)
            if w is None or w.ndim != 2:
                continue
            a_dim = int(w.shape[1])
            self.stats[name] = LayerStats(name=name, layer_id=layer_id, a_dim=a_dim)
            layer_id += 1

        if self.rank == 0:
            ig_names = list(self.stats.keys())

            all_linear_names = []
            all_mlp_linear_names = []

            for name, module in self.model.named_modules():
                lname = name.lower()
                if "embedding" in lname:
                    continue
                if not isinstance(module, _LINEAR_TYPES):
                    continue
                w = getattr(module, "weight", None)
                if w is None or getattr(w, "ndim", None) != 2:
                    continue
                all_linear_names.append(name)
                if "mlp" in lname:
                    all_mlp_linear_names.append(name)

            ig_cnt = len(ig_names)
            all_linear_cnt = len(all_linear_names)
            all_mlp_linear_cnt = len(all_mlp_linear_names)

            pct_all_linear = 100.0 * ig_cnt / max(1, all_linear_cnt)
            pct_mlp_linear = 100.0 * ig_cnt / max(1, all_mlp_linear_cnt)

            print("=" * 80, flush=True)
            print("[IG_LAYER_SUMMARY]", flush=True)
            print(f"IG layers: {ig_cnt}", flush=True)
            print(f"All non-embedding 2D-weight layers: {all_linear_cnt}", flush=True)
            print(f"All MLP 2D-weight layers: {all_mlp_linear_cnt}", flush=True)
            print(f"IG / all such layers = {pct_all_linear:.2f}%", flush=True)
            print(f"IG / MLP such layers = {pct_mlp_linear:.2f}%", flush=True)
            print("[IG_LAYER_NAMES]", flush=True)
            for n in ig_names:
                print(n, flush=True)
            print("=" * 80, flush=True)

    def _register_hooks(self):
        if self._hooks_registered:
            return
        self._hook_handles = []
        for name, module in self.model.named_modules():
            if name not in self.stats:
                continue
            h = module.register_forward_pre_hook(self._get_input_hook(name))
            self._hook_handles.append(h)
        self._hooks_registered = True

    def _remove_hooks(self):
        if not self._hooks_registered:
            return
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []
        self._hooks_registered = False

    def _get_input_hook(self, name):
        def hook(module, inputs):
            st = self.stats[name]
            if not self._is_collect_step(self._hook_step):
                return
            with torch.no_grad():
                x = inputs[0].detach()
                if x.dim() > 2:
                    x = x.view(-1, x.size(-1))
                if x.numel() == 0:
                    return
                if x.dtype != torch.float32:
                    x = x.float()

                update = (x.transpose(0, 1) @ x) / float(max(1, x.shape[0]))
                if st.A_accum is None:
                    st.A_accum = update.clone()
                else:
                    st.A_accum.add_(update)
                st.A_count += 1
        return hook

    def _flush_accum_to_stats(self):
        for st in self.stats.values():
            if st.A_accum is not None and st.A_count > 0:
                A_mean = st.A_accum / float(st.A_count)
                if st.A is None:
                    st.A = A_mean.clone()
                else:
                    st.A.mul_(self.stats_decay).add_(A_mean, alpha=1.0 - self.stats_decay)
            st.A_accum = None
            st.A_count = 0

    def _compute_factors(self):
        prepared = []
        for name in list(self.stats.keys()):
            st = self.stats[name]
            if st.A is None:
                if name in self.factors:
                    del self.factors[name]
                continue
            prepared.append((name, st.A.clone()))

        if not prepared:
            return

        if self.group is not None:
            flat = torch.cat([A.contiguous().view(-1) for _, A in prepared], dim=0)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.group)
            flat.div_(float(self.world_size))

            offset = 0
            reduced = []
            for name, A in prepared:
                n = A.numel()
                A_red = flat[offset:offset + n].view_as(A)
                offset += n
                reduced.append((name, A_red))
        else:
            reduced = prepared

        for name, A_full in reduced:
            A_sym = 0.5 * (A_full + A_full.transpose(-1, -2))
            A_hat = _build_lowrank_plus_diag(A_sym, _IG_LOWRANK_R_A, self.damping)
            L_A = _chol_with_jitter(A_hat, self.damping)

            if self.factor_decay is not None and 0.0 < self.factor_decay < 1.0 and name in self.factors:
                old = self.factors[name]
                L_A = old.L_A.mul(self.factor_decay).add(L_A, alpha=1.0 - self.factor_decay)

            self.factors[name] = LayerFactors(L_A.contiguous())

    def step_and_update(self, global_step: int):
        if global_step < self.warmup_steps:
            return
        if self._is_collect_step(self._hook_step):
            self._flush_accum_to_stats()
            self._compute_factors()

    def set_tag_callback(self, callback):
        self._tag_callback = callback

    def update_tag(self, tag):
        if hasattr(self, "_tag_callback") and self._tag_callback:
            self._tag_callback(tag)

    def state_dict(self):
        sd = {
            "stats_decay": self.stats_decay,
            "damping": self.damping,
            "update_freq": self.update_freq,
            "warmup_steps": self.warmup_steps,
            "factor_decay": self.factor_decay,
            "layer_stats": {},
            "layer_factors": {},
            "error_buffers_u": {},
        }
        for name, st in self.stats.items():
            sd["layer_stats"][name] = {
                "A": None if st.A is None else st.A.detach().cpu(),
            }
        for name, fac in self.factors.items():
            sd["layer_factors"][name] = {
                "L_A": fac.L_A.detach().cpu(),
            }
        for name, u in self.error_buffers_u.items():
            sd["error_buffers_u"][name] = u.detach().cpu()
        return sd

    def load_state_dict(self, sd):
        for name, d in sd.get("layer_stats", {}).items():
            if name not in self.stats:
                continue
            self.stats[name].A = None if d["A"] is None else d["A"].to(device=torch.cuda.current_device(), dtype=torch.float32)

        self.factors = {}
        for name, d in sd.get("layer_factors", {}).items():
            if name not in self.stats:
                continue
            L_A = d["L_A"].to(device=torch.cuda.current_device(), dtype=torch.float32).contiguous()
            self.factors[name] = LayerFactors(L_A)

        self.error_buffers_u = {}
        for name, u in sd.get("error_buffers_u", {}).items():
            self.error_buffers_u[name] = u.to(device=torch.cuda.current_device(), dtype=torch.float32)


def _get_weight_and_grad(module):
    w = getattr(module, "weight", None)
    if w is None:
        return None, None, None
    if hasattr(w, "main_grad") and (w.main_grad is not None):
        return w, w.main_grad, "main_grad"
    if getattr(w, "grad", None) is not None:
        return w, w.grad, "grad"
    return w, None, None


def _write_back_grad(weight, grad_attr: str, new_grad: torch.Tensor):
    if grad_attr == "main_grad":
        weight.main_grad.copy_(new_grad)
    elif grad_attr == "grad":
        weight.grad.copy_(new_grad)


def _get_param_grad(p: torch.nn.Parameter):
    if hasattr(p, "main_grad") and (p.main_grad is not None):
        return p.main_grad, "main_grad"
    if getattr(p, "grad", None) is not None:
        return p.grad, "grad"
    return None, None


def native_fp8_allreduce_q_async_(q_fp8: torch.Tensor, group: Optional[dist.ProcessGroup]):
    if group is None:
        return None
    if not q_fp8.is_cuda:
        raise RuntimeError("expects CUDA tensor")
    if q_fp8.dtype != torch.float8_e5m2:
        raise RuntimeError(f"Q must be torch.float8_e5m2, got {q_fp8.dtype}")
    if not q_fp8.is_contiguous():
        q_fp8 = q_fp8.contiguous()
    return dist.all_reduce(q_fp8, op=dist.ReduceOp.SUM, group=group, async_op=True)


def _chol_with_jitter(M: torch.Tensor, damping: float) -> torch.Tensor:
    I = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
    try:
        return torch.linalg.cholesky(M + damping * I)
    except RuntimeError:
        eps = max(float(damping), 1e-6)
        return torch.linalg.cholesky(M + eps * I)


def _build_lowrank_plus_diag(M: torch.Tensor, rank: int, damping: float) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(M)
    evals = evals.to(torch.float32)
    evecs = evecs.to(torch.float32)

    topk = min(rank, M.shape[0])
    idx = torch.argsort(evals, descending=True)[:topk]
    lam = torch.clamp(evals[idx], min=0.0)
    U = evecs[:, idx]

    lowrank = (U * lam.view(1, -1)) @ U.transpose(-1, -2)
    resid_diag = torch.diagonal(M - lowrank).to(torch.float32)
    resid_diag = torch.clamp(resid_diag, min=damping)

    out = lowrank + torch.diag(resid_diag)
    return 0.5 * (out + out.transpose(-1, -2))


def fp8_scale(tensor: torch.Tensor, clip_factor: float = 1.0):
    fp8_max = 57344.0
    eps = 1e-8
    amax = tensor.abs().amax()
    scaled_max = amax * clip_factor
    return (scaled_max / fp8_max).clamp(min=eps)


def fp8_encode_with_scale(tensor: torch.Tensor, scale: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e5m2) -> torch.Tensor:
    return (tensor / scale).to(fp8_dtype)


def fp8_decode_with_scale(fp8_tensor: torch.Tensor, scale: torch.Tensor, orig_shape) -> torch.Tensor:
    out = fp8_tensor.to(torch.float32) * scale
    rows, cols = orig_shape
    return out[:rows, :cols]


def _run_right_lowrank_whiten_stage(entries):
    if not entries:
        return
    for e in entries:
        e["u"] = torch.linalg.solve_triangular(
            e["L_A_f"],
            e["grad_f32"].transpose(-1, -2),
            upper=False
        ).transpose(-1, -2)


def _run_right_lowrank_mapback_stage(entries):
    if not entries:
        return
    for e in entries:
        e["grad_after_f32"] = e["u_avg"] @ e["L_A_f"].transpose(-1, -2)


def _run_scale_allreduce(entries: List[dict], group):
    if group is None or not entries:
        return
    flat = torch.cat([e["scale"].contiguous().view(-1) for e in entries], dim=0)
    dist.all_reduce(flat, op=dist.ReduceOp.MAX, group=group)
    offset = 0
    for e in entries:
        n = e["scale"].numel()
        e["scale"] = flat[offset:offset + n].view_as(e["scale"])
        offset += n


def _launch_async_q_allreduce(entries: List[dict], group):
    if not entries:
        return None

    flat = torch.cat([e["q_fp8"].contiguous().view(-1) for e in entries], dim=0)
    work = native_fp8_allreduce_q_async_(flat, group=group) if group is not None else None
    q_numels = [e["q_fp8"].numel() for e in entries]
    return {"entries": entries, "flat": flat, "q_numels": q_numels, "work": work}


def _finalize_async_q_bucket(state: dict, group, world_size: int):
    if state is None:
        return []

    work = state.get("work", None)
    if work is not None:
        work.wait()

    entries = state["entries"]
    flat = state["flat"]
    q_numels = state["q_numels"]

    offset = 0
    for e, n in zip(entries, q_numels):
        e["q_fp8_after_comm"] = flat[offset:offset + n].view_as(e["q_fp8"])
        offset += n

    for item in entries:
        u_sum = fp8_decode_with_scale(item["q_fp8_after_comm"], item["scale"], item["u_tilde_shape"])
        item["u_avg"] = u_sum / float(world_size) if group is not None else u_sum

    return entries


def _encode_entries(entries: List[dict], manager: FisherManager):
    for item in entries:
        name = item["name"]
        u = item["u"]
        err_u = manager.error_buffers_u.get(name, None)
        u_tilde = u if err_u is None else (u + err_u)

        scale = fp8_scale(u_tilde, clip_factor=1.0)
        q_fp8 = fp8_encode_with_scale(u_tilde, scale, fp8_dtype=torch.float8_e5m2)
        u_deq_local = fp8_decode_with_scale(q_fp8, scale, u_tilde.shape)
        manager.error_buffers_u[name] = (u_tilde - u_deq_local).detach()

        item["scale"] = scale
        item["u_tilde_shape"] = u_tilde.shape
        item["q_fp8"] = q_fp8


def precondition_gradients_with_ef(model: nn.Module, manager: FisherManager, global_step: int = None):
    synced_param_ids = set()

    if (global_step is not None) and (global_step < manager.warmup_steps):
        manager.update_tag("BASELINE")
        return False, synced_param_ids

    if not manager.factors:
        manager.update_tag("BASELINE")
        return False, synced_param_ids

    group = manager.group
    world_size = dist.get_world_size(group=group) if group is not None else 1
    do_scale_sync = True if (global_step is None) else manager._is_fp8_sync_step(global_step)

    modules = manager._modules
    prepared = []

    for name, factors in manager.factors.items():
        module = modules.get(name, None)
        if module is None:
            continue

        weight, grad, grad_attr = _get_weight_and_grad(module)
        if grad is None:
            continue

        grad_det = grad.detach()
        if grad_det.ndim != 2:
            continue

        original_shape = grad.shape
        g_dtype = grad.dtype
        grad_f32 = grad_det if (grad_det.dtype == torch.float32 and grad_det.is_contiguous()) else grad_det.to(torch.float32).contiguous()

        L_A = factors.L_A
        L_A_f = L_A if (L_A.device == grad_f32.device and L_A.dtype == torch.float32 and L_A.is_contiguous()) else L_A.to(dtype=torch.float32, device=grad_f32.device).contiguous()

        if L_A_f.shape[0] != grad_f32.shape[1]:
            continue

        prepared.append({
            "name": name,
            "weight": weight,
            "grad": grad,
            "grad_attr": grad_attr,
            "original_shape": original_shape,
            "g_dtype": g_dtype,
            "grad_f32": grad_f32,
            "L_A_f": L_A_f,
        })

    if not prepared:
        manager.update_tag("BASELINE")
        return False, synced_param_ids

    _run_right_lowrank_whiten_stage(prepared)
    _encode_entries(prepared, manager)

    if do_scale_sync:
        _run_scale_allreduce(prepared, group)

    state = _launch_async_q_allreduce(prepared, group)
    finalized_entries = _finalize_async_q_bucket(state, group, world_size)

    _run_right_lowrank_mapback_stage(finalized_entries)

    did_apply = False
    for item in finalized_entries:
        grad_after = item["grad_after_f32"]
        if grad_after.shape != item["original_shape"]:
            grad_after = grad_after.view(item["original_shape"])
        grad_new = grad_after.to(dtype=item["g_dtype"])

        _write_back_grad(item["weight"], item["grad_attr"], grad_new)

        did_apply = True
        synced_param_ids.add(id(item["weight"]))

    manager.update_tag("IG_FP8" if did_apply else "BASELINE")
    return did_apply, synced_param_ids


@torch.no_grad()
def allreduce_remaining_grads_fp32(model: nn.Module, group: dist.ProcessGroup, already_synced_param_ids: set):
    if group is None:
        return

    ws = dist.get_world_size(group=group)
    grads = []
    meta = []

    for p in model.parameters():
        if id(p) in already_synced_param_ids:
            continue
        g, _ = _get_param_grad(p)
        if g is None or getattr(g, "is_sparse", False):
            continue
        grads.append(g.detach())
        meta.append((g, g.shape, g.dtype))

    if not grads:
        return

    flat = torch.cat([g.float().contiguous().view(-1) for g in grads], dim=0)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    flat.div_(float(ws))

    offset = 0
    for g, shape, dtype in meta:
        n = g.numel()
        g.copy_(flat[offset:offset + n].view(shape).to(dtype=dtype))
        offset += n


def _iter_all_modules_with_direct_params(model: nn.Module):
    for _, module in model.named_modules():
        has_direct_param = False
        for _, p in module.named_parameters(recurse=False):
            g, _ = _get_param_grad(p)
            if g is None or not g.is_cuda or getattr(g, "is_sparse", False):
                continue
            has_direct_param = True
            break
        if has_direct_param:
            yield module


def _collect_module_remaining_grads(module: nn.Module, already_synced_param_ids: set) -> Tuple[List[torch.nn.Parameter], List[torch.Tensor], List[torch.Size], int]:
    params, grads, shapes = [], [], []
    total_numel = 0

    for _, p in module.named_parameters(recurse=False):
        if id(p) in already_synced_param_ids:
            continue
        g, _ = _get_param_grad(p)
        if g is None or not g.is_cuda or getattr(g, "is_sparse", False):
            continue
        params.append(p)
        grads.append(g)
        shapes.append(g.shape)
        total_numel += g.numel()

    return params, grads, shapes, total_numel


@torch.no_grad()
def allreduce_remaining_grads_fp8_euclidean_e5m2(model: nn.Module, group: dist.ProcessGroup, already_synced_param_ids: set):
    if group is None:
        return

    ws = dist.get_world_size(group=group)
    FP8_MAX = 57344.0
    EPS = 1e-8
    module_infos = []

    for module in _iter_all_modules_with_direct_params(model):
        params, grads, shapes, module_numel = _collect_module_remaining_grads(module, already_synced_param_ids)
        if module_numel == 0:
            continue

        g_flat = torch.cat([g.detach().float().contiguous().view(-1) for g in grads], dim=0).view(1, -1)
        amax = g_flat.abs().amax()
        scale = (amax / FP8_MAX).clamp(min=EPS)

        module_infos.append({
            "params": params,
            "grads": grads,
            "shapes": shapes,
            "g_flat": g_flat,
            "scale": scale,
        })

    if not module_infos:
        return

    scale_bucket = torch.stack([info["scale"].reshape(1) for info in module_infos], dim=0)
    dist.all_reduce(scale_bucket, op=dist.ReduceOp.MAX, group=group)
    for i, info in enumerate(module_infos):
        info["scale"] = scale_bucket[i].reshape(())

    q_chunks = []
    q_numels = []
    for info in module_infos:
        q = (info["g_flat"] / info["scale"]).to(torch.float8_e5m2).contiguous().view(-1)
        q_chunks.append(q)
        q_numels.append(q.numel())

    q_bucket = torch.cat(q_chunks, dim=0).view(1, -1)
    dist.all_reduce(q_bucket, op=dist.ReduceOp.SUM, group=group)

    q_bucket_flat = q_bucket.view(-1)
    q_offset = 0

    for info, qn in zip(module_infos, q_numels):
        q_part = q_bucket_flat[q_offset:q_offset + qn].view(1, -1)
        gavg = (q_part.float() * info["scale"] / float(ws)).view(-1)

        offset = 0
        for p, g, shape in zip(info["params"], info["grads"], info["shapes"]):
            n = g.numel()
            gnew = gavg[offset:offset + n].view(shape).to(dtype=g.dtype)
            _, grad_attr = _get_param_grad(p)
            _write_back_grad(p, grad_attr, gnew)
            offset += n

        q_offset += qn


def _real_model(m):
    m = m[0] if isinstance(m, (list, tuple)) else m
    return m.module if hasattr(m, "module") else m


def wrap_optimizer_step_once(model, optimizer, manager, get_step):
    if getattr(optimizer, "_kfac_fp8_wrapped", False):
        return
    optimizer._kfac_fp8_wrapped = True

    _orig_step = optimizer.step
    import megatron.core.optimizer.fisher_manager as fisher_manager_module

    def step_wrapper(*args, **kwargs):
        step = get_step()
        real = _real_model(model)
        group = manager.group

        setattr(real, "_ig_profile_manager_ref", manager)

        manager._register_hooks()
        manager.step_and_update(step)
        did_apply, synced_ids = precondition_gradients_with_ef(real, manager, global_step=step)

        if fisher_manager_module._SKIP_DP_SYNC_THIS_STEP:
            allreduce_remaining_grads_fp8_euclidean_e5m2(
                real,
                group=group,
                already_synced_param_ids=synced_ids,
            )

        out = _orig_step(*args, **kwargs)
        manager._hook_step += 1
        return out

    optimizer.step = step_wrapper
