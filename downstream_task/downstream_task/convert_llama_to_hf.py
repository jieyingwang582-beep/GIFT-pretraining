import argparse
import os
import re
import torch
from transformers import LlamaConfig

def recursive_print(name, val, spaces=0):
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val:
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", tuple(val.size()))
    else:
        print(msg, ":", val)

def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size_per_head):
    input_shape = param.size()
    if checkpoint_version == 1.0:
        saved_shape = (num_heads, hidden_size_per_head, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        saved_shape = (num_heads, num_splits, hidden_size_per_head) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param

def get_config_from_checkpoint(input_state_dict, args):
    ds_args = input_state_dict.get("args", None)

    vocab_size = args.vocab_size
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_hidden_layers = args.num_hidden_layers
    num_attention_heads = args.num_attention_heads
    num_key_value_heads = args.num_key_value_heads
    max_position_embeddings = args.max_position_embeddings
    rms_norm_eps = args.rms_norm_eps
    rope_theta = args.rope_theta

    if ds_args is not None:
        vocab_size = getattr(ds_args, "padded_vocab_size", vocab_size)
        hidden_size = getattr(ds_args, "hidden_size", hidden_size)
        intermediate_size = getattr(ds_args, "ffn_hidden_size", intermediate_size)
        num_hidden_layers = getattr(ds_args, "num_layers", num_hidden_layers)
        num_attention_heads = getattr(ds_args, "num_attention_heads", num_attention_heads)
        max_position_embeddings = getattr(ds_args, "max_position_embeddings", max_position_embeddings)

    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        tie_word_embeddings=args.tie_word_embeddings,
    )
    return config

def extract_model_parts(input_state_dict):
    model = input_state_dict["model"]
    embeddings = {}
    transformer = {}
    output_layer = None

    if "language_model" in model:
        # legacy nested
        lm = model["language_model"]
        embeddings = lm["embedding"]
        if "transformer" in lm:
            transformer = lm["transformer"]
        else:
            transformer = lm["encoder"]
        if "output_layer" in lm:
            output_layer = lm["output_layer"]
    else:
        # flat MCore
        for key, val in model.items():
            if key.startswith("embedding."):
                parts = key.split(".")
                if parts[1] == "word_embeddings" and parts[2] == "weight":
                    embeddings.setdefault("word_embeddings", {})["weight"] = val
            elif key.startswith("decoder."):
                new_key = key[len("decoder."):]
                transformer[new_key] = val
            elif key == "output_layer.weight":
                output_layer = val

    return embeddings, transformer, output_layer

def set_hf_llama_norm(output_state_dict, layer_idx, which, tensor):
    # which: "input_layernorm" or "post_attention_layernorm"
    if which == "input_layernorm":
        output_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = tensor
    elif which == "post_attention_layernorm":
        output_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = tensor

def convert_megatron_checkpoint(input_state_dict, config, print_structure=False):
    output_state_dict = {}
    checkpoint_version = input_state_dict.get("checkpoint_version", 0.0)

    embeddings, transformer, output_layer = extract_model_parts(input_state_dict)

    if print_structure:
        print("=== embeddings ===")
        recursive_print(None, embeddings)
        print("=== transformer ===")
        recursive_print(None, transformer)

    # embeddings
    word_embeddings = embeddings["word_embeddings"]["weight"][: config.vocab_size, :]
    output_state_dict["model.embed_tokens.weight"] = word_embeddings

    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    # if GQA is used, you'd need more logic here. for your current model assume kv_heads == attn_heads.
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError(
            f"Currently this converter assumes num_key_value_heads == num_attention_heads, "
            f"but got {config.num_key_value_heads} vs {config.num_attention_heads}."
        )

    # Match keys like:
    # layers.0.self_attention.linear_qkv.weight
    # layers.0.self_attention.linear_proj.weight
    # layers.0.mlp.linear_fc1.weight
    # layers.0.mlp.linear_fc2.weight
    # layers.0.input_layernorm.weight
    # layers.0.pre_mlp_layernorm.weight
    layer_re = re.compile(r"layers\.(\d+)\.([a-zA-Z0-9_.]+)\.(weight|bias|layer_norm_weight|layer_norm_bias)$")

    for key, val in transformer.items():
        m = layer_re.match(key)
        if m is None:
            # final norm
            if key in ["final_norm.weight", "final_layernorm.weight"]:
                output_state_dict["model.norm.weight"] = val
            elif key in ["final_norm.bias", "final_layernorm.bias"]:
                # LLaMA HF has no final norm bias
                pass
            continue

        layer_idx = int(m.group(1))
        op_name = m.group(2)
        suffix = m.group(3)

        # skip TE extra state
        if suffix == "_extra_state":
            continue

        # -----------------------------
        # layer norms
        # -----------------------------
        # LLaMA wants:
        #   input_layernorm.weight
        #   post_attention_layernorm.weight
        #
        # Megatron/MCore may store:
        #   input_layernorm.weight
        #   pre_mlp_layernorm.weight
        #   self_attention.linear_qkv.layer_norm_weight
        #   mlp.linear_fc1.layer_norm_weight
        #
        if suffix in ["weight", "layer_norm_weight"]:
            if op_name == "input_layernorm":
                set_hf_llama_norm(output_state_dict, layer_idx, "input_layernorm", val)
                continue
            elif op_name in ["pre_mlp_layernorm", "post_attention_layernorm"]:
                set_hf_llama_norm(output_state_dict, layer_idx, "post_attention_layernorm", val)
                continue
            elif op_name == "self_attention.linear_qkv" and suffix == "layer_norm_weight":
                set_hf_llama_norm(output_state_dict, layer_idx, "input_layernorm", val)
                continue
            elif op_name == "mlp.linear_fc1" and suffix == "layer_norm_weight":
                set_hf_llama_norm(output_state_dict, layer_idx, "post_attention_layernorm", val)
                continue

        # ignore norm bias; LLaMA RMSNorm has no bias
        if suffix in ["bias", "layer_norm_bias"] and (
            op_name in ["input_layernorm", "pre_mlp_layernorm", "post_attention_layernorm",
                        "self_attention.linear_qkv", "mlp.linear_fc1"]
        ):
            continue

        # -----------------------------
        # QKV
        # -----------------------------
        if op_name == "self_attention.linear_qkv" and suffix == "weight":
            # Megatron weight shape usually [3*hidden, hidden]
            qkv = fix_query_key_value_ordering(val, checkpoint_version, 3, num_heads, head_dim)
            if qkv.shape[0] != 3 * hidden_size:
                raise ValueError(
                    f"Unexpected qkv weight shape at layer {layer_idx}: {tuple(qkv.shape)}; "
                    f"expected first dim = 3*hidden_size = {3*hidden_size}"
                )
            q, k, v = torch.chunk(qkv, 3, dim=0)
            output_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q
            output_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k
            output_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v
            continue

        if op_name == "self_attention.linear_qkv" and suffix == "bias":
            # LLaMA HF attention has no q/k/v bias
            continue

        # -----------------------------
        # attention output proj
        # -----------------------------
        if op_name in ["self_attention.linear_proj", "self_attention.proj", "self_attention.dense"] and suffix == "weight":
            output_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = val
            continue
        if op_name in ["self_attention.linear_proj", "self_attention.proj", "self_attention.dense"] and suffix == "bias":
            continue

        # -----------------------------
        # MLP fc1 -> gate_proj + up_proj
        # -----------------------------
        if op_name in ["mlp.linear_fc1", "layernorm_mlp.fc1", "mlp.dense_h_to_4h"] and suffix == "weight":
            # SwiGLU: linear_fc1 weight is usually [2*intermediate_size, hidden_size]
            if val.shape[0] != 2 * config.intermediate_size:
                raise ValueError(
                    f"Unexpected fc1 shape at layer {layer_idx}: {tuple(val.shape)}; "
                    f"expected first dim = 2*intermediate_size = {2*config.intermediate_size}"
                )
            gate_proj, up_proj = torch.chunk(val, 2, dim=0)
            output_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_proj
            output_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_proj
            continue
        if op_name in ["mlp.linear_fc1", "layernorm_mlp.fc1", "mlp.dense_h_to_4h"] and suffix == "bias":
            continue

        # -----------------------------
        # MLP fc2 -> down_proj
        # -----------------------------
        if op_name in ["mlp.linear_fc2", "layernorm_mlp.fc2", "mlp.dense_4h_to_h"] and suffix == "weight":
            output_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = val
            continue
        if op_name in ["mlp.linear_fc2", "layernorm_mlp.fc2", "mlp.dense_4h_to_h"] and suffix == "bias":
            continue

    # lm_head
    if output_layer is not None:
        if isinstance(output_layer, dict) and "weight" in output_layer:
            output_state_dict["lm_head.weight"] = output_layer["weight"][: config.vocab_size, :]
        elif torch.is_tensor(output_layer):
            output_state_dict["lm_head.weight"] = output_layer[: config.vocab_size, :]
        else:
            output_state_dict["lm_head.weight"] = word_embeddings
    else:
        if config.tie_word_embeddings:
            output_state_dict["lm_head.weight"] = word_embeddings
        else:
            output_state_dict["lm_head.weight"] = word_embeddings

    return output_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_checkpoint", type=str)
    parser.add_argument("--print-checkpoint-structure", action="store_true")

    # overrides if checkpoint args are absent / wrong
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=2736)
    parser.add_argument("--num-hidden-layers", type=int, default=24)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--max-position-embeddings", type=int, default=4096)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--tie-word-embeddings", action="store_true")

    args = parser.parse_args()

    basename = os.path.dirname(args.path_to_checkpoint)
    print(f"Loading {args.path_to_checkpoint}")
    input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu", weights_only=False)

    config = get_config_from_checkpoint(input_state_dict, args)
    print("HF config:")
    print(config)

    print("Converting")
    output_state_dict = convert_megatron_checkpoint(
        input_state_dict,
        config,
        print_structure=args.print_checkpoint_structure
    )

    config.save_pretrained(basename)
    torch.save(output_state_dict, os.path.join(basename, "pytorch_model.bin"))
    print(f"Saved to {basename}")

if __name__ == "__main__":
    main()
