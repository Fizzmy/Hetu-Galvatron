"""End-to-end generation test for Galvatron inference (Phase 1).

Compares Galvatron generate output with HuggingFace generate (greedy)
to verify correctness.

Usage:
    torchrun --nproc_per_node=1 test_generate.py scripts/test_generate.yaml
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from galvatron.core.arguments import load_with_hydra
from galvatron.core.runtime.models.builder import build_model
from galvatron.core.runtime.initialize import initialize_galvatron
from galvatron.utils.hf_config_adapter import resolve_model_config


def test_generate(args):
    rank = torch.distributed.get_rank()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    hf_path = args.model.hf_model_name_or_path
    print(f"[Rank {rank}] Loading tokenizer from {hf_path}")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

    # ---- Build Galvatron model ----
    print(f"[Rank {rank}] Building Galvatron model...")
    resolve_model_config(args)
    galvatron_model = build_model(args)
    galvatron_model.eval()
    print(f"[Rank {rank}] Galvatron model built")

    # ---- Test 1: Random weights, verify no crash ----
    prompt = "Hello, how are you?"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"[Rank {rank}] Input: '{prompt}' → tokens {input_ids.shape}")

    print(f"[Rank {rank}] Running Galvatron generate (random weights, greedy)...")
    galvatron_output = galvatron_model.generate(
        input_ids,
        max_new_tokens=16,
        temperature=0,
    )
    galvatron_text = tokenizer.decode(galvatron_output[0], skip_special_tokens=True)
    print(f"[Rank {rank}] Galvatron output shape: {galvatron_output.shape}")
    print(f"[Rank {rank}] Galvatron generated: '{galvatron_text}'")

    # ---- Test 2: Load HF model, copy weights, verify greedy match ----
    print(f"\n[Rank {rank}] Loading HuggingFace model for comparison...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    # Copy HF weights to Galvatron model
    print(f"[Rank {rank}] Copying HF weights to Galvatron model...")
    copy_hf_weights_to_galvatron(hf_model, galvatron_model, args)

    # Debug: verify weight copy
    pipe_model = galvatron_model.model.model_cur_stage
    for name, module in pipe_model.named_children():
        if name.startswith("embedding"):
            hf_w = hf_model.state_dict()["model.embed_tokens.weight"]
            gv_w = module.embed_tokens.weight.data
            print(f"[Rank {rank}] embed_tokens match: {torch.allclose(hf_w[:gv_w.shape[0]], gv_w, atol=1e-5)}, shapes: hf={hf_w.shape} gv={gv_w.shape}")
        if name.startswith("decoder") and module.idx == 0:
            hf_sd = hf_model.state_dict()
            q_w = hf_sd["model.layers.0.self_attn.q_proj.weight"]
            k_w = hf_sd["model.layers.0.self_attn.k_proj.weight"]
            v_w = hf_sd["model.layers.0.self_attn.v_proj.weight"]
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
            gv_qkv = module.attn.attention.linear_qkv.weight.data
            print(f"[Rank {rank}] layer0 qkv match: {torch.allclose(qkv_w, gv_qkv, atol=1e-5)}, shapes: hf={qkv_w.shape} gv={gv_qkv.shape}")
            break

    # HF generate (greedy)
    print(f"[Rank {rank}] Running HF generate (greedy)...")
    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=False,
        )
    hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
    print(f"[Rank {rank}] HF output shape: {hf_output.shape}")
    print(f"[Rank {rank}] HF generated: '{hf_text}'")

    # Galvatron generate with copied weights
    print(f"[Rank {rank}] Running Galvatron generate (HF weights, greedy)...")
    galvatron_output2 = galvatron_model.generate(
        input_ids,
        max_new_tokens=16,
        temperature=0,
    )
    galvatron_text2 = tokenizer.decode(galvatron_output2[0], skip_special_tokens=True)
    print(f"[Rank {rank}] Galvatron generated: '{galvatron_text2}'")

    # Compare
    match = torch.equal(hf_output[:, :galvatron_output2.shape[1]], galvatron_output2)
    print(f"\n[Rank {rank}] === RESULTS ===")
    print(f"[Rank {rank}] HF tokens:        {hf_output[0].tolist()}")
    print(f"[Rank {rank}] Galvatron tokens:  {galvatron_output2[0].tolist()}")
    print(f"[Rank {rank}] Match: {match}")

    if match:
        print(f"[Rank {rank}] PASSED: Galvatron generate matches HuggingFace!")
    else:
        print(f"[Rank {rank}] MISMATCH: outputs differ (may need weight mapping debug)")

    del hf_model
    torch.cuda.empty_cache()


def copy_hf_weights_to_galvatron(hf_model, galvatron_model, args):
    """Copy weights from HuggingFace Qwen2 model to Galvatron model.

    Galvatron model structure (PipeSequential):
        embedding_0: GalvatronEmbedding
            .embed_tokens: VocabParallelEmbedding
        decoder_1..N: GalvatronDecoderLayer
            .attn: GalvatronAttention
                .input_layernorm
                .attention: SelfAttention
                    .linear_qkv: ColumnParallelLinear (fused QKV)
                    .linear_proj: RowParallelLinear
            .ffn: GalvatronMLP
                .post_attention_layernorm
                .mlp: MLP
                    .linear_fc1: ColumnParallelLinear (fused gate+up)
                    .linear_fc2: RowParallelLinear
        prenorm_N+1: GalvatronFinalNorm
            .norm
        lm_head_N+2: GalvatronCausalLMHead
            .lm_head: _LMHeadLinear

    HF Qwen2 model structure:
        model.embed_tokens
        model.layers[i].self_attn.{q_proj, k_proj, v_proj, o_proj}
        model.layers[i].{input_layernorm, post_attention_layernorm}
        model.layers[i].mlp.{gate_proj, up_proj, down_proj}
        model.norm
        lm_head
    """
    hf_sd = hf_model.state_dict()
    pipe_model = galvatron_model.model.model_cur_stage

    modules = list(pipe_model.named_children())

    for name, module in modules:
        if name.startswith("embedding"):
            # embed_tokens
            module.embed_tokens.weight.data.copy_(hf_sd["model.embed_tokens.weight"])

        elif name.startswith("decoder"):
            # Parse layer index from module ordering
            layer_idx = module.idx
            prefix = f"model.layers.{layer_idx}"

            # Input layernorm
            module.attn.input_layernorm.weight.data.copy_(hf_sd[f"{prefix}.input_layernorm.weight"])

            # Fused QKV: Galvatron stores as single [3*h, h] or [(nq+2nkv)*d, h]
            q_weight = hf_sd[f"{prefix}.self_attn.q_proj.weight"]
            k_weight = hf_sd[f"{prefix}.self_attn.k_proj.weight"]
            v_weight = hf_sd[f"{prefix}.self_attn.v_proj.weight"]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            module.attn.attention.linear_qkv.weight.data.copy_(qkv_weight)

            # QKV bias (Qwen has qkv bias)
            if f"{prefix}.self_attn.q_proj.bias" in hf_sd:
                q_bias = hf_sd[f"{prefix}.self_attn.q_proj.bias"]
                k_bias = hf_sd[f"{prefix}.self_attn.k_proj.bias"]
                v_bias = hf_sd[f"{prefix}.self_attn.v_proj.bias"]
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                module.attn.attention.linear_qkv.bias.data.copy_(qkv_bias)

            # Output projection
            module.attn.attention.linear_proj.weight.data.copy_(
                hf_sd[f"{prefix}.self_attn.o_proj.weight"]
            )

            # Post-attention layernorm
            module.ffn.post_attention_layernorm.weight.data.copy_(
                hf_sd[f"{prefix}.post_attention_layernorm.weight"]
            )

            # MLP: gate_proj + up_proj fused into linear_fc1
            gate_weight = hf_sd[f"{prefix}.mlp.gate_proj.weight"]
            up_weight = hf_sd[f"{prefix}.mlp.up_proj.weight"]
            fc1_weight = torch.cat([gate_weight, up_weight], dim=0)
            module.ffn.mlp.linear_fc1.weight.data.copy_(fc1_weight)

            # MLP: down_proj → linear_fc2
            module.ffn.mlp.linear_fc2.weight.data.copy_(
                hf_sd[f"{prefix}.mlp.down_proj.weight"]
            )

        elif name.startswith("prenorm"):
            module.norm.weight.data.copy_(hf_sd["model.norm.weight"])

        elif name.startswith("lm_head"):
            if "lm_head.weight" in hf_sd:
                module.lm_head.weight.data.copy_(hf_sd["lm_head.weight"])
            else:
                # tied embeddings
                module.lm_head.weight.data.copy_(hf_sd["model.embed_tokens.weight"])


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".yml")):
        config_path, overrides = sys.argv[1], sys.argv[2:]
        sys.argv = [sys.argv[0]]
        args = load_with_hydra(config_path, overrides=overrides, mode="train_dist")
    else:
        raise ValueError("Usage: python test_generate.py <config_path> [overrides...]")
    initialize_galvatron(args)
    test_generate(args)
