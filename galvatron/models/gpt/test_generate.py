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
    try:
        copy_hf_weights_to_galvatron(hf_model, galvatron_model, args)
        print(f"[Rank {rank}] Weight copy completed without errors")
    except Exception as e:
        print(f"[Rank {rank}] Weight copy FAILED: {e}")
        import traceback
        traceback.print_exc()

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

    # ---- Diagnostic: compare prefill logits ----
    print(f"\n[Rank {rank}] === PREFILL LOGITS DIAGNOSTIC ===")
    with torch.no_grad():
        hf_out = hf_model(input_ids)
        hf_logits = hf_out.logits  # [batch, seq, vocab]
    print(f"[Rank {rank}] HF prefill logits shape: {hf_logits.shape}")
    print(f"[Rank {rank}] HF last-token top5: {torch.topk(hf_logits[0, -1], 5)}")
    print(f"[Rank {rank}] HF last-token argmax: {hf_logits[0, -1].argmax().item()}")

    from galvatron.core.runtime.transformer.inference import StaticInferenceContext
    ctx = StaticInferenceContext(max_batch_size=1, max_sequence_length=128)
    ctx.enable_prefill_mode()
    with torch.inference_mode():
        gv_logits = galvatron_model.model.forward_only(input_ids, inference_context=ctx)
    print(f"[Rank {rank}] GV prefill logits shape: {gv_logits.shape}")  # [seq, batch, vocab]
    print(f"[Rank {rank}] GV last-token top5: {torch.topk(gv_logits[-1, 0].float(), 5)}")
    print(f"[Rank {rank}] GV last-token argmax: {gv_logits[-1, 0].float().argmax().item()}")

    v = min(hf_logits.shape[-1], gv_logits.shape[-1])
    max_diff = (hf_logits[0, :, :v].float() - gv_logits[:, 0, :v].float()).abs().max().item()
    mean_diff = (hf_logits[0, :, :v].float() - gv_logits[:, 0, :v].float()).abs().mean().item()
    print(f"[Rank {rank}] Max logit diff: {max_diff}, Mean logit diff: {mean_diff}")
    for pos in range(hf_logits.shape[1]):
        hf_am = hf_logits[0, pos, :v].float().argmax().item()
        gv_am = gv_logits[pos, 0, :v].float().argmax().item()
        match_str = "OK" if hf_am == gv_am else "DIFF"
        print(f"[Rank {rank}]   pos {pos}: HF argmax={hf_am}, GV argmax={gv_am}  {match_str}")

    # Galvatron generate with copied weights
    print(f"\n[Rank {rank}] Running Galvatron generate (HF weights, greedy)...")
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


def _interleave_qkv(q, k, v, num_query_groups, num_attention_heads):
    """Interleave Q/K/V weights into Galvatron's per-group fused format.

    HF stores separate [q_proj, k_proj, v_proj].
    Galvatron's SelfAttention expects fused QKV laid out as:
        [q_group0, k_group0, v_group0, q_group1, k_group1, v_group1, ...]
    where q_groupX has (num_attention_heads // num_query_groups) heads.
    """
    heads_per_group = num_attention_heads // num_query_groups
    head_dim = q.shape[0] // num_attention_heads

    q_groups = q.view(num_query_groups, heads_per_group * head_dim, *q.shape[1:])
    k_groups = k.view(num_query_groups, head_dim, *k.shape[1:])
    v_groups = v.view(num_query_groups, head_dim, *v.shape[1:])

    chunks = []
    for g in range(num_query_groups):
        chunks.append(q_groups[g])
        chunks.append(k_groups[g])
        chunks.append(v_groups[g])
    return torch.cat(chunks, dim=0)


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
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    num_attention_heads = args.model.num_attention_heads
    num_query_groups = args.model.num_query_groups or num_attention_heads

    hf_sd = hf_model.state_dict()
    pipe_model = galvatron_model.model.model_cur_stage

    # Unwrap FSDP to get PipeSequential
    unwrapped = pipe_model
    while isinstance(unwrapped, FSDP):
        unwrapped = unwrapped._fsdp_wrapped_module

    # Use summon_full_params to allow direct weight writes through FSDP
    with FSDP.summon_full_params(pipe_model, writeback=True):
        modules = list(unwrapped.named_children())

        for name, module in modules:
            # Unwrap FSDP and Module_with_relocation on individual modules
            m = module
            while hasattr(m, '_fsdp_wrapped_module') or hasattr(m, 'module'):
                if hasattr(m, '_fsdp_wrapped_module'):
                    m = m._fsdp_wrapped_module
                elif hasattr(m, 'module') and type(m).__name__ == 'Module_with_relocation':
                    m = m.module
                else:
                    break

            if name.startswith("embedding"):
                gv_w = m.embed_tokens.weight
                hf_w = hf_sd["model.embed_tokens.weight"]
                n = min(gv_w.shape[0], hf_w.shape[0])
                gv_w.data[:n].copy_(hf_w[:n])

            elif name.startswith("decoder"):
                layer_idx = m.idx
                prefix = f"model.layers.{layer_idx}"

                m.attn.input_layernorm.weight.data.copy_(hf_sd[f"{prefix}.input_layernorm.weight"])

                q_weight = hf_sd[f"{prefix}.self_attn.q_proj.weight"]
                k_weight = hf_sd[f"{prefix}.self_attn.k_proj.weight"]
                v_weight = hf_sd[f"{prefix}.self_attn.v_proj.weight"]
                qkv_weight = _interleave_qkv(q_weight, k_weight, v_weight, num_query_groups, num_attention_heads)
                m.attn.attention.linear_qkv.weight.data.copy_(qkv_weight)

                if f"{prefix}.self_attn.q_proj.bias" in hf_sd:
                    q_bias = hf_sd[f"{prefix}.self_attn.q_proj.bias"]
                    k_bias = hf_sd[f"{prefix}.self_attn.k_proj.bias"]
                    v_bias = hf_sd[f"{prefix}.self_attn.v_proj.bias"]
                    qkv_bias = _interleave_qkv(q_bias, k_bias, v_bias, num_query_groups, num_attention_heads)
                    if m.attn.attention.linear_qkv.bias is not None:
                        m.attn.attention.linear_qkv.bias.data.copy_(qkv_bias)
                    else:
                        print(f"[WARNING] layer {layer_idx}: linear_qkv has no bias param, skipping bias copy")

                m.attn.attention.linear_proj.weight.data.copy_(
                    hf_sd[f"{prefix}.self_attn.o_proj.weight"]
                )

                m.ffn.post_attention_layernorm.weight.data.copy_(
                    hf_sd[f"{prefix}.post_attention_layernorm.weight"]
                )

                gate_weight = hf_sd[f"{prefix}.mlp.gate_proj.weight"]
                up_weight = hf_sd[f"{prefix}.mlp.up_proj.weight"]
                fc1_weight = torch.cat([gate_weight, up_weight], dim=0)
                m.ffn.mlp.linear_fc1.weight.data.copy_(fc1_weight)

                m.ffn.mlp.linear_fc2.weight.data.copy_(
                    hf_sd[f"{prefix}.mlp.down_proj.weight"]
                )

            elif name.startswith("prenorm"):
                m.norm.weight.data.copy_(hf_sd["model.norm.weight"])

            elif name.startswith("lm_head"):
                hf_w = hf_sd.get("lm_head.weight", hf_sd["model.embed_tokens.weight"])
                n = min(m.lm_head.weight.shape[0], hf_w.shape[0])
                m.lm_head.weight.data[:n].copy_(hf_w[:n])


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".yml")):
        config_path, overrides = sys.argv[1], sys.argv[2:]
        sys.argv = [sys.argv[0]]
        args = load_with_hydra(config_path, overrides=overrides, mode="train_dist")
    else:
        raise ValueError("Usage: python test_generate.py <config_path> [overrides...]")
    initialize_galvatron(args)
    test_generate(args)
