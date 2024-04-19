import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from megatron.model.rotary_pos_embedding import RotaryEmbedding
from galvatron.core import get_args
from galvatron.core.tensor_parallel import ParallelMLP, ParallelAttention
from galvatron.core.tensor_parallel import AttnMaskType, AttnType, init_method_normal, scaled_init_method_normal

class LlamaAttention_tp(nn.Module):
    def __init__(self, config, tp_group = None):
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.attention = ParallelAttention(init_method, 
                                        scaled_init_method, 
                                        attention_type=AttnType.self_attn,
                                        attn_mask_type=AttnMaskType.causal,
                                        tp_group = self.tp_group)
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.rotary_pos_emb = RotaryEmbedding(
                self.head_dim,
            )

    def forward(self, hidden_states, attention_mask):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        rotary_pos_emb = self.rotary_pos_emb(self.max_position_embeddings)
        hidden_states, bias = self.attention(hidden_states, attention_mask,rotary_pos_emb=rotary_pos_emb)
        hidden_states = hidden_states + bias + input_tensor
        return hidden_states

class LlamaMLP_tp(nn.Module):
    def __init__(self, config, tp_group = None):
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(init_method, scaled_init_method, tp_group = self.tp_group)
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.mlp(hidden_states)
        hidden_states = hidden_states + bias + input_tensor
        return hidden_states

class LlamaLayer_tp(nn.Module):
    def __init__(self, config, tp_group = None):
        super().__init__()
        self.attention = LlamaAttention_tp(config, tp_group)
        self.mlp = LlamaMLP_tp(config, tp_group)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
    ):
        hidden_states = hidden_states.permute(1, 0, 2)
        attention_output = self.attention(
            hidden_states,
            attention_mask,
        )
        layer_output = self.mlp(attention_output)
        layer_output = layer_output.permute(1, 0, 2)
        # outputs = (layer_output
        return layer_output
    
def construct_tensor_parallel_model(model, config, tp_groups_enc):
    layers_tp = nn.ModuleList([LlamaLayer_tp(config, tp_group = tp_groups_enc[i]) for i in range(config.num_hidden_layers)])
    setattr(model.model, 'layers', layers_tp)
    return model