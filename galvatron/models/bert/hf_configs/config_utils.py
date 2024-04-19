import os
from galvatron.utils import dict_join_dirname

# ============= HuggingFace Model Config Paths =============
path_dict =  {
    'bert-base': 'bert-base-uncased.json',
    'bert-large': 'bert-large-uncased.json',
    'bert-huge-32': 'bert-huge-uncased-32.json',
    'bert-huge-48': 'bert-huge-uncased-48.json',
}

def bert_hf_configs(model_type):
    global path_dict
    path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
    return path_dict[model_type]

# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.vocab_size = args.vocab_size
        config.hidden_size = args.hidden_size
        config.num_hidden_layers = args.num_hidden_layers
        config.num_attention_heads = args.num_attention_heads
        config.intermediate_size = args.hidden_size*4
        config.max_position_embeddings = args.seq_length
        config.attention_probs_dropout_prob = args.dropout_prob
        config.hidden_dropout_prob = args.dropout_prob
    # Overwrite layer number only
    elif args.set_layernum_manually:
        config.num_hidden_layers = args.num_hidden_layers
    
    # ======= Model Config --> Arguments ======
    # This step is necessary that maintains the consistency of model config and arguments.
    # Overwrite the model arguments with the model config
    overwrite_model_args(config, args)
    
    if overwrite_args: # Overwrite necessary Megatron-LM arguments with the model config
        overwrite_megatron_args(config, args)
    return config

def overwrite_megatron_args(config, args):
    args.num_layers = config.num_hidden_layers
    args.ffn_hidden_size = config.intermediate_size
    args.max_position_embeddings = config.max_position_embeddings
    args.attention_dropout = config.attention_probs_dropout_prob
    args.hidden_dropout = config.hidden_dropout_prob
    args.use_cpu_initialization = True

# Need to overwrite the arguments with the model config
def overwrite_model_args(config, args):
    args.hidden_size = config.hidden_size
    args.num_hidden_layers = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.seq_length = config.max_position_embeddings
    args.vocab_size = config.vocab_size

# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    return 'hidden%d_head%d_seqlen%d'%(config.hidden_size, config.num_attention_heads, config.max_position_embeddings)

def model_layer_configs(config):
    return [
        {'hidden_size': config.hidden_size,
         'seq_len': config.max_position_embeddings,
         'layer_num': config.num_hidden_layers
         }
    ]