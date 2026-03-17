from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class OverlapCoefficientArgs:
    """Arguments for overlap coefficient profiling"""
    overlap_time_multiply: int = 4

@dataclass
class HardwareProfileArgs:
    """Arguments for hardware profiling"""
    global_tp_deg: int = -1
    global_tp_consec: int = -1
    pp_deg: int = -1
    local_batch_size: int = 32
    num_layers: int = 24
    profile_time: int = 0
    nproc_per_node: int = 8

@dataclass
class ModelProfileConfigs:
    """Arguments for model profiling"""
    layernum_min: int = 1
    layernum_max: int = 2
    profile_batch_size: int = None
    profile_min_batch_size: Optional[int] = None
    profile_max_batch_size: Optional[int] = None
    profile_batch_size_step: int = None
    profile_seq_length: int = None
    profile_min_seq_length: Optional[int] = None
    profile_max_seq_length: Optional[int] = None
    profile_seq_length_step: int = None

@dataclass
class ModelProfileArgs:
    """Arguments for model profiling"""
    # Model args
    hidden_size: int = 4096
    vocab_size: int = 32000
    num_attention_heads: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14336
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    seq_length: int = 2048

    # Profile args
    profile_unit: str = "all"
    profile_mode: str = "batch"
    model_size: str = "llama-7b"
    global_train_batch_size: int = 1
    save_profiled_memory: int = 0
    profile_forward: int = 1
    sdp: int = 0
    embed_sdp: int = 0
    default_dp_type: str = "ddp"
    num_hidden_layers: int = 2
    pp_deg: int = 1
    global_tp_deg: int = 1
    global_checkpoint: int = 0
    

    # Default args
    set_model_config_manually: int = 1
    set_layernum_manually: int = 1
    set_seqlen_manually: int = 1
    initialize_on_meta: int = 1
    dropout_prob: float = 0.1
    adam_weight_decay: float = 0.01
    check_loss: int = 0
    profile: int = 1
    profile_type: str = "allocated"
    set_experts_manually: int = 0
    shape_order: str = "SBH"
    epochs: int = 10
    lr: float = 0.0001
    check_loss: int = 0
    global_tp_consec: int = 1
    global_cp_deg: int = 1
    chunks: int = 1
    pipeline_type: str = "gpipe"
    mixed_precision: str = "bf16"
    use_flash_attn: bool = True
    sequence_parallel: bool = True
    vocab_tp: int = 1
    vocab_cp: int = 1

@dataclass
class ModelConfigs:
    """Model configuration parameters"""
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    vocab_size: Optional[int] = None

@dataclass
class SearchArgs:
    """Search arguments dataclass"""
    model_type: str
    model_size: str
    model_identifier: str
    comp_profile_mode: str
    mem_profile_mode: str
    min_bsz: int
    max_bsz: int
    settle_bsz: int
    settle_chunk: int
    memory_constraint: int
    seq_length: int
    search_space: str
    sp_space: str
    max_tp_deg: int
    max_pp_deg: int
    disable_dp: int
    disable_tp: int
    disable_pp: int
    disable_sdp: int
    disable_ckpt: int
    disable_vtp: int
    disable_tp_consec: int
    pipeline_type: str
    default_dp_type: str
    mixed_precision: str
    fine_grained_mode: int
    sequence_parallel: bool
    num_nodes: int
    num_gpus_per_node: int

    set_model_config_manually: int = 1
    recommend_min_bsz: int = 0
    bsz_scale: int = 8

    global_memory_buffer: bool = True
    async_grad_reduce: bool = True
    gui_hardware_dir: Optional[str] = None
    gui_model_dir: Optional[str] = None
    output_config_path: Optional[str] = None
    log_dir: Optional[str] = None
    time_profile_mode: Optional[str] = None
    memory_profile_mode: Optional[str] = None

@dataclass
class TrainingArgs:
    """Training arguments dataclass"""
    model_type: str
    model_size: str
    model_identifier: str
    galvatron_config_path: str
    num_nodes: int
    num_gpus_per_node: int
    train_iters: int
    lr: float
    min_lr: float
    lr_warmup_fraction: float
    lr_decay_style: str
    adam_weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float

    async_grad_reduce: bool = True

    seq_length: int = 4096

    # data_path can be a single string or a list of strings
    # For --data-path with nargs='*', it accepts:
    # (1) a single prefix: ["/path/to/data"]
    # (2) weight prefix pairs: ["weight1", "prefix1", "weight2", "prefix2"]
    # (3) a list of prefixes: ["prefix1", "prefix2"] (weights inferred from dataset lengths)
    data_path: Optional[Union[str, List[str]]] = None
    split: Optional[str] = "949,50,1"
    tokenizer_type: Optional[str] = "HuggingFaceTokenizer"
    tokenizer_model: Optional[str] = None
    
    load: Optional[str] = None
    save: Optional[str] = None
    save_interval: int = 100

    set_model_config_manually: int = 1
    check_loss: int = 0
    profile: int = 1
    save_profiled_memory: int = 0
    use_flash_attn: bool = True
    sequence_parallel: bool = True
    initialize_on_meta: int = 1
    mixed_precision: str = "bf16"

