from galvatron.core import GalvatronProfiler, initialize_galvatron
from galvatron.models.bert.arguments import model_args, layernum_arg_names
from galvatron.models.bert.hf_configs import bert_hf_configs, set_model_config, model_name
from transformers import BertConfig
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='profile')
    config = BertConfig.from_pretrained(bert_hf_configs(args.model_size))
    config = set_model_config(config, args, overwrite_args=False)
    
    profiler = GalvatronProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))
    
    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()