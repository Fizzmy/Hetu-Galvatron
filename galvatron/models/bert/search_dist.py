from galvatron.core import initialize_galvatron, GalvatronSearchEngine
from galvatron.models.bert.arguments import model_args
from galvatron.models.bert.hf_configs import bert_hf_configs, set_model_config, model_name, model_layer_configs
from transformers import BertConfig
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='search')
    config = BertConfig.from_pretrained(bert_hf_configs(args.model_size))
    config = set_model_config(config, args, overwrite_args=False)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)
    
    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(path, model_layer_configs(config), model_name(config))
    search_engine.set_microbatch_func(microbatch_size=4, max_chunk=8) # Optional
    search_engine.set_model_type('bert') # Optional
    
    search_engine.initialize_search_engine()
    search_engine.check_cost_model(bsz=128)
    search_engine.parallelism_optimization()