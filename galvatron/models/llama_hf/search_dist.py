import os

from galvatron.core import GalvatronSearchEngine, initialize_galvatron
from galvatron.models.llama_hf.arguments import model_args
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_llama_config
from galvatron.models.llama_hf.meta_configs import model_layer_configs, model_name


def search_remote(overrides):
    try:
        args = initialize_galvatron(model_args, mode="search", ignore_unknown_args=True)
        if overrides:
            for k, v in overrides.__dict__.items():
                setattr(args, k, v)
        config = get_llama_config(args)
        path = os.path.dirname(os.path.abspath(__file__))
        print(args)
        print(config)

        search_engine = GalvatronSearchEngine(args)
        search_engine.set_search_engine_info(path, model_layer_configs(config), model_name(config))
        search_engine.set_model_type("gpt")

        search_engine.initialize_search_engine()
        search_engine.parallelism_optimization()
        result = {}
        result["success"] = True
        return result
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "rank": int(os.environ.get("RANK", -1)),
            "local_rank": int(os.environ.get("LOCAL_RANK", -1))
        }

if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="search")
    config = get_llama_config(args)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)

    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(path, model_layer_configs(config), model_name(config))
    # search_engine.set_microbatch_func(microbatch_size=4, max_chunk=8) # Optional
    search_engine.set_model_type("gpt")  # Optional

    search_engine.initialize_search_engine()
    search_engine.check_cost_model(bsz=64,chunk=64,min_tp=8)
    search_engine.parallelism_optimization()
