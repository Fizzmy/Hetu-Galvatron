import os

import torch
from torch import nn
from torch.optim import Adam
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from galvatron.core import initialize_galvatron
from galvatron.models.llama_hf.arguments import model_args
from galvatron.models.llama_hf.dataloader import DataLoaderForLlama, random_collate_fn
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_llama_config, get_runtime_profiler, llama_model_hp
from galvatron.utils import distributed_dataloader, print_loss, set_seed
from megatron.training.arguments import _print_args


def train(args, return_result=False):
    local_rank = args.local_rank
    if return_result:
        device_id = 0
    else:
        device_id = local_rank
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)
    config = get_llama_config(args)
    model = llama_model_hp(config, args)

    if local_rank == 0:
        print("Creating Dataset...")

    trainloader = distributed_dataloader(
        dataset=DataLoaderForLlama(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=True,
        args=args,
        group=model.dp_groups_whole[0].group,
        collate_fn=random_collate_fn,
    )

    if local_rank == 0:
        _print_args("arguments", args)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    path = os.path.dirname(os.path.abspath(__file__))
    if return_result:
        result = {"data": None}
    else:
        result = None
    profiler = get_runtime_profiler(args, path, config, result)

    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")

    if args.profile_forward:
        torch.set_grad_enabled(False)

    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            tokens, kwargs, loss_func = batch
            profiler.profile_time_start(iter)
            profiler.profile_memory(iter, "Before Forward")

            input_ids = tokens
            batch = [input_ids]

            loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)

            profiler.profile_memory(iter, "After Backward")

            optimizer.step()

            profiler.profile_memory(iter, "After optimizer_step")

            optimizer.zero_grad()

            print_loss(args, loss, ep, iter)

            profiler.post_profile_memory(iter)
            profiler.profile_time_end(iter)

            torch.distributed.barrier()

            if return_result and hasattr(profiler, 'profiling_complete') and profiler.profiling_complete:
                return result

def train_remote(overrides):
    try:
        args = initialize_galvatron(model_args, mode="train_dist", ignore_unknown_args=True)
        if overrides:
            for k, v in overrides.__dict__.items():
                setattr(args, k, v)
        set_seed()
        result = train(args, return_result=True)
        result["success"] = True
        result["data"] = {
            f"{args.key}": result["data"],
        }
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
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args, return_result=True)

