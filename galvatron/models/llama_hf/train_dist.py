import torch
from torch import nn
from torch.optim import Adam
from transformers import LlamaConfig, LlamaForCausalLM
from tqdm import tqdm
import os
from galvatron.utils import set_seed, distributed_dataloader, print_loss
from galvatron.core import initialize_galvatron, GalvatronProfiler
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_hybrid_parallel_configs, construct_hybrid_parallel_model
from galvatron.models.llama_hf.dataloader import DataLoaderForLlama
from galvatron.models.llama_hf.meta_configs import config_from_meta, set_model_config, model_name, model_layer_configs
from galvatron.models.llama_hf.arguments import model_args

from megatron.arguments import _print_args

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, False)
    if local_rank == 0:
        print(config)
        _print_args("arguments", args)
    
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if local_rank == 0:
        print("Creating Model...")
    
    llama_model = LlamaForCausalLM(config)
    
    # if local_rank == 0:
    #     for p,q in llama_model.named_parameters():
    #         print(p,q.numel())
    
    model = construct_hybrid_parallel_model(
        model=llama_model, 
        model_config=config, 
        training_args=args, 
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    
    # if local_rank == 0:
    #     for p,q in model.named_parameters():
    #         print(p,q.numel())
    
    # from galvatron.models.llama import llama_model_hp
    # model = llama_model_hp(config, args)
    
    if local_rank == 0:
        print("Creating Dataset...")
    trainloader = distributed_dataloader(
        dataset=DataLoaderForLlama(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=True,
        args = args,
        group = model.dp_groups_whole[0].group
    )
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = GalvatronProfiler(args)
    profiler.set_profiler_dist(path, model_layer_configs(config), model_name(config))
    
    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")
        
    # torch.cuda.memory._record_memory_history(max_entries=100000)
    # t = 0
    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    if True:
        for ep in range(args.epochs):
            if not args.check_loss and not args.profile:
                trainloader = tqdm(trainloader)
            for iter, batch in enumerate(trainloader):
                profiler.profile_time_start(iter)
                profiler.profile_memory(iter, "Before Forward")

                input_ids = batch
                batch = [input_ids]
                
                loss = model.forward_backward(batch, iter, profiler)
                
                profiler.profile_memory(iter, "After Backward")
                
                optimizer.step()
                
                profiler.profile_memory(iter, "After optimizer_step")
                
                optimizer.zero_grad()
                
                # t += 1
                # if t == 3:
                #     break
                
                print_loss(args, loss, ep, iter)

                profiler.post_profile_memory(iter)
                profiler.profile_time_end(iter)
            # if t == 3:
            #     break
        # torch.cuda.memory._dump_snapshot(f"13B_layer8_rank_{local_rank}.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)
    # prof.export_chrome_trace(f"trace_{local_rank}.json")
    
if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train_dist')
    set_seed()
    train(args)