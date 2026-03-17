import torch
import torch.distributed as dist
import os
import argparse

from galvatron.utils import read_json_config, write_json_config
from galvatron.utils.training_utils import gen_profiling_groups

# Constants
SEQ_LEN = 512
HIDDEN_SIZE = 1024
BYTES_PER_FLOAT16 = 2
MB_TO_BYTES = 1024 * 1024
WARMUP_ITERATIONS = 5
PROFILE_ITERATIONS = 20
ITERATIONS_PER_MEASUREMENT = 10
TRIM_EDGES = 5  # Trim first and last N measurements for stability

def single_all_to_all(input_tensor, group):
    seq_world_size = dist.get_world_size(group)
    input_t = input_tensor.reshape(seq_world_size, -1)
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)
    return output

def set_seed(rank):
    seed = 123 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args, return_result=False):
    if hasattr(args, "local_rank") and args.local_rank >= 0:
        local_rank = args.local_rank
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set CUDA device BEFORE initializing process group
    if return_result:
        device_id = 0
    else:
        device_id = local_rank
    
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)
    
    # Initialize process group after setting device
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    set_seed(rank)
    world_size = torch.distributed.get_world_size()
    node_num = world_size // args.nproc_per_node

    if rank == 0:
        print(f'local_bsz = {args.local_batch_size}')
    
    # Calculate theoretical communication message size
    tp_size = args.global_tp_deg
    batch_size = args.local_batch_size
    comm_group = gen_profiling_groups(tp_size, args.global_tp_consec)
    all2all_message_size = (batch_size * SEQ_LEN * HIDDEN_SIZE * BYTES_PER_FLOAT16 / MB_TO_BYTES) * (tp_size - 1) / tp_size

    if local_rank == 0:
        print(f'Strategy: {args.pp_deg}_{tp_size}_{args.global_tp_consec}')
        print(f'[all2all_message_size]: per_layer {all2all_message_size:.2f} MB')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_list = []
    
    # Warm-up: directly allocate on GPU
    for _ in range(WARMUP_ITERATIONS):
        input_tensor = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        single_all_to_all(input_tensor, comm_group)
    
    # Profiling loop
    for _ in range(PROFILE_ITERATIONS):
        input_tensor = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=comm_group)
        start.record()
        for __ in range(ITERATIONS_PER_MEASUREMENT):
            single_all_to_all(input_tensor, comm_group)
        end.record()
        torch.cuda.synchronize()
        time_list.append(start.elapsed_time(end) / ITERATIONS_PER_MEASUREMENT)
    
    time_list = sorted(time_list)
    per_comm_time = sum(time_list[TRIM_EDGES:-TRIM_EDGES]) / len(time_list[TRIM_EDGES:-TRIM_EDGES])
    per_comm_time = torch.tensor([per_comm_time]).to(device)
    torch.distributed.all_reduce(per_comm_time, group=comm_group, op=torch.distributed.ReduceOp.SUM)
    per_comm_time = per_comm_time.cpu().numpy()[0] / tp_size
    
    def save_config(filename_template, key, value):
        """Helper function to save configuration to file"""
        path = os.path.dirname(os.path.abspath(__file__))
        env_config_path = os.path.join(path, filename_template % (node_num, args.nproc_per_node))
        config = read_json_config(env_config_path)
        config[key] = value
        write_json_config(config, env_config_path)
        return env_config_path
    
    if args.profile_time == 0:
        comm_coe = all2all_message_size / per_comm_time * (1.024 ** 2)
        if rank == 0:
            print(f'{per_comm_time:.4f} ms, {comm_coe:.4f} GB/s')
        if return_result:
            final_result = {
                "rank": rank,
                "local_rank": local_rank,
                "data": {
                    f"{args.key}": float(comm_coe),
                },
            }
        elif rank == 0:
            print('**********')
            print(f'comm_coe_{args.pp_deg}_{tp_size}_{args.global_tp_consec}: {comm_coe:.4f}')
            print('**********')
            key = f'all2all_size_{tp_size}_consec_{args.global_tp_consec}'
            env_config_path = save_config('./hardware_configs/all2all_bandwidth_%dnodes_%dgpus_per_node.json', key, comm_coe)
            print(f'Already written all2all bandwidth into env config file {env_config_path}!')
    else:
        if rank == 0:
            print(f'Total time: {sum(time_list):.4f} ms, Measurements: {len(time_list)}')
            print('**********')
            print(f'comm_time_{args.local_batch_size}MB_{args.pp_deg}_{tp_size}: {per_comm_time:.4f} ms')
            print('**********')
            key = f'all2all_size_{tp_size}_{args.local_batch_size}MB_time'
            env_config_path = save_config('./hardware_configs/sp_time_%dnodes_%dgpus_per_node.json', key, per_comm_time)
            print(f'Already written all2all bandwidth into env config file {env_config_path}!')
    
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if return_result:
        return final_result

def ray_profile_all2all(args):
    try:
        result = train(args, return_result=True)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8,16,32,64,128,256],
    )
    parser.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8,16,32,64,128,256],
    )
    parser.add_argument(
        "--local_batch_size", type=int, default=32, help="local training batch size"
    )
    parser.add_argument(
        "--nproc_per_node", type=int, default=-1, help="Nproc per node",
    )
    parser.add_argument(
        "--profile_time", type=int, default=0, help="Profile time",
    )
    parser.add_argument("--local-rank" ,type=int,default=-1)
    args = parser.parse_args()
    train(args)