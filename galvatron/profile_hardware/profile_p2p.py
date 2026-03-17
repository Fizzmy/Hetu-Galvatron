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

def single_p2p_send_recv(input_tensor, prev_rank, next_rank, rank, pp_rank_in_group, pp_size):
    """Perform point-to-point communication using async P2P ops (similar to actual pipeline parallel)
    
    This mimics the actual p2p communication pattern used in PipelineParallel:
    - Each stage sends to next stage and receives from previous stage
    - Uses async isend/irecv operations
    - Uses P2POp and batch_isend_irecv for better performance
    - Ensures tensor is contiguous before sending
    
    Args:
        input_tensor: Input tensor to send
        prev_rank: Previous rank in pipeline (None if first stage)
        next_rank: Next rank in pipeline (None if last stage)
        rank: Current rank
        pp_rank_in_group: Pipeline rank within the group (0 to pp_size-1)
        pp_size: Pipeline parallel size
    """
    ops = []
    
    # Send to next stage (if not last stage)
    if next_rank is not None:
        send_op = dist.P2POp(
            dist.isend,
            input_tensor.contiguous(),
            next_rank
        )
        ops.append(send_op)
    
    # Receive from previous stage (if not first stage)
    if prev_rank is not None:
        output = torch.empty_like(input_tensor)
        recv_op = dist.P2POp(
            dist.irecv,
            output,
            prev_rank
        )
        ops.append(recv_op)
    else:
        output = None
    
    # Execute all P2P operations
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    
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
    pp_size = args.pp_deg
    batch_size = args.local_batch_size
    p2p_message_size = batch_size * SEQ_LEN * HIDDEN_SIZE * BYTES_PER_FLOAT16 / MB_TO_BYTES
    
    # Determine pipeline ranks: PP groups are UNCONSECUTIVE
    # For pp_size=2, world_size=8: groups are [0,4], [1,5], [2,6], [3,7]
    # Formula: ranks = range(i, world_size, num_pp_groups) where num_pp_groups = world_size // pp_size
    num_pp_groups = world_size // pp_size
    pp_group_id = rank % num_pp_groups  # Which pipeline group this rank belongs to
    pp_rank_in_group = rank // num_pp_groups  # Position within the pipeline group (0 to pp_size-1)
    
    # Calculate prev and next ranks in pipeline (unconsecutive)
    # First stage has no previous rank, last stage has no next rank
    if pp_rank_in_group == 0:
        prev_rank = None
    else:
        # Previous rank in the same pipeline group
        prev_rank = rank - num_pp_groups
    
    if pp_rank_in_group == pp_size - 1:
        next_rank = None
    else:
        # Next rank in the same pipeline group
        next_rank = rank + num_pp_groups
    
    if local_rank == 0:
        print(f'Strategy: pp_deg = {pp_size}')
        print(f'[p2p_message_size]: {p2p_message_size:.2f} MB')
        print(f'Pipeline stages: {pp_size}, Current rank {rank} is stage {pp_rank_in_group}')
        if prev_rank is not None:
            print(f'  Receives from rank {prev_rank}')
        if next_rank is not None:
            print(f'  Sends to rank {next_rank}')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_list = []
    
    # Warm-up: directly allocate on GPU
    for _ in range(WARMUP_ITERATIONS):
        input_tensor = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        single_p2p_send_recv(input_tensor, prev_rank, next_rank, rank, pp_rank_in_group, pp_size)
    
    # Profiling loop
    for _ in range(PROFILE_ITERATIONS):
        input_tensor = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        start.record()
        for __ in range(ITERATIONS_PER_MEASUREMENT):
            single_p2p_send_recv(input_tensor, prev_rank, next_rank, rank, pp_rank_in_group, pp_size)
        end.record()
        torch.cuda.synchronize()
        # Only measure time for ranks that participate in communication
        if prev_rank is not None or next_rank is not None:
            time_list.append(start.elapsed_time(end) / ITERATIONS_PER_MEASUREMENT)
    
    # Collect and average communication time across all participating ranks
    if prev_rank is not None or next_rank is not None:
        time_list = sorted(time_list)
        per_comm_time = sum(time_list[TRIM_EDGES:-TRIM_EDGES]) / len(time_list[TRIM_EDGES:-TRIM_EDGES])
        per_comm_time = torch.tensor([per_comm_time]).to(device)
        # Average across all participating ranks (all ranks except possibly first/last)
        torch.distributed.all_reduce(per_comm_time, op=torch.distributed.ReduceOp.SUM)
        per_comm_time = per_comm_time.cpu().numpy()[0] / world_size * (1.024 ** 2)
        comm_coe = p2p_message_size / per_comm_time
    else:
        per_comm_time = 0.0
        comm_coe = 0.0
    
    def save_config(filename_template, key, value):
        """Helper function to save configuration to file"""
        path = os.path.dirname(os.path.abspath(__file__))
        env_config_path = os.path.join(path, filename_template % (node_num, args.nproc_per_node))
        config = read_json_config(env_config_path)
        config[key] = value
        write_json_config(config, env_config_path)
        return env_config_path
    final_result = {}
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
        else:
            print('**********')
            print(f'p2p_coe_pp_deg_{pp_size}: {comm_coe:.4f}')
            print('**********')
            key = f'pp_size_{pp_size}'
            env_config_path = save_config('./hardware_configs/p2p_bandwidth_%dnodes_%dgpus_per_node.json', key, comm_coe)
            print(f'Already written p2p bandwidth into env config file {env_config_path}!')
    
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if return_result:
        return final_result

def ray_profile_p2p(args):
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
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8,16,32,64,128,256],
    )
    parser.add_argument(
        "--local_batch_size", type=int, default=32, help="local training batch size"
    )
    parser.add_argument(
        "--nproc_per_node", type=int, default=-1, help="Nproc per node",
    )
    parser.add_argument("--local-rank" ,type=int,default=-1)
    args = parser.parse_args()
    train(args)