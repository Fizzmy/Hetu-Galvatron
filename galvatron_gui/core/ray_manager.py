"""Ray cluster management and remote execution"""

import ray
import subprocess
import socket
import os
from pathlib import Path
from typing import Dict, Any, Optional

def get_node_info() -> Dict[str, str]:
    """Get execution node information"""
    try:
        context = ray.get_runtime_context()
        return {
            "node_ip": ray.util.get_node_ip_address(),
            "node_id": context.get_node_id()[:16],
            "hostname": socket.gethostname(),
            "pid": os.getpid()
        }
    except:
        return {"node_ip": "unknown", "hostname": "unknown"}

@ray.remote
def get_master_addr_port() -> tuple[str, str]:
    addr = ray.util.get_node_ip_address().strip("[]")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    return addr, str(port)

from ray.util.placement_group import placement_group, PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
class RayClusterManager:
    """Manages Ray cluster connection and information"""
    
    def __init__(self):
        self.cluster_info = None
        self.groups = {}
    
    def initialize(self, address: str = None) -> str:
        """Initialize Ray cluster connection"""
        if ray.is_initialized():
            return "Ray already initialized"
        
        try:
            if address:
                ray.init(address=address, ignore_reinit_error=True)
            else:
                try:
                    ray.init(address='auto', ignore_reinit_error=True)
                except:
                    return self._get_init_error_message()
            
            self.cluster_info = self.get_cluster_info()
            return "Ray initialized successfully"
        except Exception as e:
            import traceback
            return f"Failed to initialize Ray: {e}\n\n{traceback.format_exc()}"
    
    def _get_init_error_message(self) -> str:
        """Get detailed error message for initialization failure"""
        return """Failed to connect to Ray cluster

Please start Ray cluster first:

1. Start Ray on this machine:
   ```
   ray start --head --dashboard-host=0.0.0.0
   ```

2. Check Ray status:
   ```
   ray status
   ```

3. Then click 'Initialize Ray Cluster' again

For multi-node:
- On head node: `ray start --head --port=6379 --dashboard-host=0.0.0.0`
- On worker nodes: `ray start --address=<head-ip>:6379`
- In GUI, enter: `ray://<head-ip>:10001`
"""
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get Ray cluster information with node and GPU details"""
        if not ray.is_initialized():
            return {"error": "Ray not initialized"}
        
        try:
            resources = ray.cluster_resources()
            available = ray.available_resources()
            nodes = ray.nodes()
            
            alive_nodes = [n for n in nodes if n['Alive']]
            
            # Calculate GPUs per node (assume homogeneous cluster)
            gpus_per_node_list = []
            for node in alive_nodes:
                node_gpus = int(node['Resources'].get('GPU', 0))
                if node_gpus > 0:
                    gpus_per_node_list.append(node_gpus)
            
            gpus_per_node = gpus_per_node_list[0] if gpus_per_node_list else 0
            num_nodes = len(alive_nodes)
            
            return {
                "total_cpus": resources.get('CPU', 0),
                "total_gpus": resources.get('GPU', 0),
                "available_cpus": available.get('CPU', 0),
                "available_gpus": available.get('GPU', 0),
                "num_nodes": num_nodes,
                "gpus_per_node": gpus_per_node,
                "nodes": [
                    {
                        "node_id": n['NodeID'][:16],
                        "ip": n['NodeManagerAddress'],
                        "alive": n['Alive'],
                        "resources": n['Resources'],
                        "gpus": int(n['Resources'].get('GPU', 0))
                    }
                    for n in alive_nodes
                ]
            }
        except:
            return {"error": "Failed to get cluster info"}
    
    def format_cluster_info(self, info: Dict[str, Any]) -> str:
        """Format cluster information for display"""
        if 'error' in info:
            return f"Failed: {info['error']}"
        
        display = """Ray Cluster Initialized

**Cluster Configuration:**
- Total GPUs: {total_gpus}
- GPUs per Node: {gpus_per_node}
- Total Nodes: {num_nodes}
- Total CPUs: {total_cpus}

**Available Resources:**
- Available GPUs: {available_gpus}
- Available CPUs: {available_cpus}

**Node Details:**
""".format(**info)
        
        for node in info['nodes']:
            display += f"\n- Node {node['ip']}"
            display += f"\n  - GPUs: {node.get('gpus', 0)}"
            display += f"\n  - CPUs: {node['resources'].get('CPU', 0)}"
        
        return display

    def create_placement_group_for_task(self, task_id: str, num_gpus: int = None, num_cpus: int = None) -> PlacementGroup:
        """
        Dynamically create a placement group for a specific task (non-blocking)
        
        Args:
            task_id: Unique task identifier
            num_gpus: Number of GPUs needed (defaults to gpus_per_node)
        
        Returns:
            PlacementGroup object (may not be ready yet)
        """
        if task_id in self.groups:
            return self.groups[task_id]

        if num_cpus is None:
            bundle = {"CPU": 1, "GPU": 1}
            pg_scheme = [bundle.copy() for _ in range(num_gpus)]
            
            pg = placement_group(
                bundles=pg_scheme, 
                strategy="STRICT_PACK", 
                name=f"pg_{task_id}", 
            )
            self.groups[task_id] = pg
            print(f"📦 Created placement group for task {task_id} (requires {num_gpus} GPUs)")
        else:
            bundle = {"CPU": num_cpus}
            pg = placement_group(
                bundles=[bundle],
                strategy="STRICT_PACK",
                name=f"pg_{task_id}",
            )
            self.groups[task_id] = pg
            print(f"📦 Created placement group for task {task_id} (requires {num_cpus} CPUs)")
        return pg
    
    def check_placement_group_ready(self, task_id: str, timeout: float = 0.01) -> Optional[PlacementGroup]:
        """
        Check if a placement group is ready
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait (default 0.5s for quick check)
        
        Returns:
            PlacementGroup if ready, None if not ready yet
            
        Raises:
            RuntimeError: If placement group doesn't exist (should create first)
        """
        if task_id not in self.groups:
            raise RuntimeError(f"Placement group for task {task_id} not found. Call create_placement_group_for_task first.")
        
        pg = self.groups[task_id]
        
        # Check with short timeout
        ready_refs = pg.ready()
        ready, _ = ray.wait([ready_refs], timeout=timeout)
        
        # Debug output
        # pg_state = ray.util.placement_group_table().get(pg.id.hex(), {})
        # state_str = pg_state.get("state", "UNKNOWN")
        # print(f"🔍 PG check for {task_id}: state={state_str}, ready={len(ready) > 0}")
        
        if not ready:
            # Not ready yet, return None (not an error, just wait)
            return None
        
        print(f"✅ Placement group {task_id} is ready!")
        return pg
    
    def remove_placement_group(self, task_id: str):
        """
        Remove and cleanup a placement group after task completion
        
        Args:
            task_id: Task identifier
        """
        if task_id not in self.groups:
            print(f"⚠️  Placement group for task {task_id} not found")
            return
        
        pg = self.groups[task_id]
        try:
            # Remove the placement group to free resources
            ray.util.remove_placement_group(pg)
            del self.groups[task_id]
            print(f"🗑️  Removed placement group for task {task_id}")
        except Exception as e:
            print(f"⚠️  Error removing placement group for task {task_id}: {e}")
            # Still remove from our tracking dict
            del self.groups[task_id]
    
    def get_master_addr_port(self, pg: PlacementGroup):
        return ray.get(
            get_master_addr_port.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote()
        )
