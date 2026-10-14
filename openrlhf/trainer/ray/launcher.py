import logging
import os
import socket
from typing import Dict, Optional, Type
import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm
from openrlhf.models import Actor, PauseLM
from openrlhf.trainer.ray.utils import ray_noset_visible_devices
from openrlhf.utils.deepspeed import DeepspeedStrategy

_RAY_ACTOR_GROUP_PORT_COUNTER = 0


def _next_actor_group_master_port():
    base = os.environ.get("OPENRLHF_ACTOR_MASTER_PORT_BASE")
    if not base:
        return None
    global _RAY_ACTOR_GROUP_PORT_COUNTER
    stride = int(os.environ.get("OPENRLHF_ACTOR_MASTER_PORT_STRIDE", "10"))
    port = int(base) + _RAY_ACTOR_GROUP_PORT_COUNTER * stride
    _RAY_ACTOR_GROUP_PORT_COUNTER += 1
    if not 1024 <= port <= 65535:
        raise ValueError(
            f"OPENRLHF actor master port {port} is outside [1024, 65535]. Adjust OPENRLHF_ACTOR_MASTER_PORT_BASE or OPENRLHF_ACTOR_MASTER_PORT_STRIDE."
        )
    return port


def _ray_actor_type_name(ray_actor_type) -> str:
    metadata = getattr(ray_actor_type, "__ray_metadata__", None)
    class_name = getattr(metadata, "class_name", None)
    if class_name:
        return class_name
    return getattr(ray_actor_type, "__name__", ray_actor_type.__class__.__name__)


class BaseDistributedActor:
    def __init__(self, world_size, rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = (
            str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"
        )

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return (self._master_addr, self._master_port)


class BaseModelActor(BaseDistributedActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def execute_batch(self, method_name: str, all_data, start_idx, end_idx):
        """Process input data by calling specified function for each item in the lists.

        Args:
            method_name (str): Name of the function to execute
            kwargs: Reference to the chunk of data to process

        Returns:
            List[Any]: List of results from function execution
        """
        kwargs = {key: value[start_idx:end_idx] for (key, value) in all_data.items()}
        first_param = next(iter(kwargs.values()))
        list_length = len(first_param)
        for (param_name, param_value) in kwargs.items():
            if len(param_value) != list_length:
                raise ValueError(
                    f"Parameter {param_name} has length {len(param_value)}, expected {list_length}"
                )
        func = getattr(self, method_name)
        if not callable(func):
            raise ValueError(f"Function {method_name} is not callable")
        results = []
        for i in tqdm(
            range(list_length),
            desc=f"{method_name}",
            disable=not self.strategy.is_rank_0(),
        ):
            sample_kwargs = {
                param_name: param_value[i]
                for (param_name, param_value) in kwargs.items()
            }
            result = func(**sample_kwargs)
            results.append(result)
        return results


@ray.remote(num_gpus=1)
class ReferenceModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(
                offload=strategy.args.ref_reward_offload
            ),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(model)
        if strategy.args.ref_reward_offload:
            model._offload = True
        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")


@ray.remote(num_gpus=1)
class PauseReferenceModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = PauseLM(
            pretrain,
            strategy=strategy,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(
                offload=strategy.args.ref_reward_offload
            ),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
            pause_token=strategy.args.pause_token,
            pause_init_method=strategy.args.pause_init_method,
            pause_tokenizer_save_path=strategy.args.pause_tokenizer_save_path,
            use_fast=not strategy.args.disable_fast_tokenizer,
            padding_side="left",
        )
        strategy.print(model)
        if strategy.args.ref_reward_offload:
            model._offload = True
        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")


class RayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BaseModelActor]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BaseModelActor],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        duplicate_actors: int = 1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
        bundle_offset: int = 0,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        self.duplicate_actors = duplicate_actors
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node
        self.bundle_offset = bundle_offset
        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        master_port = _next_actor_group_master_port()
        if master_port is not None:
            print(
                f"[RayActorGroup] {_ray_actor_type_name(self.ray_actor_type)} uses distributed master_port={master_port}"
            )
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [
                {"GPU": 1, "CPU": 1}
                for _ in range(self._num_nodes * self._num_gpus_per_node)
            ]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=self.bundle_offset
                ),
            ).remote(world_size, 0, None, master_port)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, None, master_port)
        self._actor_handlers = [master_actor]
        if world_size > 1:
            (master_addr, master_port) = ray.get(
                master_actor.get_master_addr_port.remote()
            )
            for rank in range(1, world_size):
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg, placement_group_bundle_index=rank
                        ),
                    ).remote(world_size, rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(self, *args, **kwargs):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [
            actor.init_model_from_pretrained.remote(*args, **kwargs)
            for actor in self._actor_handlers
        ]

    def async_save_model(self):
        """Save actor model on rank 0.

        Returns:
            List: list of remote object refs.
        """
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs

    def async_run_method_batch(self, method_name, **kwargs):
        """Run method on all actors with batched input data asynchronously using round-robin scheduling.
        Each actor processes one chunk of data at a time. Actors in the same ring / tensor parallel group process the same chunk.

        Args:
            method_name (str): Name of the method to run
            **kwargs: Keyword arguments for the method. Each value should be a list/tensor of the same length.

        Returns:
            List[ray.ObjectRef]: List of remote object references to the results
        """
        for (key, value) in kwargs.items():
            if not hasattr(value, "__len__"):
                raise ValueError(f"Parameter {key} must be iterable")
        first_param = next(iter(kwargs.values()))
        total_length = len(first_param)
        for (key, value) in kwargs.items():
            if len(value) != total_length:
                raise ValueError(
                    f"All parameters must have the same length. {key} has length {len(value)}, expected {total_length}"
                )
        num_actors = len(self._actor_handlers)
        effective_actors = num_actors // self.duplicate_actors
        chunk_size = total_length // effective_actors
        assert (
            total_length >= effective_actors
        ), f"Total length {total_length} must be greater than or equal to effective actors {effective_actors}"
        if total_length % effective_actors != 0:
            chunk_size += 1
        all_data_ref = ray.put(kwargs)
        refs = []
        for chunk_idx in range(effective_actors):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_length)
            for j in range(self.duplicate_actors):
                actor_idx = chunk_idx * self.duplicate_actors + j
                actor = self._actor_handlers[actor_idx]
                refs.append(
                    actor.execute_batch.remote(
                        method_name, all_data_ref, start_idx, end_idx
                    )
                )
        return refs
