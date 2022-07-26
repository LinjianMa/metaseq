# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import contextlib
import logging
import os
from typing import Optional

import torch

from metaseq.dataclass.configs import DistributedTrainingConfig
from metaseq.distributed import utils as dist_utils

logger = logging.getLogger(__name__)

try:
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from fairscale.utils.testing import DummyProcessGroup

    has_FSDP = True
except ImportError:
    FSDP = torch.nn.Module
    has_FSDP = False


class FullyShardedDataParallel(FSDP):
    """
    A small wrapper around PyTorch Distributed FullyShardedDataParallel (FSDP) with some
    metaseq-specific checkpoint saving/loading logic.

    Args:
        use_sharded_state (bool): if True, then ``state_dict`` will return
            ``FSDP.local_state_dict`` and ``load_state_dict`` will call
            ``FSDP.load_local_state_dict``. Otherwise, ``state_dict`` will
            return the full model weights on data parallel rank 0 (empty on
            other ranks) and ``load_state_dict`` will broadcast model weights
            from rank 0 to other ranks.
    """

    def __init__(self, *args, use_sharded_state: bool = False, **kwargs):
        if not has_FSDP:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install PyTorch with: pip3 install torch torchvision torchaudio"
            )
        super().__init__(*args, **kwargs)
        self.use_sharded_state = use_sharded_state

    @property
    def unwrapped_module(self) -> torch.nn.Module:
        return self.module.module

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if self.use_sharded_state:
            with FSDP.state_dict_type(self, StateDictType.LOCAL_STATE_DICT):
                return super().state_dict(
                    destination=destination, prefix=prefix, keep_vars=keep_vars
                )
        else:
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=False, rank0_only=True
            )
            with FSDP.state_dict_type(
                self, StateDictType.FULL_STATE_DICT, full_state_dict_config
            ):
                return super().state_dict(
                    destination=destination, prefix=prefix, keep_vars=keep_vars
                )

    def load_state_dict(self, state_dict, strict=None, model_cfg=None):
        if self.use_sharded_state:
            with FSDP.state_dict_type(self, StateDictType.LOCAL_STATE_DICT):
                return super().load_state_dict(state_dict)
        else:
            if not isinstance(self.process_group, DummyProcessGroup):
                state_dict = dist_utils.broadcast_object(
                    state_dict, src_rank=0, group=self.process_group
                )
            return super().load_state_dict(state_dict)


@contextlib.contextmanager
def fsdp_enable_wrap(
    cfg: DistributedTrainingConfig, use_sharded_state: bool = False, **kwargs
):
    try:
        from torch.distributed.fsdp.wrap import enable_wrap
    except ImportError:
        raise ImportError(
            "Cannot find FullyShardedDataParallel. "
            "Please install PyTorch with: pip3 install torch torchvision torchaudio"
        )
    if cfg.memory_efficient_fp16:
        assert cfg.fp16  # memory_efficient_fp16 should imply fp16
    group = dist_utils.get_data_parallel_group()
    if group is None and cfg.distributed_world_size == 1:
        group = DummyProcessGroup(rank=0, size=1)
    fsdp_config = {
        "process_group": group,
        "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP
        if cfg.no_reshard_after_forward
        else ShardingStrategy.FULL_SHARD,  # SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
        "mixed_precision": MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32 if cfg.fp32_reduce_scatter else torch.float16,
            buffer_dtype=torch.float16,
        )
        if cfg.fp16 and not cfg.memory_efficient_fp16
        else None,
        "cpu_offload": CPUOffload(offload_params=True) if cfg.cpu_offload else None,
        "backward_prefetch": None
        if cfg.backward_prefetch is None
        else BackwardPrefetch.BACKWARD_PRE
        if cfg.backward_prefetch == "pre"
        else BackwardPrefetch.BACKWARD_POST,
        **kwargs,
    }
    with enable_wrap(
        wrapper_cls=FullyShardedDataParallel,
        use_sharded_state=use_sharded_state,
        **fsdp_config,
    ):
        yield


def construct_fsdp(
    model: torch.nn.Module, cfg: DistributedTrainingConfig, use_sharded_state: bool = False, **kwargs
):
    if cfg.memory_efficient_fp16:
        assert cfg.fp16  # memory_efficient_fp16 should imply fp16
    group = dist_utils.get_data_parallel_group()
    if group is None and cfg.distributed_world_size == 1:
        group = DummyProcessGroup(rank=0, size=1)
    fsdp_config = {
        "process_group": group,
        "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP
        if cfg.no_reshard_after_forward
        else ShardingStrategy.FULL_SHARD,  # SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
        "mixed_precision": MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32 if cfg.fp32_reduce_scatter else torch.float16,
            buffer_dtype=torch.float16,
        )
        if cfg.fp16 and not cfg.memory_efficient_fp16
        else None,
        "cpu_offload": CPUOffload(offload_params=True) if cfg.cpu_offload else None,
        "backward_prefetch": None
        if cfg.backward_prefetch is None
        else BackwardPrefetch.BACKWARD_PRE
        if cfg.backward_prefetch == "pre"
        else BackwardPrefetch.BACKWARD_POST,
        **kwargs,
    }
    from metaseq.model_parallel.modules import (
        ModelParallelTransformerDecoderLayer,
    )
    recursive_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ModelParallelTransformerDecoderLayer},
    )
    from torch.distributed.fsdp.wrap import (
        always_wrap_policy,
        ParamExecOrderPolicy,
        HandleInitMode,
    )
    nonrecursive_policy = ParamExecOrderPolicy(
        handle_init_mode=HandleInitMode.MODULE_LEVEL,
        bucket_size=int(17000000 * 50 + 1),
        module_level_group_policy=always_wrap_policy,
    )
    logger.info(f"fsdp_config is {fsdp_config}")
    fsdp_model = FullyShardedDataParallel(
        model,
        use_sharded_state=use_sharded_state,
        auto_wrap_policy=nonrecursive_policy,
        **fsdp_config,
    )
    # activation checkpointing
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing_wrapper,
    )
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, ModelParallelTransformerDecoderLayer)
    apply_activation_checkpointing_wrapper(
        fsdp_model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
    return fsdp_model


def fsdp_wrap(module, min_num_params: Optional[int] = None, **kwargs):
    """
    Helper to wrap layers/modules in FSDP. This falls back to a no-op if
    fairscale is not available.

    Args:
        module (nn.Module): module to (maybe) wrap
        min_num_params (int, Optional): minimum number of layer params to wrap
    """
    return module
    # try:
    #     from torch.distributed.fsdp.wrap import wrap

    #     if os.environ.get("RESHARD_OVERRIDE_PROCESS_GROUP", "False") == "True":
    #         logger.info("Process group was None, overriding to DummyProcessGroup")
    #         kwargs["process_group"] = DummyProcessGroup(rank=0, size=1)

    #     if min_num_params is not None:
    #         logger.info(f"start to wrap {module} with {min_num_params}")
    #         num_params = sum(p.numel() for p in module.parameters())
    #         if num_params >= min_num_params:
    #             return wrap(module, **kwargs)
    #         else:
    #             return module
    #     else:
    #         logger.info(f"start to wrap {module}")
    #         return wrap(module, **kwargs)
    # except ImportError:
    #     return module
