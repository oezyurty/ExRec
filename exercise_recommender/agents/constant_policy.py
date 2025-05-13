from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    ReplayBuffer,
    SequenceSummaryStats,
    to_torch,
    to_torch_as,
)
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    BatchWithReturnsProtocol,
    DistBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import BasePolicy
from tianshou.policy.modelfree.pg import (
    PGTrainingStats,
    TDistFnDiscrOrCont,
    TPGTrainingStats,
)
from tianshou.utils import RunningMeanStd
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor
from tianshou.utils.torch_utils import policy_within_training_step, torch_train_mode


class ConstantPolicy(BasePolicy):
    """Implementation of REINFORCE algorithm."""

    
    def __init__(
        self, 
        action_space: gym.Space,
        observation_space: gym.Space
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space
        )


    def process_fn(
        self, 
        batch: RolloutBatchProtocol, 
        buffer: ReplayBuffer, 
        indices: np.ndarray
    ) -> BatchWithReturnsProtocol:
        """Compute the discounted returns for each transition."""
        return batch

    def forward(
        self, 
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any
    ) -> DistBatchProtocol:
        """Compute action over the given batch data."""
        student_num = batch.obs.shape[0]
        action = torch.zeros((student_num, 768))
        return Batch(act=action)

    def learn(
        self,
        batch: BatchWithReturnsProtocol, 
        batch_size: int | None, 
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TPGTrainingStats:
        """Perform the back-propagation."""
        return PGTrainingStats(loss=loss_summary_stat)