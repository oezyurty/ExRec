from tianshou.policy.modelfree.rainbow import RainbowTrainingStats, TRainbowTrainingStats, RainbowPolicy, _sample_noise
import torch
import numpy as np
from tianshou.data.types import RolloutBatchProtocol
from exercise_recommender.wrappers.c51_wrapper import C51PolicyWrapper
from typing import Any, TypeVar

device = "cuda" if torch.cuda.is_available() else "cpu"

class RainbowPolicyWrapper(C51PolicyWrapper):

    def learn(
        self,
        batch: RolloutBatchProtocol,
        *args: Any,
        **kwargs: Any,
    ) -> TRainbowTrainingStats:
        _sample_noise(self.model)
        if self._target and _sample_noise(self.model_old):
            self.model_old.train()  # so that NoisyLinear takes effect
        return super().learn(batch, **kwargs)
    
