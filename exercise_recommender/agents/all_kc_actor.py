import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import csv
"""
    KC Actor acts on student hidden states. 
    KC Embeddings should not be part of the observation.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

class AllKCActor(nn.Module):
    def __init__(self,
                student_hidden_size: int=300,
                action_size: int=768,
                hidden_size: int=300,
                up_projection_size: int=1200
                ):
        super().__init__()

        self.student_hidden_size = student_hidden_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.hidden_state_fc = nn.Linear(self.student_hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_size)
    
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.shape[1] != self.student_hidden_size:
            raise ValueError(f"Dim=1 should be equal to {self.student_hidden_size}, given: {obs.shape[1]}")
        obs = obs.to(device)
        # Disect obs to student_hidden_state and kc_emb
        hid_state_transformed = F.relu(self.hidden_state_fc(obs))
        output = self.output(hid_state_transformed)
        norms = output.norm(p=2, dim=1, keepdim=True)
        output = output/norms
        output = output * np.sqrt(output.shape[1])
        return output, state

        