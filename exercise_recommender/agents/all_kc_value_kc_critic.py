import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
"""
    AllKC Value KC Critic acts on observations to estimate the V value.
    All KC Value KC Critic accepts observations to be student hidden states
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

class AllKCValueKCCritic(nn.Module):
    def __init__(self,
                student_hidden_size: int=300,
                hidden_size: int=300,
                up_projection_size: int=1200,
                log_path:str = "",
            ):

        super().__init__()

        self.student_hidden_size = student_hidden_size 
        self.hidden_size = hidden_size

        self.hidden_state_fc = nn.Linear(self.student_hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.log_path = log_path
    
    def forward(self, obs, info=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        student_hidden_state = obs.to(device)
        hid_state_transformed = F.relu(self.hidden_state_fc(student_hidden_state))
        output = self.output(hid_state_transformed)
        if self.log_path != "":
            with open(os.path.join(self.log_path, "critic_outputs.csv"), "a+") as f:
                np.savetxt(f, output.detach().cpu().numpy(), delimiter=",")
        return output