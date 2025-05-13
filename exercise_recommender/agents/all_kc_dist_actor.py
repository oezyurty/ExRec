import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import csv
"""
    Dist Actor acts on student hidden states and KC embeddings assigned to each student.
    Expected obs is the concatenation of student hidden states and kc_embeddings
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

class AllKCDistActor(nn.Module):
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
        self.output1 = nn.Linear(hidden_size, action_size)
        self.output2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        student_hidden_state = obs.to(device)
        hid_state_transformed = F.relu(self.hidden_state_fc(student_hidden_state))
        output1 = self.output1(hid_state_transformed)
        norms = output1.norm(p=2, dim=1, keepdim=True)
        output1 = output1/norms
        output1 = output1 * np.sqrt(output1.shape[1])
        output2 = torch.sigmoid(self.output2(hid_state_transformed)) * 0.2
        return (output1, output2), state

        