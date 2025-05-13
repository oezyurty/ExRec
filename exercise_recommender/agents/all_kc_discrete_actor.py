import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import csv
"""
    Discrete Actor acts on student hidden states and KC embeddings assigned to each student.
    Expected obs is the concatenation of student hidden states and kc_embeddings
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

class AllKCDiscreteActor(nn.Module):
    def __init__(self,
                student_hidden_size: int=300,
                hidden_size: int=300,
                up_projection_size: int=1200,
                num_questions: int=7652
                ):
        super().__init__()

        self.student_hidden_size = student_hidden_size
        self.hidden_size = hidden_size
        self.num_questions = num_questions

        self.hidden_state_fc = nn.Linear(self.student_hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_questions)
    
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        student_hidden_state = obs.to(device)
        hid_state_transformed = F.relu(self.hidden_state_fc(student_hidden_state))
        output = F.relu(self.output(hid_state_transformed)) 
        return output, state
        