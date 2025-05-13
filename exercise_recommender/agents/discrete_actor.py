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

class DiscreteActor(nn.Module):
    def __init__(self,
                student_hidden_size: int=300,
                kc_emb_size: int=768,
                hidden_size: int=300,
                up_projection_size: int=1200,
                num_questions: int=7652
                ):
        super().__init__()

        self.student_hidden_size = student_hidden_size
        self.kc_emb_size = kc_emb_size
        self.hidden_size = hidden_size
        self.num_questions = num_questions

        self.hidden_state_fc = nn.Linear(self.student_hidden_size, hidden_size)
        self.kc_emb_fc = nn.Linear(self.kc_emb_size, hidden_size)
        self.fc1 = nn.Linear(2*hidden_size, up_projection_size)
        self.fc2 = nn.Linear(up_projection_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_questions)
    
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.to(device)
        # Disect obs to student_hidden_state and kc_emb
        student_hidden_state = obs[:, :self.student_hidden_size]
        kc_embs = obs[:, self.student_hidden_size:]
        hid_state_transformed = F.relu(self.hidden_state_fc(student_hidden_state))
        kc_embs_transformed = F.relu(self.kc_emb_fc(kc_embs))
        concat = torch.cat((hid_state_transformed, kc_embs_transformed), dim=1)
        x = F.relu(self.fc1(concat))
        x = F.relu(self.fc2(x))
        output = F.relu(self.output(x))
        return output, state
        