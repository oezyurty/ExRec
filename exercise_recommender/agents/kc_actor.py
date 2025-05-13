import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import csv
"""
    KC Actor acts on student hidden states and KC embeddings assigned to each student.
    Expected obs is the concatenation of student hidden states and kc_embeddings
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

class KCActor(nn.Module):
    def __init__(self,
                student_hidden_size: int=300,
                kc_emb_size: int=768,
                action_size: int=768,
                hidden_size: int=300,
                up_projection_size: int=1200
                ):
        super().__init__()

        self.student_hidden_size = student_hidden_size
        self.kc_emb_size = kc_emb_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.hidden_state_fc = nn.Linear(self.student_hidden_size, hidden_size)
        self.kc_emb_fc = nn.Linear(self.kc_emb_size, hidden_size)
        self.fc1 = nn.Linear(2*hidden_size, up_projection_size)
        self.output = nn.Linear(up_projection_size, action_size)
    
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
        output = self.output(x)
        norms = output.norm(p=2, dim=1, keepdim=True)
        output = output/norms
        output = output * np.sqrt(output.shape[1])
        return output, state

        