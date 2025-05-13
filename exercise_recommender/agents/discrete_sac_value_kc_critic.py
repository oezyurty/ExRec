import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
"""
    Value KC Critic acts on observations to estimate the V value.
    Value KC Critic accepts observations to be concatenated tensors of student
        hidden states and assigned KC embeddings.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

class DiscreteSACValueKCCritic(nn.Module):
    def __init__(self,
                student_hidden_size: int=300,
                kc_emb_size: int=768,
                hidden_size: int=300,
                up_projection_size: int=1200,
                log_path:str = "",
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
        self.output = nn.Linear(up_projection_size, num_questions)
        self.log_path = log_path
    
    def forward(self, obs, info=None):
        # TODO: Remove after debugging
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.to(device)
        student_hidden_state = obs[:, :self.student_hidden_size]
        kc_embs = obs[:, self.student_hidden_size:]
        hid_state_transformed = F.relu(self.hidden_state_fc(student_hidden_state))
        kc_embs_transformed = F.relu(self.kc_emb_fc(kc_embs))
        concat = torch.cat((hid_state_transformed, kc_embs_transformed), dim=1)
        x = F.relu(self.fc1(concat))
        output = self.output(x)
        if self.log_path != "":
            with open(os.path.join(self.log_path, "critic_outputs.csv"), "a+") as f:
                np.savetxt(f, output.detach().cpu().numpy(), delimiter=",")
        return output