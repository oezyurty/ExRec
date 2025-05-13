import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
"""
    KC Critic acts on observations and actions to estimate the Q value.
    KC Critic accepts observations to be concatenated tensors of student
        hidden states and assigned KC embeddings.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

class KCCritic(nn.Module):
    def __init__(self,
                student_hidden_size: int=300,
                kc_emb_size: int=768,
                action_size: int=768,
                hidden_size: int=300,
                up_projection_size: int=1200,
                log_path:str = "",
            ):
        super().__init__()

        self.student_hidden_size = student_hidden_size 
        self.kc_emb_size = kc_emb_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.hidden_state_fc = nn.Linear(self.student_hidden_size, hidden_size)
        self.kc_emb_fc = nn.Linear(self.kc_emb_size, hidden_size)
        self.concat_fc = nn.Linear(2*hidden_size, hidden_size)
        self.action_fc = nn.Linear(self.action_size, hidden_size)
        self.fc1 = nn.Linear(2*hidden_size, up_projection_size)
        self.output = nn.Linear(up_projection_size, 1)
        self.log_path = log_path
    
    def forward(self, obs, actions, info=None):
        # TODO: Remove after debugging
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        obs = obs.to(device)
        actions = actions.to(device)
        student_hidden_state = obs[:, :self.student_hidden_size]
        kc_embs = obs[:, self.student_hidden_size:]
        hid_state_transformed = F.relu(self.hidden_state_fc(student_hidden_state))
        kc_embs_transformed = F.relu(self.kc_emb_fc(kc_embs))
        concat = torch.cat((hid_state_transformed, kc_embs_transformed), dim=1)
        concat_transformed = F.relu(self.concat_fc(concat))
        action_transformed = F.relu(self.action_fc(actions))
        action_state_concat = torch.cat((concat_transformed, action_transformed), dim=1)
        x = F.relu(self.fc1(action_state_concat))
        output = self.output(x)
        output = output.squeeze(-1)
        if self.log_path != "":
            with open(os.path.join(self.log_path, "critic_outputs.csv"), "a+") as f:
                np.savetxt(f, output.detach().cpu().numpy(), delimiter=",")
        # TODO: Remove after debugging
        #print(output.shape)
        return output