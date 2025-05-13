import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from exercise_recommender.wrappers.calibrationqdkt_wrapper import CalibrationQDKTWrapper
from pykt.models.qdkt import CalibrationQDKT
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

class DiscreteSACValueCriticDKT(nn.Module):
    def __init__(self,
                path_dkt: str="",
                student_hidden_size: int=300,
                reward_scale: int=1000,
                log_path:str ="",
                num_questions:int=7652,
                up_projection_size: int=1200):
        if path_dkt == "":
            raise ValueError("A path to a pretrained DKT model should be given to initialize this critic.")
        
        super().__init__()
        self.path_dkt = path_dkt
        self.student_hidden_size = student_hidden_size 
        self.reward_scale = reward_scale
        self.log_path = log_path
        self.num_questions = num_questions

        self._init_dkt_layers()

        self.fc1_up = nn.Linear(1, up_projection_size)
        self.output = nn.Linear(up_projection_size, num_questions)
    
    def _init_dkt_layers(self):
        dkt_model = CalibrationQDKT()
        dkt_model = dkt_model.to(device)
        net = torch.load(self.path_dkt, map_location=device)
        print(f"Pretrained model mapped device: {device}")
        dkt_model.load_state_dict(net)
        dkt_model.model.model.eval()

        # Deepcopy the specific layers
        self.lstm_layer = copy.deepcopy(dkt_model.model.model.lstm_layer)
        self.prediction_layer = copy.deepcopy(dkt_model.model.model.prediction_layer)
        self.correctness_encoding = copy.deepcopy(dkt_model.model.model.correctness_encoding)

        # Move them to the correct device
        self.lstm_layer = self.lstm_layer.to(device)
        self.prediction_layer = self.prediction_layer.to(device)
        self.correctness_encoding = self.correctness_encoding.to(device)

        # Ensure they are trainable
        self.lstm_layer.train()
        self.prediction_layer.train()
        self.correctness_encoding.train()
    
    def _process_info(self, info):
        if "y_kc_before" not in info:
            raise ValueError("y_kc_before is not found in info")
        if "y_que" not in info:
            raise ValueError("y_que is not found in info")
        if "cell_state" not in info:
            raise ValueError("cell_state is not found in info")
        
        # Process y_kc and y_que to tensor and device
        y_kc_before = info["y_kc_before"]
        y_que = info["y_que"]
        cell_state = info["cell_state"]
        if not isinstance(y_kc_before, torch.Tensor):
            y_kc_before = torch.tensor(y_kc_before, dtype=torch.float32)
        if not isinstance(y_que, torch.Tensor):
            y_que = torch.tensor(y_que, dtype=torch.float32)
        if not isinstance(cell_state, torch.Tensor):
            cell_state = torch.tensor(cell_state, dtype=torch.float32)
        y_kc_before = y_kc_before.to(device)
        y_que = y_que.to(device)
        cell_state = cell_state.to(device)

        return y_kc_before, y_que, cell_state
    
    def _calculate_value(self, student_hidden_state, kc_embs):

        y = self.prediction_layer(student_hidden_state, kc_embs)
        y = y.view(-1, 1)

        return y
    
    def forward(self, obs, info={}):
        # Process obs to tensor and device
        student_hidden_state = obs[:, :self.student_hidden_size]
        kc_embs = obs[:, self.student_hidden_size:]
        if not isinstance(student_hidden_state, torch.Tensor):
            student_hidden_state = torch.tensor(student_hidden_state, dtype=torch.float32)
        if not isinstance(kc_embs, torch.Tensor):
            kc_embs = torch.tensor(kc_embs, dtype=torch.float32)

        student_hidden_state = student_hidden_state.to(device)
        kc_embs = kc_embs.to(device)

        #student_hidden_state = obs[:, :self.student_hidden_size]
        #kc_embs = obs[:, self.student_hidden_size:]

        # Process info
        y_kc_before, y_que, cell_state = self._process_info(info)

        student_hidden_state = student_hidden_state.contiguous()

        value = self._calculate_value(student_hidden_state, kc_embs)

        dkt_output = self._calculate_value(student_hidden_state, kc_embs) # output shape: batch_size x 1
        dkt_output = F.relu(self.fc1_up(dkt_output))
        action_logits = self.output(dkt_output)

        return action_logits
    

    
