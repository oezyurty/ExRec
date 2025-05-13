import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from exercise_recommender.wrappers.calibrationqdkt_wrapper import CalibrationQDKTWrapper
from pykt.models.qdkt import CalibrationQDKT
import copy

"""
    CriticDKT acts on observations and actions to estimate the Q value.
    CriticDKT accepts observations to be concatenated tensors of student
        hidden states and assigned KC embeddings.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

class CriticDKT(nn.Module):
    def __init__(self,
                path_dkt: str = "", 
                student_hidden_size: int=300,
                reward_scale: int= 1000,
                log_path:str = "",
            ):
        super().__init__()
        if path_dkt == "":
            raise ValueError("A path to a pretrained DKT model should be given to initialize this critic.")
        self.path_dkt = path_dkt
        
        self.student_hidden_size = student_hidden_size 
        self.reward_scale = reward_scale
        self.log_path = log_path

        self._init_dkt_layers()

    # LOAD MODULES FROM CalibrationDKT
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

    # Calculate next step's y_kc based on given response (0 or 1).
    def _calculate_value(self, student_hidden_state, actions, kc_embs, cell_state, response=0):
        # Get response tensor (0 or 1)
        r = torch.zeros(actions.shape[:-1]).unsqueeze(-1) + response
        r = r.to(device)

        correctness_encoding = self.correctness_encoding(r.long())
        x = actions + correctness_encoding
        cell_state = cell_state.unsqueeze(0)
        student_hidden_state = student_hidden_state.unsqueeze(0)
        self.lstm_layer.flatten_parameters()
        lstm_out, _ = self.lstm_layer(x, (student_hidden_state, cell_state))
        lstm_out = lstm_out[:, -1, :]

        y = self.prediction_layer(lstm_out, kc_embs)

        return y


    def forward(self, obs, actions, info={}):
        # Process obs and actions to tensor and device
        student_hidden_state = obs[:, :self.student_hidden_size]
        kc_embs = obs[:, self.student_hidden_size:]
        if not isinstance(student_hidden_state, torch.Tensor):
            student_hidden_state = torch.tensor(student_hidden_state, dtype=torch.float32)
        if not isinstance(kc_embs, torch.Tensor):
            kc_embs = torch.tensor(kc_embs, dtype=torch.float32)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)

        student_hidden_state = student_hidden_state.to(device)
        kc_embs = kc_embs.to(device)
        actions = actions.to(device)

        #student_hidden_state = obs[:, :self.student_hidden_size]
        #kc_embs = obs[:, self.student_hidden_size:]

        # Process info
        y_kc_before, y_que, cell_state = self._process_info(info)

        student_hidden_state = student_hidden_state.contiguous()

        value_0 = self._calculate_value(student_hidden_state, actions, kc_embs, cell_state, 0)
        value_1 = self._calculate_value(student_hidden_state, actions, kc_embs, cell_state, 1)

        value = y_que * value_1 + (1-y_que) * value_0

        #Scale the value to the correct range for each student 
        value = (value - y_kc_before) * self.reward_scale 
        #TODO: Remove after debugging
        #print(value.shape)
        return value

            