import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from exercise_recommender.wrappers.calibrationqdkt_wrapper import CalibrationQDKTWrapper
from pykt.models.qdkt import CalibrationQDKT
import copy

"""
    ValueCriticDKT acts on observations to estimate the V value.
    ValueCriticDKT accepts observations to be concatenated tensors of student
        hidden states and assigned KC embeddings.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

class AllKCValueCriticDKT(nn.Module):
    def __init__(self,
                path_dkt: str = "", 
                student_hidden_size: int=300,
                reward_scale: int= 1000,
                log_path:str = "",
                cluster_embs=None,
                cluster_batch_size=512
            ):
        print("HERE")
        if cluster_embs is None:
            raise ValueError("cluster_embs cannot be empty to use AllKCValueCriticDKT model")
        if path_dkt == "":
            raise ValueError("A path to a pretrained DKT model should be given to initialize this critic.")
        super().__init__()
        self.path_dkt = path_dkt
        self.cluster_embs = cluster_embs
        self.student_hidden_size = student_hidden_size 
        self.reward_scale = reward_scale
        self.log_path = log_path
        self.cluster_batch_size = cluster_batch_size

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


    def _calculate_value(self, student_hidden_state):
        num_students = student_hidden_state.shape[0]
        num_clusters = self.cluster_embs.shape[0]
        student_hidden_state_expanded = student_hidden_state.unsqueeze(1).expand(-1, num_clusters, -1)
        all_preds = []
        for start in range(0, num_clusters, self.cluster_batch_size):
            end = min(start+self.cluster_batch_size, num_clusters)
            kc_emb_batched = self.cluster_embs[start:end].unsqueeze(0).expand(num_students, -1, -1)
            kc_emb_batched = kc_emb_batched.reshape(-1, 768)
            student_hidden_state_batched = student_hidden_state_expanded[:, start:end, :]
            student_hidden_state_batched = student_hidden_state_batched.reshape(-1, 300)
            preds_batched = self.prediction_layer(student_hidden_state_batched, kc_emb_batched)
            preds_batched = preds_batched.cpu().reshape(num_students, end-start)
            all_preds.append(preds_batched)
            del kc_emb_batched, student_hidden_state_batched, preds_batched
            torch.cuda.empty_cache()
        
        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1).to("cuda")

        return mean_preds_per_student


    def forward(self, obs, info={}):
        # Process obs to tensor and device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        student_hidden_state = obs.to(device)

        #student_hidden_state = obs[:, :self.student_hidden_size]
        #kc_embs = obs[:, self.student_hidden_size:]

        # Process info
        y_kc_before, y_que, cell_state = self._process_info(info)

        student_hidden_state = student_hidden_state.contiguous()

        value = self._calculate_value(student_hidden_state)

        return value

            