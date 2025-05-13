import torch
import numpy as np
import os

from .dkt import DKT
from .dkt_plus import DKTPlus
from .dkvmn import DKVMN
from .deep_irt import DeepIRT
from .sakt import SAKT
from .saint import SAINT
from .kqn import KQN
from .atkt import ATKT
from .dkt_forget import DKTForget
from .akt import AKT
from .gkt import GKT
from .gkt_utils import get_gkt_graph
from .lpkt import LPKT
from .lpkt_utils import generate_qmatrix
from .skvmn import SKVMN
from .hawkes import HawkesKT
from .iekt import IEKT
from .atdkt import ATDKT
from .simplekt import simpleKT
from .bakt_time import BAKTTime
from .qdkt import QDKT, EmbeddedQueDKT, CalibrationQDKT
from .qikt import QIKT
from .dimkt import DIMKT
from .sparsekt import sparseKT
from .rkt import RKT
from .cskt import CausalSimpleKT
from .akt_que import AKTQue
from .iekt_que import IEKTQue
from .qikt_que import QIKTQue
from .simplekt_que import SimpleKTQue
from .sparsekt_que import SparseKTQue
from .kqn_que import KQNQue
from .atkt_que import ATKTQue
from .deep_irt_que import DeepIRTQue
from .dkvmn_que import DKVMNQue
from .skvmn_que import SKVMNQue
from .dkt_plus_que import DKTPlusQue
from .atdkt_que import ATDKTQue
from .sakt_que import SAKTQue
from .saint_que import SAINTQue

def get_device():
    if torch.backends.mps.is_available():  # Check for Apple Silicon GPU support
        return torch.device("mps")
    elif torch.cuda.is_available():  # Check for CUDA GPU support
        return torch.device("cuda")
    else:  # Fallback to CPU if neither MPS nor CUDA is available
        return torch.device("cpu")

device = get_device()
#device = torch.device("cpu")

def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "calibration_qdkt":
        model = CalibrationQDKT(**model_config, emb_type=emb_type).to(device)
    elif model_name == "embedded_que_dkt":
        model = EmbeddedQueDKT(**model_config, emb_type=emb_type).to(device)
    elif model_name == "dkt":
        model = DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "deep_irt":
        model = DeepIRT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt":
        model = SAKT(data_config["num_c"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "saint":
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        model = DKTForget(data_config["num_c"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    elif model_name == "akt":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "kqn":
        model = KQN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "atkt":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=False).to(device)
    elif model_name == "atktfix":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=True).to(device)
    elif model_name == "gkt":
        graph_type = model_config['graph_type']
        fname = f"gkt_graph_{graph_type}.npz"
        graph_path = os.path.join(data_config["dpath"], fname)
        if os.path.exists(graph_path):
            graph = torch.tensor(np.load(graph_path, allow_pickle=True)['matrix']).float()
        else:
            graph = get_gkt_graph(data_config["num_c"], data_config["dpath"], 
                    data_config["train_valid_original_file"], data_config["test_original_file"], graph_type=graph_type, tofile=fname)
            graph = torch.tensor(graph).float()
        model = GKT(data_config["num_c"], **model_config,graph=graph,emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lpkt":
        qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
        if os.path.exists(qmatrix_path):
            q_matrix = np.load(qmatrix_path, allow_pickle=True)['matrix']
        else:
            q_matrix = generate_qmatrix(data_config)
        q_matrix = torch.tensor(q_matrix).float().to(device)
        model = LPKT(data_config["num_at"], data_config["num_it"], data_config["num_q"], data_config["num_c"], **model_config, q_matrix=q_matrix, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "skvmn":
        model = SKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)   
    elif model_name == "hawkes":
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions ans concepts! but the dataset has no both")
            return None
        model = HawkesKT(data_config["num_c"], data_config["num_q"], **model_config)
        model = model.double()
        # print("===before init weights"+"@"*100)
        # model.printparams()
        model.apply(model.init_weights)
        # print("===after init weights")
        # model.printparams()
        model = model.to(device)
    elif model_name == "iekt":
        model = IEKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)   
    elif model_name == "qdkt":
        # Recent addition: model config needs to include flag_load_embf lag_emb_freezed booleans
        model = QDKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "qikt":
        model = QIKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "atdkt":
        model = ATDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "bakt_time":
        model = BAKTTime(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt":
        model = simpleKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dimkt":
        model = DIMKT(data_config["num_q"],data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sparsekt":
        model = sparseKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rkt":
        model = RKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    ## YO Edition:
    elif model_name == "cskt":
        model = CausalSimpleKT(data_config["num_c"], model_config["kc_dim"], model_config["state_dim"], model_config["dropout"], question_embedding_path=data_config["emb_path"]).to(device)     
    elif model_name == "akt_que":
        model = AKTQue(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "iekt_que":
        model = IEKTQue(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "qikt_que":
        model = QIKTQue(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)   
    elif model_name == "simplekt_que":
        # In this special version, we treat questions and concepts the same.
        model = SimpleKTQue(data_config["num_q"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "sparsekt_que":
        model = SparseKTQue(data_config["num_q"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "kqn_que":
        model = KQNQue(data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "atkt_que":
        model = ATKTQue(data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=True,device=device).to(device)
    elif model_name == "deep_irt_que":
        model =  DeepIRTQue(data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "dkvmn_que":
        model =  DKVMNQue(data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "skvmn_que":
        model =  SKVMNQue(data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], use_onehot=False,device=device).to(device)
    elif model_name == "dkt_plus_que":
        model =  DKTPlusQue(data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "atdkt_que":
        print("MODEL CONFIG BELOW AGAIN")
        print(model_config)
        model = ATDKTQue(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "sakt_que":
        model = SAKTQue(data_config["num_q"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "saint_que":
        model = SAINTQue(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    return model
