#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
# from torch.cuda import FloatTensor, LongTensor
from torch import FloatTensor, LongTensor
import numpy as np
import json
# TODO: Remove the timing of the functions
import time
from pykt.models.qdkt import EmbeddedQueDKT
import copy
import random
import csv

class KTQueDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).

    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds,concept_num,max_concepts, qtest=False, subset_rate=1.0):
        super(KTQueDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        if "questions" not in input_type or "concepts" not in input_type:
            raise("The input types must contain both questions and concepts")
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        processed_data = file_path + folds_str + "_qlevel.pkl"


        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")

            self.dori = self.__load_data__(sequence_path, folds)
            save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            self.dori = pd.read_pickle(processed_data)
        # Check if subsetting will be applied
        if subset_rate < 1.0:
            self.__subset_data__(subset_rate)
        
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            if key=='cseqs':
                seqs = self.dori[key][index][:-1,:]
                shft_seqs = self.dori[key][index][1:,:]
            else:
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        return dcur

    def get_skill_multi_hot(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.

        Returns: 
            (tuple): tuple containing

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()#[0:1000]
        interaction_num = 0
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills +[-1]*(self.max_concepts-len(skills))
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)


        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:,:-1] != pad_val) * (dori["rseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])
        return dori

    def __subset_data__(self, subset_rate):
        """
        Subset the original self.dori with certain rate. 
        """
        N = len(self.dori["rseqs"])

        # Calculate the number of rows to select
        num_rows_to_select = int(N * subset_rate)

        # Generate random indices
        random_indices = np.random.choice(N, num_rows_to_select, replace=False)

        # For all types of data in self.dori, we'll do the subsetting
        for k in self.dori.keys():
            if len(self.dori[k]) > 0:
                self.dori[k] = self.dori[k][random_indices]

class KTQueEmbeddingDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).

    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds,concept_num,max_concepts, qtest=False, subset_rate=1.0, emb_path=""):
        if emb_path == "":
            raise("Embedding path cannot be empty to use this dataset")
        super(KTQueEmbeddingDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        if "questions" not in input_type or "concepts" not in input_type:
            raise("The input types must contain both questions and concepts")
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        processed_data = file_path + folds_str + "_qlevel.pkl"


        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")

            self.dori = self.__load_data__(sequence_path, folds)
            save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            self.dori = pd.read_pickle(processed_data)
        # Check if subsetting will be applied
        if subset_rate < 1.0:
            self.__subset_data__(subset_rate)
        
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")
        self.emb_path = emb_path
        self._init_embeddings()

    def _init_embeddings(self):
        emb_dir = {}
        with open(self.emb_path, "r") as f:
            emb_dir = json.load(f)
        precomputed_embeddings_tensor = torch.tensor([emb_dir[str(i)] for i in range(len(emb_dir))], dtype=torch.float)

        num_q, emb_size = precomputed_embeddings_tensor.shape

        # For debug
        orig_norm = precomputed_embeddings_tensor[0].norm()
        #print(f"The original norm of the embeddings provided is {orig_norm} .")

        # Normalize the lengths to 1, for convenience.
        norms = precomputed_embeddings_tensor.norm(p=2, dim=1, keepdim=True)
        precomputed_embeddings_tensor = precomputed_embeddings_tensor/norms

        # Now scale to expected size.
        precomputed_embeddings_tensor = precomputed_embeddings_tensor * np.sqrt(emb_size)

        # For debug
        cur_norm = precomputed_embeddings_tensor[0].norm()
        #print(f"The norm of the embeddings are now scaled to {cur_norm} .")
        # number of questions is 7652
        self.que_emb = nn.Embedding.from_pretrained(precomputed_embeddings_tensor, freeze=True)


    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        #start_time = time.perf_counter()
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            if key=='cseqs':
                seqs = self.dori[key][index][:-1,:]
                shft_seqs = self.dori[key][index][1:,:]
            elif key=="qseqs":
                question_ids = self.dori[key][index][:-1] * mseqs
                dcur["qid_seqs"] = question_ids
                shft_question_ids = self.dori[key][index][1:] * mseqs
                dcur["qid_shft_seqs"] = shft_question_ids
                seqs = self.que_emb(question_ids)
                shft_seqs = self.que_emb(shft_question_ids)
            else:
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
            if key != "qseqs":
                dcur[key] = seqs
                dcur["shft_"+key] = shft_seqs
            else:
                dcur["qseqs"] = seqs
                dcur["shft_qseqs"] = shft_seqs
                dcur["target_embs"] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        #end_time = time.perf_counter()
        #elapsed_miliseconds = (end_time - start_time) * 1000
        #print(f"KTQueEmbeddingDataset elapsed miliseconds: {elapsed_miliseconds:.2f}")
        return dcur

    def get_skill_multi_hot(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.

        Returns: 
            (tuple): tuple containing

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()#[0:1000]
        interaction_num = 0
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills +[-1]*(self.max_concepts-len(skills))
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)


        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:,:-1] != pad_val) * (dori["rseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])
        return dori

    def __subset_data__(self, subset_rate):
        """
        Subset the original self.dori with certain rate. 
        """
        N = len(self.dori["rseqs"])

        # Calculate the number of rows to select
        num_rows_to_select = int(N * subset_rate)

        # Generate random indices
        random_indices = np.random.choice(N, num_rows_to_select, replace=False)

        # For all types of data in self.dori, we'll do the subsetting
        for k in self.dori.keys():
            if len(self.dori[k]) > 0:
                self.dori[k] = self.dori[k][random_indices]

class KTCalibrationDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).

    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds,concept_num,max_concepts, qtest=False, subset_rate=1.0, emb_path="", frozen_model_path="", kc_to_questions_path="", kc_emb_path="", num_q_pred=1):
        if emb_path == "":
            raise("Embedding path cannot be empty to use this dataset")
        if frozen_model_path == "":
            raise("The frozen_model_path cannot be None to use this dataset")
        if kc_to_questions_path == "":
            raise("Knowledge Concept to Questions mapping should be given to use this dataset")
        if kc_emb_path == "":
            raise("Knowledge Concept embeddings paths cannot be empty to use this dataset")

        super(KTCalibrationDataset, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        self.num_q_pred = num_q_pred
        if "questions" not in input_type or "concepts" not in input_type:
            raise("The input types must contain both questions and concepts")
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        processed_data = file_path + folds_str + "_qlevel.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")

            self.dori = self.__load_data__(sequence_path, folds)
            save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            self.dori = pd.read_pickle(processed_data)
        # Check if subsetting will be applied
        if subset_rate < 1.0:
            self.__subset_data__(subset_rate)
        
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")
        self.emb_path = emb_path
        self._init_que_embeddings()

        self.frozen_model_path = frozen_model_path
        self._init_model()

        self.kc_to_questions_path = kc_to_questions_path
        self.kc_to_que_map = self._get_kc_to_que_map()

        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()
    
    def _init_model(self):
        self.model = EmbeddedQueDKT()
        net = torch.load(self.frozen_model_path, map_location="cpu")
        self.model.load_state_dict(net)
        print(f"For Calibration Dataset, frozen model loaded to: {self.device}")
        #print(f"Loaded frozen model is: {model}")
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.required_grad = False
    
    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        self.kc_name_to_id = dict()
        for i,key in enumerate(list(emb_dir.keys())):
            self.kc_name_to_id[key] = i

        precomputed_kc_emb_tensors = torch.tensor([emb_dir[key] for key in emb_dir], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _get_kc_to_que_map(self):
        kc2qmap = {}
        with open(self.kc_to_questions_path, "r") as f:
            kc2qmap = json.load(f)
        return kc2qmap
    
    def _get_questions_for_kcs(self, kc_keys, num_q):
        all_question_embeddings = []
        for kc_key in kc_keys:
            if kc_key not in self.kc_to_que_map:
                raise(f"Given knowledge concept key: {kc_key} does not exist int the map")
            question_ids = self.kc_to_que_map[kc_key]
            sampled_question_ids = random.choices(question_ids, k=num_q)
            sampled_question_ids = torch.tensor(sampled_question_ids, dtype=int)
            embeddings = self.que_emb(sampled_question_ids)
            all_question_embeddings.append(embeddings)
        stacked_questions = torch.stack(all_question_embeddings, dim=1)
        return stacked_questions
    
    def _sample_kcs(self, n):
        kc_keys = list(self.kc_to_que_map.keys())
        sampled_kcs = random.sample(kc_keys, k=n)
        return sampled_kcs

    def _init_que_embeddings(self):
        emb_dir = {}
        with open(self.emb_path, "r") as f:
            emb_dir = json.load(f)
        precomputed_embeddings_tensor = torch.tensor([emb_dir[str(i)] for i in range(len(emb_dir))], dtype=torch.float)

        num_q, emb_size = precomputed_embeddings_tensor.shape

        # For debug
        orig_norm = precomputed_embeddings_tensor[0].norm()
        #print(f"The original norm of the embeddings provided is {orig_norm} .")

        # Normalize the lengths to 1, for convenience.
        norms = precomputed_embeddings_tensor.norm(p=2, dim=1, keepdim=True)
        precomputed_embeddings_tensor = precomputed_embeddings_tensor/norms

        # Now scale to expected size.
        precomputed_embeddings_tensor = precomputed_embeddings_tensor * np.sqrt(emb_size)

        # For debug
        cur_norm = precomputed_embeddings_tensor[0].norm()
        #print(f"The norm of the embeddings are now scaled to {cur_norm} .")
        self.que_emb = nn.Embedding.from_pretrained(precomputed_embeddings_tensor, freeze=True)


    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        #start_time = time.perf_counter()
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            if key=='cseqs':
                seqs = self.dori[key][index][:-1,:]
                shft_seqs = self.dori[key][index][1:,:]
            elif key=="qseqs":
                question_ids = self.dori[key][index][:-1] * mseqs
                shft_question_ids = self.dori[key][index][1:] * mseqs
                seqs = self.que_emb(question_ids)
                shft_seqs = self.que_emb(shft_question_ids)
            else:
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
            if key != "qseqs":
                dcur[key] = seqs
                dcur["shft_"+key] = shft_seqs
            else:
                dcur[key] = seqs
                dcur["shft_"+key] = shft_seqs
                dcur["que_target_embs"] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        #end_time = time.perf_counter()
        #elapsed_miliseconds = (end_time - start_time) * 1000
        #print(f"KTQueEmbeddingDataset elapsed miliseconds: {elapsed_miliseconds:.2f}")

        # Random KC and soft_target_y values
        seq_len = len(dcur["qseqs"])
        sampled_kcs = self._sample_kcs(seq_len)
        dcur["kc_seqs"] = [self.kc_name_to_id[kc_name] for kc_name in sampled_kcs]
        dcur["kc_seqs"] = torch.tensor(dcur["kc_seqs"])
        dcur["kc_embs"] = self.kc_emb(dcur["kc_seqs"])
        with torch.no_grad():
            q_embs = torch.cat((dcur["qseqs"][0:1, :], dcur["shft_qseqs"]), dim=0).to(self.device)
            extended_q_embs = q_embs.unsqueeze(0).repeat(self.num_q_pred, 1, 1).to(self.device)
            target_embs = self._get_questions_for_kcs(sampled_kcs, self.num_q_pred).to(self.device)
            rseqs = torch.cat((dcur["rseqs"][0:1], dcur["shft_rseqs"]), dim=0).to(self.device)
            output = self.model.model(extended_q_embs, rseqs, target_embs)
            probs = output["y"].mean(dim=0)
            dcur["kc_preds"] = probs.detach().cpu()
        return dcur

    def get_skill_multi_hot(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.

        Returns: 
            (tuple): tuple containing

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()#[0:1000]
        interaction_num = 0
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills +[-1]*(self.max_concepts-len(skills))
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)


        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:,:-1] != pad_val) * (dori["rseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])
        return dori

    def __subset_data__(self, subset_rate):
        """
        Subset the original self.dori with certain rate. 
        """
        N = len(self.dori["rseqs"])

        # Calculate the number of rows to select
        num_rows_to_select = int(N * subset_rate)

        # Generate random indices
        random_indices = np.random.choice(N, num_rows_to_select, replace=False)

        # For all types of data in self.dori, we'll do the subsetting
        for k in self.dori.keys():
            if len(self.dori[k]) > 0:
                self.dori[k] = self.dori[k][random_indices]

class KTClusterCalibrationDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).

    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds,concept_num,max_concepts, qtest=False, subset_rate=1.0, emb_path="", frozen_model_path="", kc_to_questions_path="", kc_emb_path="",  clusters_to_kcs_path="", clusters_to_qids_path="", num_q_pred=1):
        if emb_path == "":
            raise("Embedding path cannot be empty to use this dataset")
        if frozen_model_path == "":
            raise("The frozen_model_path cannot be None to use this dataset")
        if kc_to_questions_path == "":
            raise("Knowledge Concept to Questions mapping should be given to use this dataset")
        if kc_emb_path == "":
            raise("Knowledge Concept embeddings paths cannot be empty to use this dataset")
        if clusters_to_kcs_path == "":
            raise("Clusters to KCs path cannot be empty to use this dataset")
        if clusters_to_qids_path == "":
            raise("Clusters to Question IDs must be precomputed to use this dataset")
        super(KTClusterCalibrationDataset, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        self.num_q_pred = num_q_pred
        if "questions" not in input_type or "concepts" not in input_type:
            raise("The input types must contain both questions and concepts")
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        processed_data = file_path + folds_str + "_qlevel.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")

            self.dori = self.__load_data__(sequence_path, folds)
            save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            self.dori = pd.read_pickle(processed_data)
        # Check if subsetting will be applied
        if subset_rate < 1.0:
            self.__subset_data__(subset_rate)
        
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")
        self.emb_path = emb_path
        self._init_que_embeddings()

        self.frozen_model_path = frozen_model_path
        self._init_model()

        self.kc_to_questions_path = kc_to_questions_path
        self.kc_to_que_map = self._get_kc_to_que_map()

        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()

        self.clusters_to_kcs_path = clusters_to_kcs_path
        self._init_clusters_to_kcs_map()

        self.clusters_to_qids_path = clusters_to_qids_path
        self._init_clusters_to_qids_map()
    
    def _init_clusters_to_kcs_map(self):
        # Clusters to KC name mapping
        clusters_to_kcs_dir = {}
        with open(self.clusters_to_kcs_path, "r") as f:
            clusters_to_kcs_dir = json.load(f)
        self.clusters_to_kcs_map = clusters_to_kcs_dir
    
    def _init_clusters_to_qids_map(self):
        clusters_to_qids_dir = {}
        with open(self.clusters_to_qids_path, "r") as f:
            clusters_to_qids_dir = json.load(f)
        self.clusters_to_qids_map = clusters_to_qids_dir
        
    
    def _init_model(self):
        self.model = EmbeddedQueDKT()
        net = torch.load(self.frozen_model_path, map_location="cpu")
        self.model.load_state_dict(net)
        print(f"For Cluster Calibration Dataset, frozen model loaded to: {self.device}")
        #print(f"Loaded frozen model is: {model}")
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.required_grad = False
    
    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        self.kc_name_to_id = dict()
        for i,key in enumerate(list(emb_dir.keys())):
            self.kc_name_to_id[key] = i

        precomputed_kc_emb_tensors = torch.tensor([emb_dir[key] for key in emb_dir], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _get_kc_to_que_map(self):
        kc2qmap = {}
        with open(self.kc_to_questions_path, "r") as f:
            kc2qmap = json.load(f)
        return kc2qmap
    
    def _get_questions_for_kcs(self, kc_keys, num_q):
        all_question_embeddings = []
        for kc_key in kc_keys:
            if kc_key not in self.kc_to_que_map:
                raise(f"Given knowledge concept key: {kc_key} does not exist int the map")
            question_ids = self.kc_to_que_map[kc_key]
            sampled_question_ids = random.choices(question_ids, k=num_q)
            sampled_question_ids = torch.tensor(sampled_question_ids, dtype=int)
            embeddings = self.que_emb(sampled_question_ids)
            all_question_embeddings.append(embeddings)
        stacked_questions = torch.stack(all_question_embeddings, dim=1)
        return stacked_questions
    
    def _get_questions_for_clusters(self, clusters, num_q):
        all_question_embeddings = []
        for cluster_key in clusters:
            if cluster_key not in self.clusters_to_qids_map:
                raise(f"Given cluster key: {cluster_key} does not exist in the map")
            question_ids = self.clusters_to_qids_map[cluster_key]
            sampled_question_ids = random.choices(question_ids, k=num_q)
            sampled_question_ids = torch.tensor(sampled_question_ids, dtype=int)
            embeddings = self.que_emb(sampled_question_ids)
            all_question_embeddings.append(embeddings)
        stacked_questions = torch.stack(all_question_embeddings, dim=1)
        return stacked_questions
    
    def _get_kcs_for_clusters(self, clusters):
        all_kc_keys = []
        for cluster_key in clusters:
            sampled_kc = self._sample_one_kc_from_cluster(cluster_key)
            all_kc_keys.append(sampled_kc)
        all_kc_ids = [self.kc_name_to_id[kc_key] for kc_key in all_kc_keys]
        return all_kc_ids
    
    def _sample_clusters(self, n):
        cluster_keys = list(self.clusters_to_qids_map.keys())
        sampled_clusters = random.sample(cluster_keys, k=n)
        return sampled_clusters
    
    def _sample_one_kc_from_cluster(self, cluster_key):
        kc_keys = self.clusters_to_kcs_map[cluster_key]
        sampled_kc = random.sample(kc_keys, k=1)[0]
        return sampled_kc

    def _init_que_embeddings(self):
        emb_dir = {}
        with open(self.emb_path, "r") as f:
            emb_dir = json.load(f)
        precomputed_embeddings_tensor = torch.tensor([emb_dir[str(i)] for i in range(len(emb_dir))], dtype=torch.float)

        num_q, emb_size = precomputed_embeddings_tensor.shape

        # For debug
        orig_norm = precomputed_embeddings_tensor[0].norm()
        #print(f"The original norm of the embeddings provided is {orig_norm} .")

        # Normalize the lengths to 1, for convenience.
        norms = precomputed_embeddings_tensor.norm(p=2, dim=1, keepdim=True)
        precomputed_embeddings_tensor = precomputed_embeddings_tensor/norms

        # Now scale to expected size.
        precomputed_embeddings_tensor = precomputed_embeddings_tensor * np.sqrt(emb_size)

        # For debug
        cur_norm = precomputed_embeddings_tensor[0].norm()
        #print(f"The norm of the embeddings are now scaled to {cur_norm} .")
        self.que_emb = nn.Embedding.from_pretrained(precomputed_embeddings_tensor, freeze=True)


    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        #start_time = time.perf_counter()
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            if key=='cseqs':
                seqs = self.dori[key][index][:-1,:]
                shft_seqs = self.dori[key][index][1:,:]
            elif key=="qseqs":
                question_ids = self.dori[key][index][:-1] * mseqs
                shft_question_ids = self.dori[key][index][1:] * mseqs
                seqs = self.que_emb(question_ids)
                shft_seqs = self.que_emb(shft_question_ids)
            else:
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
            if key != "qseqs":
                dcur[key] = seqs
                dcur["shft_"+key] = shft_seqs
            else:
                dcur[key] = seqs
                dcur["shft_"+key] = shft_seqs
                dcur["que_target_embs"] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        #end_time = time.perf_counter()
        #elapsed_miliseconds = (end_time - start_time) * 1000
        #print(f"KTQueEmbeddingDataset elapsed miliseconds: {elapsed_miliseconds:.2f}")

        # Random KC and soft_target_y values
        seq_len = len(dcur["qseqs"])
        sampled_clusters = self._sample_clusters(seq_len)
        dcur["kc_seqs"] = self._get_kcs_for_clusters(sampled_clusters)
        dcur["kc_seqs"] = torch.tensor(dcur["kc_seqs"])
        dcur["kc_embs"] = self.kc_emb(dcur["kc_seqs"])
        with torch.no_grad():
            q_embs = torch.cat((dcur["qseqs"][0:1, :], dcur["shft_qseqs"]), dim=0).to(self.device)
            extended_q_embs = q_embs.unsqueeze(0).repeat(self.num_q_pred, 1, 1).to(self.device)
            target_embs = self._get_questions_for_clusters(sampled_clusters, self.num_q_pred).to(self.device)
            rseqs = torch.cat((dcur["rseqs"][0:1], dcur["shft_rseqs"]), dim=0).to(self.device)
            output = self.model.model(extended_q_embs, rseqs, target_embs)
            probs = output["y"].mean(dim=0)
            dcur["kc_preds"] = probs.detach().cpu()
        return dcur

    def get_skill_multi_hot(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.

        Returns: 
            (tuple): tuple containing

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()#[0:1000]
        interaction_num = 0
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills +[-1]*(self.max_concepts-len(skills))
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)


        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:,:-1] != pad_val) * (dori["rseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])
        return dori

    def __subset_data__(self, subset_rate):
        """
        Subset the original self.dori with certain rate. 
        """
        N = len(self.dori["rseqs"])

        # Calculate the number of rows to select
        num_rows_to_select = int(N * subset_rate)

        # Generate random indices
        random_indices = np.random.choice(N, num_rows_to_select, replace=False)

        # For all types of data in self.dori, we'll do the subsetting
        for k in self.dori.keys():
            if len(self.dori[k]) > 0:
                self.dori[k] = self.dori[k][random_indices]