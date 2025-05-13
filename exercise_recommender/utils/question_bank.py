import json
import torch
import torch.nn.functional as F
import numpy as np
import gc
device = "cuda" if torch.cuda.is_available() else "cpu"

import faiss

class QuestionBank():
    def __init__(self,
                que_emb_path="",
                que_emb_size:int = 768
                ):
        
        if que_emb_path == "":
            raise ValueError("Question Embedding Path must be provided to create a QuestionBank instance")
        
        self.que_emb_path = que_emb_path
        self.que_emb_size = que_emb_size
        self._init_que_embeddings()

    def _init_que_embeddings(self):
        emb_dir = {}
        with open(self.que_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        self.num_q = len(emb_dir)
        print(f"Loaded {self.num_q} questions into Question Bank")

        self.qid_to_index = {}
        self.index_to_qid = {}
        for idx, qid in enumerate(list(emb_dir.keys())):
            self.qid_to_index[qid] = idx
            self.index_to_qid[idx] = qid


        precomputed_embeddings_tensor = torch.empty(self.num_q, self.que_emb_size)
        for key in list(emb_dir.keys()):
            index = self.qid_to_index[key]
            precomputed_embeddings_tensor[index] = torch.tensor(emb_dir[key], dtype=torch.float)

        # For debug
        orig_norm = precomputed_embeddings_tensor[0].norm()
        print(f"The original norm of the embeddings provided is {orig_norm} .")

        # Normalize the lengths to 1, for convenience.
        norms = precomputed_embeddings_tensor.norm(p=2, dim=1, keepdim=True)
        precomputed_embeddings_tensor = precomputed_embeddings_tensor/norms

        # Now scale to expected size.
        precomputed_embeddings_tensor = precomputed_embeddings_tensor * np.sqrt(self.que_emb_size)

        # For debug
        cur_norm = precomputed_embeddings_tensor[0].norm()
        print(f"The norm of the embeddings are now scaled to {cur_norm} .")
        self.embeddings = precomputed_embeddings_tensor.to(device)
        self.embeddings_numpy = self.embeddings.to("cpu").numpy()
        faiss.normalize_L2(self.embeddings_numpy)
        self.index = faiss.IndexFlatIP(self.que_emb_size)
        self.index.add(self.embeddings_numpy)
        print(f"Embeddings for QuestionBank loaded into: {self.embeddings.device}")
    
    def _compute_cosine_similarity(self, tensor1, tensor2, batch_size=1024):
        tensor2 = tensor2.to(device)
        results = []
        for start in range(0, tensor1.size(0), batch_size):
            end = start+batch_size
            chunk = tensor1[start:end].to(device)
            sim = F.cosine_similarity(chunk, tensor2, dim=2).detach().cpu()
            results.append(sim)
        return torch.cat(results, dim=0)

    
    def find_closest_embeddings(self, query_embed, report_max_dist=False, return_qids=False):
        if not isinstance(query_embed, torch.Tensor):
            query_embed = torch.tensor(query_embed, dtype=torch.float)
        if query_embed.dim() == 1:
            query_embed = query_embed.unsqueeze(0)
        query_embed_numpy = query_embed.to("cpu").numpy()
        faiss.normalize_L2(query_embed_numpy)
        D, I = self.index.search(query_embed_numpy, 1)
        embs = []
        qids = []
        for i in range(query_embed.size(0)):
            index = I[i][0]
            qid = self.index_to_qid[index]
            qids.append(qid)
            emb = self.embeddings[self.qid_to_index[qid], :].detach().cpu()
            embs.append(emb)
        embs = torch.stack(embs, dim=0)
        if not return_qids:
            return embs
        else:
            return embs, qids
        """
        norms = query_embed.norm(p=2, dim=1, keepdim=True)
        query_embed = query_embed/norms
        query_embed = query_embed*np.sqrt(query_embed.shape[1])
        similarities = torch.mm(query_embed, self.embeddings.t())
        query_expanded = query_embed.unsqueeze(1).detach()
        corpus_expanded = self.embeddings.unsqueeze(0).detach()
        if on_cpu:
            similarities = F.cosine_similarity(query_expanded.cpu(), corpus_expanded.cpu(), dim=2)
        else:
            similarities = F.cosine_similarity(query_expanded, corpus_expanded, dim=2)
        similarities_cuda = self._compute_cosine_similarity(query_expanded, corpus_expanded, batch_size=256)
        similarities = similarities_cuda.detach().cpu()
        top_k_sims, top_k_indices = torch.topk(similarities, k=1, dim=1)
        best_sim_scores = []
        best_qids = []
        best_embs = []
        best_dists = []
        if report_max_dist:
            smallest_k_sims, smallest_k_indices = torch.topk(similarities, k=1, dim=1, largest=False)
            smallest_sim_scores = []
            smallest_qids = []
            smallest_embs = []
            worst_dists = []
        for i in range(query_embed.size(0)):
            row_sim = top_k_sims[i]
            row_ids = top_k_indices[i]
            qid = self.index_to_qid[row_ids.item()]
            emb = self.embeddings[self.qid_to_index[qid], :].detach().cpu()
            best_sim_scores.append(row_sim.item())
            best_qids.append(qid)
            best_embs.append(emb)
            best_dist = (query_embed[i].detach().cpu()-emb).norm(p=2).item()
            best_dists.append(best_dist)
            if report_max_dist:
                smallest_sim = smallest_k_sims[i]
                smallest_ids = smallest_k_indices[i]
                smallest_qid = self.index_to_qid[smallest_ids.item()]
                smallest_emb = self.embeddings[self.qid_to_index[smallest_qid], :].detach().cpu()
                smallest_sim_scores.append(smallest_sim.item())
                smallest_qids.append(smallest_qid)
                smallest_embs.append(smallest_emb)
                worst_dist = (query_embed[i].detach().cpu()-smallest_emb).norm(p=2).item()
                worst_dists.append(worst_dist)
        best_sim_scores = torch.tensor(best_sim_scores, dtype=torch.float)
        best_embs = torch.stack(best_embs, dim=0)
        best_dists = torch.tensor(best_dists, dtype=torch.float)
        if not report_max_dist:
            return best_sim_scores, best_qids, best_embs, best_dists
        else:
            smallest_sim_scores = torch.tensor(smallest_sim_scores, dtype=torch.float)
            smallest_embs = torch.stack(smallest_embs, dim=0)
            worst_dists = torch.tensor(worst_dists, dtype=torch.float)
            return best_sim_scores, best_qids, best_embs, best_dists, smallest_sim_scores, smallest_qids, smallest_embs, worst_dists
        """
