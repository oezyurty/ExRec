from pykt.datasets.que_data_loader import KTQueEmbeddingDataset
from torch.utils.data import DataLoader
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class CircularDataLoader:
    """
        Wraps a PyTorch DataLoader to cycle through it indefinitely
    """
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

class HistoryGenerator():
    def __init__(self, 
                dataset: KTQueEmbeddingDataset, 
                batch_size=1, 
                seed=42, 
                is_random:bool = True):
        if not dataset:
            raise("A KTQueEmbeddingDataset should be provided to use the HistoryGenerator")
        self.dataset = dataset
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.is_random = is_random
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=self.is_random)
        self.dataloader = CircularDataLoader(dataloader=dataloader)
        self.iter_dataloader = iter(self.dataloader)
    
    def get_question_response_pairs(self):
        # Returned qseqs shape: [batch_size, seq_len, emb_size]
        # Returned rseqs shape: [batch_size, seq_len]
        data = next(self.iter_dataloader)
        qseqs = torch.cat((data["qseqs"][:, 0:1], data["shft_qseqs"]),dim=1)
        rseqs = torch.cat((data["rseqs"][:, 0:1], data["shft_rseqs"]), dim=1)
        qid_seqs = torch.cat((data["qid_seqs"][:, 0:1], data["qid_shft_seqs"]), dim=1)
        qseqs = qseqs.to(device)
        rseqs = rseqs.to(device)
        qid_seqs = qid_seqs.to(device)
        return qseqs, rseqs, qid_seqs


