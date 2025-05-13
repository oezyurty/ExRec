from pykt.datasets.que_data_loader import KTQueEmbeddingDataset
from exercise_recommender.utils.history_generator import HistoryGenerator
import pandas as pd

def get_dataset():
    dataset = KTQueEmbeddingDataset("../data/XES3G5M/question_level/train_valid_sequences_quelevel.csv", 
    input_type=["questions","concepts"], folds={1,2,3,4}, concept_num=865, max_concepts=7,
    emb_path="../../data/XES3G5M_embeddings/qid2content_sol_avg_emb.json")
    return dataset

def get_history_generator(batch_size, folds, seed=42, is_random=False):
    dataset = KTQueEmbeddingDataset("../data/XES3G5M/question_level/filtered_train_valid_sequences_quelevel.csv", 
    input_type=["questions","concepts"], folds=folds, concept_num=865, max_concepts=7,
    emb_path="../data/XES3G5M_embeddings/qid2content_sol_avg_emb.json")
    history_generator = HistoryGenerator(dataset, batch_size=batch_size, seed=seed, is_random=is_random)
    return history_generator