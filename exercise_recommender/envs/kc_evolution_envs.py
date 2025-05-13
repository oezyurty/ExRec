import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import sys
from exercise_recommender.wrappers.calibrationqdkt_wrapper import CalibrationQDKTWrapper
from exercise_recommender.utils.data import get_history_generator
import json
from collections import defaultdict
import random
import os
import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"

class WorstKCEvolutionContinuousEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                dataset_folds = {1},
                max_steps: int=10,
                init_seq_size:int=200,
                last_n_steps: int=200,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                question_bank = None,
                reward_scale = None,
                reward_func: str = "step_by_step",
                enforce_corpus: bool = True,
                seed=42,
                log_wandb:bool = False,
                log_path:str = "",
                cluster_batch_size: int=256,
                wandb_prefix: str="",
                log_impr_path=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")

        if wandb_prefix == "":
            raise ValueError("Please provide a valid Wandb prefix") 
        
        super(WorstKCEvolutionContinuousEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.cluster_batch_size = cluster_batch_size
        self.log_wandb = log_wandb
        self.log_path = log_path
        self.enforce_corpus = enforce_corpus
        if self.enforce_corpus:
            if not question_bank:
                raise ValueError("To enforce corpus, a QuestionBank instance must be provided")
        self.question_bank = question_bank
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()
        self._assign_kc_to_clusters()

        self.batch_size = batch_size
        history_generation_batch_size = 64 if self.batch_size > 64 else self.batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)

        self.pretrained_model_path = pretrained_model_path
        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # Episode step count is the number of steps until reaching max steps
        # Total step count is the ongoing number of steps until changing students
        self.episode_step_count = 0
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.step_count = 0
        self.question_bank = question_bank
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.question_embed_size,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._internal_step_batch = -1 if self.batch_size <= 64 else 64

        self._is_reset = False

        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids

    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states(seq_size=self.init_seq_size)
        self._initial_student_states = self._curr_student_states
        self._initial_lstm = self.kt_model._initial_lstm
    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)

    def _assign_cluster_embeddings(self):
        all_preds = self._get_all_kc_preds() # shape: self.batch_size, num_questions
        self.all_kc_preds[:, :, self.step_count] = all_preds
        lowest_indices = torch.argmin(all_preds, dim=1) # shape: self.batch_size
        self._student_kc_map = self._cluster_kc_map[lowest_indices]
        self.chosen_kc_indices[:, self.step_count] = lowest_indices

        lowest_preds, _ = all_preds.min(dim=1)
        self.kc_preds[:, self.step_count] = lowest_preds
        self.kc_evolution_matrix[:, self.step_count, self.step_count] = lowest_preds
    
    def _get_preds_on_kc_embs(self, kc_embs):
        with torch.no_grad():
            return self.kt_model.predict_in_rl(kc_embs)
        
    def _get_starting_preds(self):
        self._starting_preds = self._get_preds_on_kc_embs(self._student_kc_map)
        self.kc_evolution_matrix[:, 0, 0] = self._starting_preds
    
    def _assign_kc_to_clusters(self):
        self._cluster_kc_map = torch.zeros((self.num_clusters, self._kc_emb_dim), dtype=torch.float)
        for cluster_key in list(self.cluster_to_kc_map.keys()):
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            sampled_kc = random.sample(kc_keys, k=1)[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0)
            self._cluster_kc_map[int(cluster_key)] = _kc_emb
        self._cluster_kc_map = self._cluster_kc_map.to(device)
    
    def _init_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        self._old_levels = mean_preds_per_student
    
    def get_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        return mean_preds_per_student
    
    def _get_all_kc_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)
        
        all_preds = torch.cat(all_preds, dim=1)
        return all_preds

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.chosen_kc_indices = torch.full((self.batch_size, self.max_steps), -1.0, dtype=torch.int)
            self.chosen_question_indices = []
            self.kc_preds = torch.full((self.batch_size, self.max_steps), -1.0, dtype=torch.float)
            self.kc_evolution_matrix = torch.full((self.batch_size, self.max_steps, self.max_steps+1), -1.0, dtype=torch.float)
            self.all_kc_preds = torch.full((self.batch_size, self._cluster_kc_map.shape[0], self.max_steps+1), -1.0, dtype=torch.float)
            self._init_kt_model()
            self._assign_cluster_embeddings()
            self._get_starting_preds()
            if self.log_path != "":
                with open(os.path.join(self.log_path, "test_cluster_env.txt"), "a+") as f:
                    print("Setting up Students", file=f, flush=True)
                    print(self._qid_seqs, file=f, flush=True)
                    print("="*10, file=f, flush=True)
            self._is_reset = True
        else:
            print("Environment ended!")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            self._assign_cluster_embeddings()
            if self.log_path != "":
                with open(os.path.join(self.log_path, "test_cluster_env.txt"), "a+") as f:
                    print("Reverting back to initial states", file=f, flush=True)
                    print(self._qid_seqs, file=f, flush=True)
                    print("="*10, file=f, flush=True)
            """
        obs = torch.cat((self._curr_student_states, self._student_kc_map), dim=1).to(device)
        info = {}
        return obs.detach().cpu().numpy(), info

    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }
    
    def _get_preds_on_chosen_kcs(self):
        all_preds = self._get_all_kc_preds()
        self.all_kc_preds[:, :, self.step_count] = all_preds
        with torch.no_grad():
            for i in range(self.step_count):
                for j in range(self.step_count+1):
                    indices = self.chosen_kc_indices[:, i]
                    student_idx = torch.arange(self.batch_size)
                    _preds = self.all_kc_preds[student_idx, indices, j]
                    self.kc_evolution_matrix[:, i, j] = _preds
        
    def log_improvements(self):
        old_preds = torch.empty((self.batch_size, self.chosen_kc_indices.size(1)), dtype=torch.float)
        with torch.no_grad():
            for index in range(self.chosen_kc_indices.size(1)):
                indices = self.chosen_kc_indices[:, index]
                kc_embs = self._cluster_kc_map[indices]
                preds = self.kt_model.pretrained_model.model.model.prediction_layer(self._initial_lstm, kc_embs)
                old_preds[:, index] = preds
        new_preds = torch.empty((self.batch_size, self.chosen_kc_indices.size(1)), dtype=torch.float)
        with torch.no_grad():
            for index in range(self.chosen_kc_indices.size(1)):
                indices = self.chosen_kc_indices[:, index]
                kc_embs = self._cluster_kc_map[indices]
                new_preds[:, index] = self.kt_model.predict_in_rl(kc_embs)
        improvement = new_preds - old_preds
        mean_improvement = improvement.mean(dim=1)
        max_upper_bound = 1 - old_preds.mean(dim=1).mean()
        if self.log_wandb:
            wandb.log({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement.mean(),
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(mean_improvement),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(mean_improvement),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound
            })
        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            print({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement.mean().item(),
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(mean_improvement).item(),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(mean_improvement).item(),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        if action.shape[1] != self.question_embed_size:
            raise ValueError(f"Got second dimension {action.shape[1]}, expecting: {self.question_embed_size}")
        if self.enforce_corpus:
            if self._internal_step_batch == -1:
                embs, qids = self.question_bank.find_closest_embeddings(action, return_qids=True)
            else:
                embs = []
                qids = []
                total_processed = 0
                while total_processed < action.shape[0]:
                    end_index = total_processed + self._internal_step_batch
                    batch_act = action[total_processed:end_index]
                    _embs, _qids = self.question_bank.find_closest_embeddings(batch_act, return_qids=True)
                    qids.append(_qids)
                    embs.append(_embs)
                    total_processed += self._internal_step_batch
                embs = torch.cat(embs, dim=0)
        new_que = action # Shape becomes [batch_size, que_emb_size]
        self.chosen_question_indices.append(qids)
        del action
        if self.enforce_corpus:
            new_que = embs.to(device)
        # KC expertise level before
        if self.reward_func == "step_by_step":
            old_levels = self._get_preds_on_kc_embs(self._student_kc_map)
        with torch.no_grad():
            probs = self.kt_model.predict_in_rl(new_que)
        # The following line turns the prediction to 0 or 1.
        student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        new_levels = self._get_preds_on_kc_embs(self._student_kc_map)
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        if not terminated:
            self._assign_cluster_embeddings()
        else:
            self._get_preds_on_chosen_kcs()
            self.log_improvements()
        obs = torch.cat((obs, self._student_kc_map), dim=1)
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        if self.log_wandb:
            wandb.log({
                "Test Mean Sim Score": mean_sim_score,
                "Test Mean Reward": reward.mean(),
                "Test Mean Dist": dists.mean(),
                }, commit=True)
        truncated = False
        info = {}
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation

class WorstKCEvolutionDiscreteEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                dataset_folds = {0},
                max_steps: int=10,
                init_seq_size:int=200,
                last_n_steps: int=200,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                question_bank = None,
                reward_scale = None,
                reward_func: str = "step_by_step",
                enforce_corpus: bool = True,
                seed=42,
                log_wandb:bool = False,
                log_path:str = "",
                cluster_batch_size: int=256,
                wandb_prefix: str="",
                log_impr_path=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")

        if wandb_prefix == "":
            raise ValueError("Please provide a valid Wandb prefix") 
        
        super(WorstKCEvolutionDiscreteEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.cluster_batch_size = cluster_batch_size
        self.log_wandb = log_wandb
        self.log_path = log_path
        self.enforce_corpus = enforce_corpus
        if self.enforce_corpus:
            if not question_bank:
                raise ValueError("To enforce corpus, a QuestionBank instance must be provided")
        self.question_bank = question_bank
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()
        self._assign_kc_to_clusters()

        self.batch_size = batch_size
        history_generation_batch_size = 64 if self.batch_size > 64 else self.batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)

        self.pretrained_model_path = pretrained_model_path
        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # Episode step count is the number of steps until reaching max steps
        # Total step count is the ongoing number of steps until changing students
        self.episode_step_count = 0
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.step_count = 0
        self.question_bank = question_bank
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Discrete(
            n=self.question_bank.num_q, seed=seed, start=0
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._internal_step_batch = -1 if self.batch_size <= 64 else 64

        self._is_reset = False

        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids

    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states(seq_size=self.init_seq_size)
        self._initial_student_states = self._curr_student_states
        self._initial_lstm = self.kt_model._initial_lstm
    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)

    def _assign_cluster_embeddings(self):
        all_preds = self._get_all_kc_preds() # shape: self.batch_size, num_questions
        self.all_kc_preds[:, :, self.step_count] = all_preds
        lowest_indices = torch.argmin(all_preds, dim=1) # shape: self.batch_size
        self._student_kc_map = self._cluster_kc_map[lowest_indices]
        self.chosen_kc_indices[:, self.step_count] = lowest_indices

        lowest_preds, _ = all_preds.min(dim=1)
        self.kc_preds[:, self.step_count] = lowest_preds
        self.kc_evolution_matrix[:, self.step_count, self.step_count] = lowest_preds
    
    def _get_preds_on_kc_embs(self, kc_embs):
        with torch.no_grad():
            return self.kt_model.predict_in_rl(kc_embs)
        
    def _get_starting_preds(self):
        self._starting_preds = self._get_preds_on_kc_embs(self._student_kc_map)
        self.kc_evolution_matrix[:, 0, 0] = self._starting_preds
    
    def _assign_kc_to_clusters(self):
        self._cluster_kc_map = torch.zeros((self.num_clusters, self._kc_emb_dim), dtype=torch.float)
        for cluster_key in list(self.cluster_to_kc_map.keys()):
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            sampled_kc = random.sample(kc_keys, k=1)[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0)
            self._cluster_kc_map[int(cluster_key)] = _kc_emb
        self._cluster_kc_map = self._cluster_kc_map.to(device)
    
    def _init_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        self._old_levels = mean_preds_per_student
    
    def get_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        return mean_preds_per_student
    
    def _get_all_kc_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)
        
        all_preds = torch.cat(all_preds, dim=1)
        return all_preds

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.chosen_kc_indices = torch.full((self.batch_size, self.max_steps), -1.0, dtype=torch.int)
            self.chosen_question_indices = []
            self.kc_preds = torch.full((self.batch_size, self.max_steps), -1.0, dtype=torch.float)
            self.kc_evolution_matrix = torch.full((self.batch_size, self.max_steps, self.max_steps+1), -1.0, dtype=torch.float)
            self.all_kc_preds = torch.full((self.batch_size, self._cluster_kc_map.shape[0], self.max_steps+1), -1.0, dtype=torch.float)
            self._init_kt_model()
            self._assign_cluster_embeddings()
            self._get_starting_preds()
            if self.log_path != "":
                with open(os.path.join(self.log_path, "test_cluster_env.txt"), "a+") as f:
                    print("Setting up Students", file=f, flush=True)
                    print(self._qid_seqs, file=f, flush=True)
                    print("="*10, file=f, flush=True)
            self._is_reset = True
        else:
            print("Environment ended!")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            self._assign_cluster_embeddings()
            if self.log_path != "":
                with open(os.path.join(self.log_path, "test_cluster_env.txt"), "a+") as f:
                    print("Reverting back to initial states", file=f, flush=True)
                    print(self._qid_seqs, file=f, flush=True)
                    print("="*10, file=f, flush=True)
            """
        obs = torch.cat((self._curr_student_states, self._student_kc_map), dim=1).to(device)
        info = {}
        return obs.detach().cpu().numpy(), info

    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }
    
    def _get_preds_on_chosen_kcs(self):
        all_preds = self._get_all_kc_preds()
        self.all_kc_preds[:, :, self.step_count] = all_preds
        with torch.no_grad():
            for i in range(self.step_count):
                for j in range(self.step_count+1):
                    indices = self.chosen_kc_indices[:, i]
                    student_idx = torch.arange(self.batch_size)
                    _preds = self.all_kc_preds[student_idx, indices, j]
                    self.kc_evolution_matrix[:, i, j] = _preds

    def log_improvements(self):
        old_preds = torch.empty((self.batch_size, self.chosen_kc_indices.size(1)), dtype=torch.float)
        with torch.no_grad():
            for index in range(self.chosen_kc_indices.size(1)):
                indices = self.chosen_kc_indices[:, index]
                kc_embs = self._cluster_kc_map[indices]
                preds = self.kt_model.pretrained_model.model.model.prediction_layer(self._initial_lstm, kc_embs)
                old_preds[:, index] = preds
        new_preds = torch.empty((self.batch_size, self.chosen_kc_indices.size(1)), dtype=torch.float)
        with torch.no_grad():
            for index in range(self.chosen_kc_indices.size(1)):
                indices = self.chosen_kc_indices[:, index]
                kc_embs = self._cluster_kc_map[indices]
                new_preds[:, index] = self.kt_model.predict_in_rl(kc_embs)
        improvement = new_preds - old_preds
        mean_improvement = improvement.mean(dim=1)
        max_upper_bound = 1 - old_preds.mean(dim=1).mean()
        if self.log_wandb:
            wandb.log({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement.mean(),
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(mean_improvement),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(mean_improvement),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound
            })
        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            print({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement.mean().item(),
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(mean_improvement).item(),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(mean_improvement).item(),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        new_que = self.question_bank.embeddings[action, :]
        qids = []
        for q_idx in action:
            qids.append(self.question_bank.index_to_qid[q_idx.item()])
        #new_que = action # Shape becomes [batch_size, que_emb_size]
        self.chosen_question_indices.append(qids)
        # KC expertise level before
        old_levels = self._get_preds_on_kc_embs(self._student_kc_map)
        with torch.no_grad():
            probs = self.kt_model.predict_in_rl(new_que)
        # The following line turns the prediction to 0 or 1.
        student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        new_levels = self._get_preds_on_kc_embs(self._student_kc_map)
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        if not terminated:
            self._assign_cluster_embeddings()
        else:
            self._get_preds_on_chosen_kcs()
            self.log_improvements()
        obs = torch.cat((obs, self._student_kc_map), dim=1)
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        if self.log_wandb:
            wandb.log({
                "Test Mean Sim Score": mean_sim_score,
                "Test Mean Reward": reward.mean(),
                "Test Mean Dist": dists.mean(),
                }, commit=True)
        truncated = False
        info = {}
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation

    
class WorstKCEvolutionGroundTruthEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                dataset_folds = {0},
                max_steps: int=10,
                max_steps_until_student_change: int=100,
                init_seq_size:int=200,
                last_n_steps: int=200,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                reward_scale = None,
                reward_func: str = "step_by_step",
                seed=42,
                log_path:str = "",
                cluster_batch_size: int=256,
                use_ground_truth_response: bool = False,
                wandb_prefix="",
                log_impr_path:str = ""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")
        
        super(WorstKCEvolutionGroundTruthEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.use_ground_truth_response = use_ground_truth_response
        self.log_path = log_path
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()
        self._assign_kc_to_clusters()

        self.batch_size = batch_size
        history_generation_batch_size = 64 if self.batch_size > 64 else self.batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)
        self._init_history()

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        self.step_count = 0
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.question_embed_size,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._is_reset = False
        self.cluster_batch_size = cluster_batch_size

        self.chosen_kc_indices = torch.empty((self.batch_size, self.max_steps), dtype=torch.int)

        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
    
    def _init_history(self):
        all_qseqs = []
        all_rseqs = []
        all_qid_seqs = []
        total_batch = 0
        while total_batch < self.batch_size:
            qseqs, rseqs, qid_seqs = self.history.get_question_response_pairs()
            batch = qseqs.shape[0]
            total_batch += batch
            all_qseqs.append(qseqs)
            all_rseqs.append(rseqs)
            all_qid_seqs.append(qid_seqs)

        self._qseqs = torch.cat(all_qseqs, dim=0).to(device)
        self._rseqs = torch.cat(all_rseqs, dim=0).to(device)
        self._all_qid_seqs = torch.cat(all_qid_seqs, dim=0).to(device)
    
    def _init_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        self._old_levels = mean_preds_per_student
    
    def get_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        return mean_preds_per_student
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids

    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        mini_batch_size = 64 if self.batch_size >= 64 else self.batch_size
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states_with_data(
                                            mini_batch_size=mini_batch_size, 
                                            seq_size=self.init_seq_size,
                                            qseqs=self._qseqs,
                                            rseqs=self._rseqs,
                                            qid_seqs=self._all_qid_seqs)

        self._initial_student_states = self._curr_student_states
        self._initial_lstm = self.kt_model._initial_lstm
    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_kc_to_clusters(self):
        self._cluster_kc_map = torch.zeros((self.num_clusters, self._kc_emb_dim), dtype=torch.float)
        for cluster_key in list(self.cluster_to_kc_map.keys()):
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            sampled_kc = random.sample(kc_keys, k=1)[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0)
            self._cluster_kc_map[int(cluster_key)] = _kc_emb
        self._cluster_kc_map = self._cluster_kc_map.to(device)
    
    def _assign_cluster_embeddings(self):
        all_preds = self._get_all_kc_preds()
        self.all_kc_preds[:, :, self.step_count] = all_preds
        lowest_indices = torch.argmin(all_preds, dim=1)
        self._student_kc_map = self._cluster_kc_map[lowest_indices]
        self.chosen_kc_indices[:, self.step_count] = lowest_indices

        lowest_preds, _ = all_preds.min(dim=1)
        self.kc_preds[:, self.step_count] = lowest_preds
        self.kc_evolution_matrix[:, self.step_count, self.step_count] = lowest_preds
    
    def _get_preds_on_kc_embs(self, kc_embs):
        with torch.no_grad():
            return self.kt_model.predict_in_rl(kc_embs)
        
    def _get_starting_preds(self):
        self._starting_preds = self._get_preds_on_kc_embs(self._student_kc_map)
        self.kc_evolution_matrix[:, 0, 0] = self._starting_preds
    
    def _get_all_kc_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)
        
        all_preds = torch.cat(all_preds, dim=1)
        return all_preds

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.chosen_kc_indices = torch.empty((self.batch_size, self.max_steps), dtype=torch.int)
            self.chosen_question_indices = []
            self.kc_preds = torch.full((self.batch_size, self.max_steps), -1.0, dtype=torch.float)
            self.kc_evolution_matrix = torch.full((self.batch_size, self.max_steps, self.max_steps+1), -1.0, dtype=torch.float)
            self.all_kc_preds = torch.full((self.batch_size, self._cluster_kc_map.shape[0], self.max_steps+1), -1.0, dtype=torch.float)
            self._init_kt_model()
            self._assign_cluster_embeddings()
            self._get_starting_preds()
            self._is_reset = True
        else:
            print("Environment ended!")
            #self._curr_student_states = self._initial_student_states
            #self.kt_model.reset_in_rl()
            #self._assign_cluster_embeddings()
        obs = torch.cat((self._curr_student_states, self._student_kc_map), dim=1).to(device)
        info = {}
        return obs.detach().cpu().numpy(), info
    
    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }
    
    def log_improvements(self):
        old_preds = torch.empty((self.batch_size, self.chosen_kc_indices.size(1)), dtype=torch.float)
        with torch.no_grad():
            for index in range(self.chosen_kc_indices.size(1)):
                indices = self.chosen_kc_indices[:, index]
                kc_embs = self._cluster_kc_map[indices]
                preds = self.kt_model.pretrained_model.model.model.prediction_layer(self._initial_lstm, kc_embs)
                old_preds[:, index] = preds
        new_preds = torch.empty((self.batch_size, self.chosen_kc_indices.size(1)), dtype=torch.float)
        with torch.no_grad():
            for index in range(self.chosen_kc_indices.size(1)):
                indices = self.chosen_kc_indices[:, index]
                kc_embs = self._cluster_kc_map[indices]
                new_preds[:, index] = self.kt_model.predict_in_rl(kc_embs)
        
        improvement = new_preds - old_preds
        mean_improvement = improvement.mean(dim=1)
        max_upper_bound = 1 - old_preds.mean(dim=1).mean()

        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            if self.use_ground_truth_response:
                print("Using Ground Truth Responses Results:", file=f, flush=True)
            else:
                print("NOT Using Ground Truth Responses Results:", file=f, flush=True)

            print({
                "GT Env Episode End - Mean Improvement": mean_improvement.mean().item(),
                "GT Env Episode End - Max Improvement": torch.max(mean_improvement).item(),
                "GT Env Episode End - Min Improvement": torch.min(mean_improvement).item(),
                "GT Env Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)

    def _get_preds_on_chosen_kcs(self):
        all_preds = self._get_all_kc_preds()
        self.all_kc_preds[:, :, self.step_count] = all_preds
        with torch.no_grad():
            for i in range(self.step_count):
                for j in range(self.step_count+1):
                    indices = self.chosen_kc_indices[:, i]
                    student_idx = torch.arange(self.batch_size)
                    _preds = self.all_kc_preds[student_idx, indices, j]
                    self.kc_evolution_matrix[:, i, j] = _preds

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation

        Expected action shape is (self.batch_size, self.question_bank.num_q)
        """
        info = {}
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")

        question_idx = self.init_seq_size + self.step_count
        new_que = self._qseqs[:, question_idx, :]
        old_levels = self._get_preds_on_kc_embs(self._student_kc_map)
        np_old_levels = old_levels.clone().detach().cpu().numpy()
        info["y_kc_before"] = np_old_levels

        with torch.no_grad():
            probs = self.kt_model.predict_in_rl(new_que)
        
        info["y_que"] = probs.clone().detach().cpu().numpy()
        cell_state = self.kt_model.last_c.clone()
        cell_state = cell_state.squeeze(0)
        info["cell_state"] = cell_state.clone().detach().cpu().numpy()
        # The following line turns the prediction to 0 or 1.
        if self.use_ground_truth_response:
            student_responses = self._rseqs[:, question_idx]
        else:
            student_responses = (torch.rand_like(probs) < probs).float()

        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        new_levels = self._get_preds_on_kc_embs(self._student_kc_map)
        info["y_kc_after"] = new_levels.clone().detach().cpu().numpy()
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        if not terminated:
            self._assign_cluster_embeddings()
        else:
            self._get_preds_on_chosen_kcs()
            self.log_improvements()
        obs = torch.cat((obs, self._student_kc_map), dim=1)
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        truncated = False
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation

class AllKCEvolutionContinuousEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                dataset_folds = {0},
                max_steps: int=10,
                init_seq_size:int=200,
                last_n_steps: int=200,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                question_bank = None,
                reward_scale = None,
                reward_func: str = "step_by_step",
                enforce_corpus: bool = True,
                seed=42,
                log_wandb:bool = False,
                log_path:str = "",
                cluster_batch_size: int=256,
                wandb_prefix: str="",
                log_impr_path: str=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")
        
        super(AllKCEvolutionContinuousEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.cluster_batch_size = cluster_batch_size
        self.log_wandb = log_wandb
        self.log_path = log_path
        self.enforce_corpus = enforce_corpus
        if self.enforce_corpus:
            if not question_bank:
                raise ValueError("To enforce corpus, a QuestionBank instance must be provided")
        self.question_bank = question_bank
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()
        self._assign_kc_to_clusters()

        self.batch_size = batch_size
        history_generation_batch_size = 64 if self.batch_size > 64 else self.batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # Episode step count is the number of steps until reaching max steps
        # Total step count is the ongoing number of steps until changing students
        self.episode_step_count = 0
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.step_count = 0
        self.question_bank = question_bank
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.question_embed_size,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._internal_step_batch = -1 if self.batch_size <= 64 else 64

        self._is_reset = False

        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids

    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states(seq_size=self.init_seq_size)
        self._initial_student_states = self._curr_student_states
    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_cluster_embeddings(self):
        self._student_kc_map = torch.zeros((self.batch_size, self._kc_emb_dim), dtype=torch.float)
        for sid in range(self.batch_size):
            cluster_key = self.max_seen_clusters[sid]
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            # TODO: Change it back to randomness
            sampled_kc = random.sample(kc_keys, k=1)[0]
            #sampled_kc = kc_keys[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0) # Returns 1 by 768 -> squeezed to 768
            self._student_kc_map[sid] = _kc_emb
        self._student_kc_map = self._student_kc_map.to(device)
    
    def _assign_kc_to_clusters(self):
        self._cluster_kc_map = torch.zeros((self.num_clusters, self._kc_emb_dim), dtype=torch.float)
        for cluster_key in list(self.cluster_to_kc_map.keys()):
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            sampled_kc = random.sample(kc_keys, k=1)[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0)
            self._cluster_kc_map[int(cluster_key)] = _kc_emb
        self._cluster_kc_map = self._cluster_kc_map.to(device)
    
    def _init_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        self._old_levels = mean_preds_per_student
        self.all_kc_preds[:, self.step_count] = mean_preds_per_student
    
    def get_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        return mean_preds_per_student

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.all_kc_preds = torch.full((self.batch_size, self.max_steps+1), -1.0, dtype=torch.float)
            self._init_kt_model()
            self._init_preds()
            if self.log_path != "":
                with open(os.path.join(self.log_path, "test_cluster_env.txt"), "a+") as f:
                    print("Setting up Students", file=f, flush=True)
                    print(self._qid_seqs, file=f, flush=True)
                    print("="*10, file=f, flush=True)
            self._is_reset = True
        else:
            print("Environment ended")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            if self.log_path != "":
                with open(os.path.join(self.log_path, "test_cluster_env.txt"), "a+") as f:
                    print("Reverting back to initial states", file=f, flush=True)
                    print(self._qid_seqs, file=f, flush=True)
                    print("="*10, file=f, flush=True)
            """
        obs = self._curr_student_states
        info = {}
        return obs.detach().cpu().numpy(), info
    
    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }
    
    def log_improvements(self):
        old_preds = self.all_kc_preds[:, 0]
        new_preds = self.all_kc_preds[:, self.max_steps]
        improvement = new_preds - old_preds
        mean_improvement = improvement.mean()
        max_upper_bound = 1 - old_preds.mean()

        if self.log_wandb:
            wandb.log({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement,
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound
            })
        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            print({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement.item(),
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement).item(),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement).item(),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        if action.shape[1] != self.question_embed_size:
            raise ValueError(f"Got second dimension {action.shape[1]}, expecting: {self.question_embed_size}")
        if self.enforce_corpus:
            if self._internal_step_batch == -1:
                embs = self.question_bank.find_closest_embeddings(action)
            else:
                embs = []
                total_processed = 0
                while total_processed < action.shape[0]:
                    end_index = total_processed + self._internal_step_batch
                    batch_act = action[total_processed:end_index]
                    _embs = self.question_bank.find_closest_embeddings(batch_act)
                    embs.append(_embs)
                    total_processed += self._internal_step_batch
                embs = torch.cat(embs, dim=0)
        new_que = action # Shape becomes [batch_size, que_emb_size]
        del action
        if self.enforce_corpus:
            new_que = embs.to(device)
        # KC expertise level before
        if self.reward_func == "step_by_step":
            old_levels = self.get_preds()
        with torch.no_grad():
            probs = self.kt_model.predict_in_rl(new_que)
        # The following line turns the prediction to 0 or 1.
        student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        new_levels = self.get_preds()
        self.step_count += 1

        # Record the predictions in the all_kc_preds matrix
        self.all_kc_preds[:, self.step_count] = new_levels

        terminated = self.step_count >= self.max_steps
        if terminated:
            self.log_improvements()
            
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        if self.log_wandb:
            wandb.log({
                "Test Mean Sim Score": mean_sim_score,
                "Test Mean Reward": reward.mean(),
                "Test Mean Dist": dists.mean(),
                }, commit=True)
        truncated = False
        info = {}
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation


class AllKCEvolutionDiscreteEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                dataset_folds = {1},
                max_steps: int=10,
                max_steps_until_student_change: int=100,
                init_seq_size:int=200,
                last_n_steps: int=200,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                reward_scale = None,
                reward_func: str = "step_by_step",
                seed=42,
                log_path:str = "",
                log_wandb:bool=False,
                question_bank=None,
                cluster_batch_size: int=256,
                wandb_prefix:str="",
                log_impr_path:str=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")
        
        super(AllKCEvolutionDiscreteEnv, self).__init__()
        self.cluster_batch_size = cluster_batch_size
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        self.log_path = log_path
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()
        self._assign_kc_to_clusters()

        self.batch_size = batch_size
        history_generation_batch_size = 64 if self.batch_size > 64 else self.batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # Episode step count is the number of steps until reaching max steps
        # Total step count is the ongoing number of steps until changing students
        self.episode_step_count = 0
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.max_steps_until_student_change = max_steps_until_student_change
        self._is_reset = False
        self._done = False
        self.question_bank = question_bank
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size

        self.question_bank = question_bank
        
        self.action_space = spaces.Discrete(
            n=self.question_bank.num_q, seed=seed, start=0
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
        self.log_wandb = log_wandb

    def _init_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        self._old_levels = mean_preds_per_student
        self.all_kc_preds[:, 0] = mean_preds_per_student
    
    def get_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        return mean_preds_per_student
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids

    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states(seq_size=self.init_seq_size)
        self._initial_student_states = self._curr_student_states
    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_kc_to_clusters(self):
        self._cluster_kc_map = torch.zeros((self.num_clusters, self._kc_emb_dim), dtype=torch.float)
        for cluster_key in list(self.cluster_to_kc_map.keys()):
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            sampled_kc = random.sample(kc_keys, k=1)[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0)
            self._cluster_kc_map[int(cluster_key)] = _kc_emb
        self._cluster_kc_map = self._cluster_kc_map.to(device)
    
    def _assign_cluster_embeddings(self):
        self._student_kc_map = torch.zeros((self.batch_size, self._kc_emb_dim), dtype=torch.float)
        for sid in range(self.batch_size):
            cluster_key = self.max_seen_clusters[sid]
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            sampled_kc = random.sample(kc_keys, k=1)[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0) # Returns 1 by 768 -> squeezed to 768
            self._student_kc_map[sid] = _kc_emb
        self._student_kc_map = self._student_kc_map.to(device)

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.all_kc_preds = torch.full((self.batch_size, self.max_steps+1), -1.0, dtype=torch.float)
            self._init_kt_model()
            self._init_preds()
            self._is_reset = True
        else:
            print("Environment ended!")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            """
        
        obs = self._curr_student_states
        info = {}
        return obs.detach().cpu().numpy(), info
    
    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }
    
    def log_improvements(self):
        old_preds = self.all_kc_preds[:, 0]
        new_preds = self.all_kc_preds[:, self.max_steps]
        improvement = new_preds - old_preds
        mean_improvement = improvement.mean()
        max_upper_bound = 1 - old_preds.mean()

        if self.log_wandb:
            wandb.log({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement,
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound
            })
        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            print({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement.item(),
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement).item(),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement).item(),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation

        Expected action shape is (self.batch_size, self.question_bank.num_q)
        """
        info = {}
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        new_que = self.question_bank.embeddings[action, :]
        old_levels = self.get_preds()
        np_old_levels = old_levels.clone().detach().cpu().numpy()
        info["y_kc_before"] = np_old_levels
        with torch.no_grad():
            probs = self.kt_model.predict_in_rl(new_que)
        info["y_que"] = probs.clone().detach().cpu().numpy()
        cell_state = self.kt_model.last_c.clone()
        cell_state = cell_state.squeeze(0)
        info["cell_state"] = cell_state.clone().detach().cpu().numpy()
        # The following line turns the prediction to 0 or 1.
        student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        new_levels = self.get_preds()
        info["y_kc_after"] = new_levels.clone().detach().cpu().numpy()
        self.step_count += 1

        # Record the predictions in the all_kc_preds matrix
        self.all_kc_preds[:, self.step_count] = new_levels


        terminated = self.step_count >= self.max_steps
        if terminated:
            self.log_improvements()
        
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        truncated = False
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation


class AllKCEvolutionGroundTruthEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                history_generation_batch_size=64,
                dataset_folds = {0},
                max_steps: int=10,
                init_seq_size:int=100,
                last_n_steps: int=100,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                reward_scale = None,
                reward_func: str = "step_by_step",
                seed=42,
                log_wandb:bool = False,
                log_path:str = "",
                use_ground_truth_response: bool = False,
                cluster_batch_size=256,
                wandb_prefix:str="",
                log_impr_path:str=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")
        
        super(AllKCEvolutionGroundTruthEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.use_ground_truth_response = use_ground_truth_response
        self.log_wandb = log_wandb
        self.log_path = log_path
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()
        self._assign_kc_to_clusters()

        self.batch_size = batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)
        self._init_history()

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.step_count = 0
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.question_embed_size,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._internal_step_batch = -1 if self.batch_size <= 128 else 128

        self._is_reset = False
        self.cluster_batch_size = cluster_batch_size
        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
    
    def _init_history(self):

        all_qseqs = []
        all_rseqs = []
        all_qid_seqs = []
        total_batch = 0
        while total_batch < self.batch_size:
            qseqs, rseqs, qid_seqs = self.history.get_question_response_pairs()
            batch = qseqs.shape[0]
            total_batch += batch
            all_qseqs.append(qseqs)
            all_rseqs.append(rseqs)
            all_qid_seqs.append(qid_seqs)

        self._qseqs = torch.cat(all_qseqs, dim=0).to(device)
        self._rseqs = torch.cat(all_rseqs, dim=0).to(device)
        self._all_qid_seqs = torch.cat(all_qid_seqs, dim=0).to(device)
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids

    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        mini_batch_size = 64 if self.batch_size >= 64 else self.batch_size
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states_with_data(
                                            mini_batch_size=mini_batch_size, 
                                            seq_size=self.init_seq_size,
                                            qseqs=self._qseqs,
                                            rseqs=self._rseqs,
                                            qid_seqs=self._all_qid_seqs)

        self._initial_student_states = self._curr_student_states

    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_kc_to_clusters(self):
        self._cluster_kc_map = torch.zeros((self.num_clusters, self._kc_emb_dim), dtype=torch.float)
        for cluster_key in list(self.cluster_to_kc_map.keys()):
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            sampled_kc = random.sample(kc_keys, k=1)[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0)
            self._cluster_kc_map[int(cluster_key)] = _kc_emb
        self._cluster_kc_map = self._cluster_kc_map.to(device)
    
    def _init_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        self._old_levels = mean_preds_per_student
        self.all_kc_preds[:, 0] = mean_preds_per_student
    
    def get_preds(self):
        num_clusters = self._cluster_kc_map.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, num_clusters, self.cluster_batch_size):
                end = min(start + self.cluster_batch_size, num_clusters)
                current_batch_size = end - start
                kc_emb_repeated = self._cluster_kc_map[start:end].unsqueeze(0).expand(self.batch_size, -1, -1)
                kc_emb_batched = kc_emb_repeated.reshape(-1, 768)

                last_lstm_out_repeated = self.kt_model.last_lstm_out.unsqueeze(1).expand(-1, end-start, -1)
                last_lstm_out_batched = last_lstm_out_repeated.reshape(-1, 300)

                preds_batched = self.kt_model.pretrained_model.model.model.prediction_layer(last_lstm_out_batched, kc_emb_batched)

                preds_batched = preds_batched.reshape(self.batch_size, end-start)

                all_preds.append(preds_batched)

        all_preds = torch.cat(all_preds, dim=1)

        mean_preds_per_student = all_preds.mean(dim=1)

        return mean_preds_per_student

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.all_kc_preds = torch.full((self.batch_size, self.max_steps+1), -1.0, dtype=torch.float)
            self._init_kt_model()
            self._init_preds()
            self._is_reset = True
        else:
            print("Environment ended!")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            """
        obs = self._curr_student_states
        info = {}
        return obs.detach().cpu().numpy(), info
    
    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }
    
    def log_improvements(self):
        old_preds = self.all_kc_preds[:, 0]
        new_preds = self.all_kc_preds[:, self.max_steps]
        improvement = new_preds - old_preds
        mean_improvement = improvement.mean()
        max_upper_bound = 1 - old_preds.mean()

        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            if self.use_ground_truth_response:
                print("Using Ground Truth Responses Results:", file=f, flush=True)
            else:
                print("NOT Using Ground Truth Responses Results:", file=f, flush=True)

            print({
                "GT Env Episode End - Mean Improvement": mean_improvement.mean().item(),
                "GT Env Episode End - Max Improvement": torch.max(mean_improvement).item(),
                "GT Env Episode End - Min Improvement": torch.min(mean_improvement).item(),
                "GT Env Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        if action.shape[1] != self.question_embed_size:
            raise ValueError(f"Got second dimension {action.shape[1]}, expecting: {self.question_embed_size}")
        
        # The index of the next question from the dataset is self.init_seq_size + self.step_count
        # EX: init_seq_size = 100, step_count is 0 -> 100th question is the next question
        question_idx = self.init_seq_size + self.step_count
        new_que = self._qseqs[:, question_idx, :]
        # KC expertise level before
        if self.reward_func == "step_by_step":
            old_levels = self.get_preds()
        # The following line turns the prediction to 0 or 1.
        if self.use_ground_truth_response:
            student_responses = self._rseqs[:, question_idx]
        else:
            with torch.no_grad():
                probs = self.kt_model.predict_in_rl(new_que)
            student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        new_levels = self.get_preds()
        self.step_count += 1

        # Record the predictions in the all_kc_preds matrix
        self.all_kc_preds[:, self.step_count] = new_levels

        terminated = self.step_count >= self.max_steps
        if terminated:
            self.log_improvements()
        
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        if self.log_wandb:
            wandb.log({
                "Test Mean Reward": reward.mean()
                }, commit=True)
        if self.log_path != "":
            with open(os.path.join(self.log_path, "ground_truth_env.txt"), "a+") as f:
                print(reward, file=f, flush=True)
        truncated = False
        info = {}
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation


class BroaderKCEvolutionContinuousEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                dataset_folds = {0},
                max_steps: int=10,
                init_seq_size:int=200,
                last_n_steps: int=200,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                question_bank = None,
                reward_scale = None,
                reward_func: str = "step_by_step",
                enforce_corpus: bool = True,
                seed=42,
                log_wandb:bool = False,
                log_path:str = "",
                wandb_prefix:str="",
                log_impr_path:str=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")
        
        super(BroaderKCEvolutionContinuousEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        self.log_wandb = log_wandb
        self.log_path = log_path
        self.enforce_corpus = enforce_corpus
        if self.enforce_corpus:
            if not question_bank:
                raise ValueError("To enforce corpus, a QuestionBank instance must be provided")
        self.question_bank = question_bank
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()
        self._init_transition_matrix()

        self.batch_size = batch_size
        history_generation_batch_size = 64 if self.batch_size > 64 else self.batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # Episode step count is the number of steps until reaching max steps
        # Total step count is the ongoing number of steps until changing students
        self.episode_step_count = 0
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.step_count = 0
        self.question_bank = question_bank
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.question_embed_size,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._internal_step_batch = -1 if self.batch_size <= 64 else 64

        self._is_reset = False

        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids

    def _init_transition_matrix(self):
        transition_matrix = torch.load("../../train_test/utils/cluster_transition_matrix.pt")
        transition_matrix = transition_matrix.numpy()

        epsilon = 1e-6 #added for preventing division by 0
        transition_prob = transition_matrix / (transition_matrix.sum(axis=1)[:,None] + epsilon)

        row_sums = transition_matrix.sum(axis=1)

        col_sums = transition_matrix.sum(axis=0)
        col_probs = col_sums / col_sums.sum()

        for i in range(len(transition_prob)):
            if row_sums[i] == 0:
                transition_prob[i,:] = col_probs
            else:
                transition_prob[i,:] /= transition_prob[i].sum()
        self.transition_prob = transition_prob
        self.transition_matrix = transition_matrix
    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states(seq_size=self.init_seq_size)
        self._initial_student_states = self._curr_student_states
        self._initial_lstm = self.kt_model._initial_lstm
    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_cluster_embeddings(self):
        self._student_kc_map = torch.zeros((self.batch_size, self._kc_emb_dim), dtype=torch.float)
        for sid in range(self.batch_size):
            cluster_key = self._student_target_ids[sid]
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            sampled_kc = random.sample(kc_keys, k=1)[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0) # Returns 1 by 768 -> squeezed to 768
            self._student_kc_map[sid] = _kc_emb
        self._student_kc_map = self._student_kc_map.to(device)
    
    def _find_target_kc(self):
        self._student_target_ids = np.zeros((self.batch_size), dtype=int)
        for i, max_seen_cluster_id in enumerate(self.max_seen_clusters):
            target_cluster_id = np.random.choice(np.arange(len(self.transition_matrix)), 1, p=self.transition_prob[max_seen_cluster_id])[0]
            #print(f"Max Seen Cluster ID: {max_seen_cluster_id} -> Sampled: {target_cluster_id}", file=f, flush=True)
            self._student_target_ids[i] = target_cluster_id
    
    def _init_preds(self):
        old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self._old_levels = old_levels
        self.all_kc_preds[:, 0] = old_levels

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.all_kc_preds = torch.full((self.batch_size, self.max_steps+1), -1.0, dtype=torch.float, requires_grad=False)
            self._init_kt_model()
            self._init_cluster_statistics()
            self._find_target_kc()
            self._assign_cluster_embeddings()
            self._init_preds()
            self._is_reset = True
        else:
            print("Environment ended!")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            """
        obs = torch.cat((self._curr_student_states, self._student_kc_map), dim=1).to(device)
        info = {}
        return obs.detach().cpu().numpy(), info
    
    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }
    
    def log_improvements(self):
        old_preds = self.all_kc_preds[:, 0]
        new_preds = self.all_kc_preds[:, self.max_steps]
        improvement = new_preds - old_preds

        mean_improvement = improvement.mean()
        max_upper_bound = 1 - old_preds.mean()

        if self.log_wandb:
            wandb.log({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement,
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound
            })
        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            print({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement.item(),
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement).item(),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement).item(),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        if action.shape[1] != self.question_embed_size:
            raise ValueError(f"Got second dimension {action.shape[1]}, expecting: {self.question_embed_size}")
        if self.enforce_corpus:
            if self._internal_step_batch == -1:
                embs = self.question_bank.find_closest_embeddings(action)
            else:
                embs = []
                total_processed = 0
                while total_processed < action.shape[0]:
                    end_index = total_processed + self._internal_step_batch
                    batch_act = action[total_processed:end_index]
                    _embs = self.question_bank.find_closest_embeddings(batch_act)
                    embs.append(_embs)
                    total_processed += self._internal_step_batch
                embs = torch.cat(embs, dim=0)
        new_que = action # Shape becomes [batch_size, que_emb_size]
        del action
        if self.enforce_corpus:
            new_que = embs.to(device)
        # KC expertise level before
        if self.reward_func == "step_by_step":
            old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        with torch.no_grad():
            probs = self.kt_model.predict_in_rl(new_que)
        # The following line turns the prediction to 0 or 1.
        student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        obs = torch.cat((obs, self._student_kc_map), dim=1)
        new_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self.step_count += 1

        # Record the predictions in the all_kc_preds matrix
        self.all_kc_preds[:, self.step_count] = new_levels

        terminated = self.step_count >= self.max_steps

        if terminated:
            self.log_improvements()
        
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        if self.log_wandb:
            wandb.log({
                "Test Mean Sim Score": mean_sim_score,
                "Test Mean Reward": reward.mean(),
                "Test Mean Dist": dists.mean(),
                }, commit=True)
        truncated = False
        info = {}
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation


class BroaderKCEvolutionDiscreteEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                dataset_folds = {0},
                max_steps: int=10,
                init_seq_size:int=200,
                last_n_steps: int=200,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                question_bank = None,
                reward_scale = None,
                reward_func: str = "step_by_step",
                seed=42,
                log_wandb:bool = False,
                log_path:str = "",
                wandb_prefix:str ="",
                log_impr_path:str=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")
        
        super(BroaderKCEvolutionDiscreteEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        self.log_wandb = log_wandb
        self.log_path = log_path
        if not question_bank:
            raise ValueError("Question Bank cannot be None to use this environment")
        self.question_bank = question_bank
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()
        self._init_transition_matrix()

        self.batch_size = batch_size
        history_generation_batch_size = 64 if self.batch_size > 64 else self.batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # Episode step count is the number of steps until reaching max steps
        # Total step count is the ongoing number of steps until changing students
        self.episode_step_count = 0
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.step_count = 0
        self.question_bank = question_bank
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Discrete(
            n=self.question_bank.num_q, seed=seed, start=0
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._internal_step_batch = -1 if self.batch_size <= 64 else 64

        self._is_reset = False
        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids
    
    def _init_transition_matrix(self):
        transition_matrix = torch.load("../../train_test/utils/cluster_transition_matrix.pt")
        transition_matrix = transition_matrix.numpy()

        epsilon = 1e-6 #added for preventing division by 0
        transition_prob = transition_matrix / (transition_matrix.sum(axis=1)[:,None] + epsilon)

        row_sums = transition_matrix.sum(axis=1)

        col_sums = transition_matrix.sum(axis=0)
        col_probs = col_sums / col_sums.sum()

        for i in range(len(transition_prob)):
            if row_sums[i] == 0:
                transition_prob[i,:] = col_probs
            else:
                transition_prob[i,:] /= transition_prob[i].sum()
        self.transition_prob = transition_prob
        self.transition_matrix = transition_matrix

    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states(seq_size=self.init_seq_size)
        self._initial_student_states = self._curr_student_states
    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)
    
    def _find_target_kc(self):
        self._student_target_ids = np.zeros((self.batch_size), dtype=int)
        for i, max_seen_cluster_id in enumerate(self.max_seen_clusters):
            target_cluster_id = np.random.choice(np.arange(len(self.transition_matrix)), 1, p=self.transition_prob[max_seen_cluster_id])[0]
            #print(f"Max Seen Cluster ID: {max_seen_cluster_id} -> Sampled: {target_cluster_id}", file=f, flush=True)
            self._student_target_ids[i] = target_cluster_id

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_cluster_embeddings(self):
        self._student_kc_map = torch.zeros((self.batch_size, self._kc_emb_dim), dtype=torch.float)
        for sid in range(self.batch_size):
            cluster_key = self._student_target_ids[sid]
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            sampled_kc = random.sample(kc_keys, k=1)[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0) # Returns 1 by 768 -> squeezed to 768
            self._student_kc_map[sid] = _kc_emb
        self._student_kc_map = self._student_kc_map.to(device)
    
    def _init_preds(self):
        old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self._old_levels = old_levels
        self.all_kc_preds[:, 0] = old_levels

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.all_kc_preds = torch.full((self.batch_size, self.max_steps+1), -1.0, dtype=torch.float, requires_grad=False)
            self._init_kt_model()
            self._init_cluster_statistics()
            self._find_target_kc()
            self._assign_cluster_embeddings()
            self._init_preds()
            self._is_reset = True
        else:
            print("Environment ended!")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            """
        obs = torch.cat((self._curr_student_states, self._student_kc_map), dim=1).to(device)
        info = {}
        return obs.detach().cpu().numpy(), info
    
    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }
    
    def log_improvements(self):
        old_preds = self.all_kc_preds[:, 0]
        new_preds = self.all_kc_preds[:, self.max_steps]
        improvement = new_preds - old_preds

        mean_improvement = improvement.mean()
        max_upper_bound = 1 - old_preds.mean()

        if self.log_wandb:
            wandb.log({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement,
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound
            })
        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            print({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement.item(),
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement).item(),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement).item(),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        
        new_que = self.question_bank.embeddings[action, :]
        # KC expertise level before
        if self.reward_func == "step_by_step":
            old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        with torch.no_grad():
            probs = self.kt_model.predict_in_rl(new_que)
        # The following line turns the prediction to 0 or 1.
        student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        obs = torch.cat((obs, self._student_kc_map), dim=1)
        new_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self.step_count += 1

        # Record the predictions in the all_kc_preds matrix
        self.all_kc_preds[:, self.step_count] = new_levels

        terminated = self.step_count >= self.max_steps
        if terminated:
            self.log_improvements()
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        if self.log_wandb:
            wandb.log({
                "Test Mean Sim Score": mean_sim_score,
                "Test Mean Reward": reward.mean(),
                "Test Mean Dist": dists.mean(),
                }, commit=True)
        truncated = False
        info = {}
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation


class BroaderKCEvolutionGroundTruthEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                dataset_folds = {0},
                max_steps: int=10,
                init_seq_size:int=200,
                last_n_steps: int=200,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                question_bank = None,
                reward_scale = None,
                reward_func: str = "step_by_step",
                enforce_corpus: bool = True,
                seed=42,
                log_wandb:bool = False,
                log_path:str = "",
                use_ground_truth_response:bool = False,
                wandb_prefix:str="",
                log_impr_path:str=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")
        
        super(BroaderKCEvolutionGroundTruthEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        self.log_wandb = log_wandb
        self.log_path = log_path
        self.enforce_corpus = enforce_corpus
        if self.enforce_corpus:
            if not question_bank:
                raise ValueError("To enforce corpus, a QuestionBank instance must be provided")
        self.question_bank = question_bank
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()
        self._init_transition_matrix()

        self.batch_size = batch_size
        history_generation_batch_size = 64 if self.batch_size > 64 else self.batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)
        self._init_history()

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # Episode step count is the number of steps until reaching max steps
        # Total step count is the ongoing number of steps until changing students
        self.episode_step_count = 0
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.step_count = 0
        self.question_bank = question_bank
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.question_embed_size,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._internal_step_batch = -1 if self.batch_size <= 64 else 64

        self._is_reset = False

        self.use_ground_truth_response = use_ground_truth_response
        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path

    def _init_history(self):
        all_qseqs = []
        all_rseqs = []
        all_qid_seqs = []
        total_batch = 0
        while total_batch < self.batch_size:
            qseqs, rseqs, qid_seqs = self.history.get_question_response_pairs()
            batch = qseqs.shape[0]
            total_batch += batch
            all_qseqs.append(qseqs)
            all_rseqs.append(rseqs)
            all_qid_seqs.append(qid_seqs)

        self._qseqs = torch.cat(all_qseqs, dim=0).to(device)
        self._rseqs = torch.cat(all_rseqs, dim=0).to(device)
        self._all_qid_seqs = torch.cat(all_qid_seqs, dim=0).to(device)
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids

    def _init_transition_matrix(self):
        transition_matrix = torch.load("../../train_test/utils/cluster_transition_matrix.pt")
        transition_matrix = transition_matrix.numpy()

        epsilon = 1e-6 #added for preventing division by 0
        transition_prob = transition_matrix / (transition_matrix.sum(axis=1)[:,None] + epsilon)

        row_sums = transition_matrix.sum(axis=1)

        col_sums = transition_matrix.sum(axis=0)
        col_probs = col_sums / col_sums.sum()

        for i in range(len(transition_prob)):
            if row_sums[i] == 0:
                transition_prob[i,:] = col_probs
            else:
                transition_prob[i,:] /= transition_prob[i].sum()
        self.transition_prob = transition_prob
        self.transition_matrix = transition_matrix
    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        mini_batch_size = 64 if self.batch_size >= 64 else self.batch_size
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states_with_data(
                                            mini_batch_size=mini_batch_size, 
                                            seq_size=self.init_seq_size,
                                            qseqs=self._qseqs,
                                            rseqs=self._rseqs,
                                            qid_seqs=self._all_qid_seqs)
        self._initial_student_states = self._curr_student_states
    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_cluster_embeddings(self):
        self._student_kc_map = torch.zeros((self.batch_size, self._kc_emb_dim), dtype=torch.float)
        for sid in range(self.batch_size):
            cluster_key = self._student_target_ids[sid]
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            sampled_kc = random.sample(kc_keys, k=1)[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0) # Returns 1 by 768 -> squeezed to 768
            self._student_kc_map[sid] = _kc_emb
        self._student_kc_map = self._student_kc_map.to(device)
    
    def _find_target_kc(self):
        self._student_target_ids = np.zeros((self.batch_size), dtype=int)
        for i, max_seen_cluster_id in enumerate(self.max_seen_clusters):
            target_cluster_id = np.random.choice(np.arange(len(self.transition_matrix)), 1, p=self.transition_prob[max_seen_cluster_id])[0]
            #print(f"Max Seen Cluster ID: {max_seen_cluster_id} -> Sampled: {target_cluster_id}", file=f, flush=True)
            self._student_target_ids[i] = target_cluster_id
    
    def _init_preds(self):
        old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self._old_levels = old_levels
        self.all_kc_preds[:, 0] = old_levels

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.all_kc_preds = torch.full((self.batch_size, self.max_steps+1), -1.0, dtype=torch.float, requires_grad=False)
            self._init_kt_model()
            self._init_cluster_statistics()
            self._find_target_kc()
            self._assign_cluster_embeddings()
            self._init_preds()
            self._is_reset = True
        else:
            print("Environment ended!")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            """
        obs = torch.cat((self._curr_student_states, self._student_kc_map), dim=1).to(device)
        info = {}
        return obs.detach().cpu().numpy(), info
    
    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }
    
    def log_improvements(self):
        old_preds = self.all_kc_preds[:, 0]
        new_preds = self.all_kc_preds[:, self.max_steps]
        improvement = new_preds - old_preds
        # Mean improvement: mean improvement of a student in 10 steps
        mean_improvement = improvement.mean()
        max_upper_bound = 1 - old_preds.mean()

        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            if self.use_ground_truth_response:
                print("Using Ground Truth Responses Results:", file=f, flush=True)
            else:
                print("NOT Using Ground Truth Responses Results:", file=f, flush=True)
            print({
                "GT Env Episode End - Mean Improvement": mean_improvement.mean().item(),
                "GT Env Episode End - Max Improvement": torch.max(mean_improvement).item(),
                "GT Env Episode End - Min Improvement": torch.min(mean_improvement).item(),
                "GT Env Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)
        if self.log_wandb:  
            wandb.log({
                "GT Env Episode End - Mean Improvement": mean_improvement,
                "GT Env Episode End - Max Improvement": torch.max(improvement),
                "GT Env Episode End - Min Improvement": torch.min(improvement),
                "GT Env Episode End - Upper Bound Improvement": max_upper_bound
            })
        

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        if action.shape[1] != self.question_embed_size:
            raise ValueError(f"Got second dimension {action.shape[1]}, expecting: {self.question_embed_size}")
        question_idx = self.init_seq_size + self.step_count
        new_que = self._qseqs[:, question_idx, :]
        if self.reward_func == "step_by_step":
            old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        if self.use_ground_truth_response:
            student_responses = self._rseqs[:, question_idx]
        else:
            with torch.no_grad():
                probs = self.kt_model.predict_in_rl(new_que)
            student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        obs = torch.cat((obs, self._student_kc_map), dim=1)
        new_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self.step_count += 1

        # Record the predictions in the all_kc_preds matrix
        self.all_kc_preds[:, self.step_count] = new_levels

        terminated = self.step_count >= self.max_steps
        if terminated:
            self.log_improvements()
        
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        truncated = False
        info = {}
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation


class FirstKCEvolutionContinuousEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                dataset_folds = {0},
                max_steps: int=10,
                init_seq_size:int=200,
                last_n_steps: int=200,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                question_bank = None,
                reward_scale = None,
                reward_func: str = "step_by_step",
                enforce_corpus: bool = True,
                seed=42,
                log_wandb:bool = False,
                log_path:str = "",
                wandb_prefix:str="",
                log_impr_path:str=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")
        
        super(FirstKCEvolutionContinuousEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        self.log_wandb = log_wandb
        self.log_path = log_path
        self.enforce_corpus = enforce_corpus
        if self.enforce_corpus:
            if not question_bank:
                raise ValueError("To enforce corpus, a QuestionBank instance must be provided")
        self.question_bank = question_bank
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()

        self.batch_size = batch_size
        history_generation_batch_size = 64 if self.batch_size > 64 else self.batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # Episode step count is the number of steps until reaching max steps
        # Total step count is the ongoing number of steps until changing students
        self.episode_step_count = 0
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.step_count = 0
        self.question_bank = question_bank
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.question_embed_size,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._internal_step_batch = -1 if self.batch_size <= 64 else 64

        self._is_reset = False

        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids

    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states(seq_size=self.init_seq_size)
        self._initial_student_states = self._curr_student_states
    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_cluster_embeddings(self):
        self._student_kc_map = torch.zeros((self.batch_size, self._kc_emb_dim), dtype=torch.float)
        for sid in range(self.batch_size):
            cluster_key = self.max_seen_clusters[sid]
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            # TODO: Change it back to randomness
            sampled_kc = random.sample(kc_keys, k=1)[0]
            #sampled_kc = kc_keys[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0) # Returns 1 by 768 -> squeezed to 768
            self._student_kc_map[sid] = _kc_emb
        self._student_kc_map = self._student_kc_map.to(device)
    
    def _init_preds(self):
        old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self._old_levels = old_levels
        self.all_kc_preds[:, 0] = old_levels

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.all_kc_preds = torch.full((self.batch_size, self.max_steps+1), -1.0, dtype=torch.float)
            self._init_kt_model()
            self._init_cluster_statistics()
            self._assign_cluster_embeddings()
            self._init_preds()
            self._is_reset = True
        else:
            print("Environment ended!")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            """
        obs = torch.cat((self._curr_student_states, self._student_kc_map), dim=1).to(device)
        info = {}
        return obs.detach().cpu().numpy(), info
    
    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }

    def log_improvements(self):
        old_preds = self.all_kc_preds[:, 0]
        new_preds = self.all_kc_preds[:, self.max_steps]
        improvement = new_preds - old_preds
        mean_improvement = improvement.mean()
        max_upper_bound = 1 - old_preds.mean()

        if self.log_wandb:
            wandb.log({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement,
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound
            })

        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            print({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement.item(),
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement).item(),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement).item(),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)


    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation
        """
        info = {}
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        if action.shape[1] != self.question_embed_size:
            raise ValueError(f"Got second dimension {action.shape[1]}, expecting: {self.question_embed_size}")
        if self.enforce_corpus:
            if self._internal_step_batch == -1:
                embs = self.question_bank.find_closest_embeddings(action)
            else:
                embs = []
                total_processed = 0
                while total_processed < action.shape[0]:
                    end_index = total_processed + self._internal_step_batch
                    batch_act = action[total_processed:end_index]
                    _embs = self.question_bank.find_closest_embeddings(batch_act)
                    embs.append(_embs)
                    total_processed += self._internal_step_batch
                embs = torch.cat(embs, dim=0)
        new_que = action # Shape becomes [batch_size, que_emb_size]
        del action
        if self.enforce_corpus:
            new_que = embs.to(device)
        # KC expertise level before
        if self.reward_func == "step_by_step":
            old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        np_old_levels = old_levels.clone().detach().cpu().numpy()
        info["y_kc_before"] = np_old_levels
        with torch.no_grad():
            probs = self.kt_model.predict_in_rl(new_que)
        info["y_que"] = probs.clone().detach().cpu().numpy()
        cell_state = self.kt_model.last_c.clone()
        cell_state = cell_state.squeeze(0)
        info["cell_state"] = cell_state.clone().detach().cpu().numpy()
        # The following line turns the prediction to 0 or 1.
        student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        obs = torch.cat((obs, self._student_kc_map), dim=1)
        new_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self.step_count += 1

        # Record preds
        self.all_kc_preds[:, self.step_count] = new_levels

        terminated = self.step_count >= self.max_steps
        if terminated:
            self.log_improvements()
        
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
            info["y_kc_after"] = new_levels.clone().detach().cpu().numpy()
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        if self.log_wandb:
            wandb.log({
                "Test Mean Sim Score": mean_sim_score,
                "Test Mean Reward": reward.mean(),
                "Test Mean Dist": dists.mean(),
                }, commit=True)
        truncated = False
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation


class FirstKCEvolutionDiscreteEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                dataset_folds = {0},
                max_steps: int=10,
                init_seq_size:int=200,
                last_n_steps: int=200,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                question_bank = None,
                reward_scale = None,
                reward_func: str = "step_by_step",
                seed=42,
                log_wandb:bool = False,
                log_path:str = "",
                wandb_prefix:str="",
                log_impr_path:str=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")
        
        super(FirstKCEvolutionDiscreteEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        self.log_wandb = log_wandb
        self.log_path = log_path
        if not question_bank:
            raise ValueError("Question Bank cannot be None to use this environment")
        self.question_bank = question_bank
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()

        self.batch_size = batch_size
        history_generation_batch_size = 64 if self.batch_size > 64 else self.batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # Episode step count is the number of steps until reaching max steps
        # Total step count is the ongoing number of steps until changing students
        self.episode_step_count = 0
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.step_count = 0
        self.question_bank = question_bank
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Discrete(
            n=self.question_bank.num_q, seed=seed, start=0
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._internal_step_batch = -1 if self.batch_size <= 64 else 64

        self._is_reset = False

        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids
    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states(seq_size=self.init_seq_size)
        self._initial_student_states = self._curr_student_states
    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_cluster_embeddings(self):
        self._student_kc_map = torch.zeros((self.batch_size, self._kc_emb_dim), dtype=torch.float)
        for sid in range(self.batch_size):
            cluster_key = self.max_seen_clusters[sid]
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            # TODO: Change it back to randomness
            sampled_kc = random.sample(kc_keys, k=1)[0]
            #sampled_kc = kc_keys[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0) # Returns 1 by 768 -> squeezed to 768
            self._student_kc_map[sid] = _kc_emb
        self._student_kc_map = self._student_kc_map.to(device)
    
    def _init_preds(self):
        old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self._old_levels = old_levels
        self.all_kc_preds[:, 0] = old_levels

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.all_kc_preds = torch.full((self.batch_size, self.max_steps+1), -1.0, dtype=torch.float)
            self._init_kt_model()
            self._init_cluster_statistics()
            self._assign_cluster_embeddings()
            self._init_preds()
            self._is_reset = True
        else:
            print("Environment ended!")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            """
        obs = torch.cat((self._curr_student_states, self._student_kc_map), dim=1).to(device)
        info = {}
        cell_state = self.kt_model.last_c.clone()
        cell_state = cell_state.squeeze(0)
        info["cell_state"] = cell_state.clone().detach().cpu().numpy()
        return obs.detach().cpu().numpy(), info
    
    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }

    def log_improvements(self):
        old_preds = self.all_kc_preds[:, 0]
        new_preds = self.all_kc_preds[:, self.max_steps]
        improvement = new_preds - old_preds

        mean_improvement = improvement.mean()
        max_upper_bound = 1 - old_preds.mean()

        if self.log_wandb:
            wandb.log({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement,
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound
            })
        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            print({
                f"{self.wandb_prefix} Episode End - Mean Improvement": mean_improvement.item(),
                f"{self.wandb_prefix} Episode End - Max Improvement": torch.max(improvement).item(),
                f"{self.wandb_prefix} Episode End - Min Improvement": torch.min(improvement).item(),
                f"{self.wandb_prefix} Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        
        new_que = self.question_bank.embeddings[action, :]
        # KC expertise level before
        if self.reward_func == "step_by_step":
            old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        with torch.no_grad():
            probs = self.kt_model.predict_in_rl(new_que)
        # The following line turns the prediction to 0 or 1.
        student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        obs = torch.cat((obs, self._student_kc_map), dim=1)
        new_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self.step_count += 1

        # Record the predictions in the all_kc_preds matrix
        self.all_kc_preds[:, self.step_count] = new_levels

        terminated = self.step_count >= self.max_steps
        if terminated:
            self.log_improvements()
        
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        if self.log_wandb:
            wandb.log({
                "Test Mean Sim Score": mean_sim_score,
                "Test Mean Reward": reward.mean(),
                "Test Mean Dist": dists.mean(),
                }, commit=True)
        truncated = False
        info = {}
        cell_state = self.kt_model.last_c.clone()
        cell_state = cell_state.squeeze(0)
        info["cell_state"] = cell_state.clone().detach().cpu().numpy()
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation


class FirstKCGroundTruthEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                history_generation_batch_size=64,
                dataset_folds = {0},
                max_steps: int=10,
                init_seq_size:int=100,
                last_n_steps: int=100,
                kc_to_que_path="",
                kc_emb_path="",
                cluster_to_kc_path= "",
                cluster_to_que_path= "",
                student_state_size: int=300,
                reward_scale = None,
                reward_func: str = "step_by_step",
                seed=42,
                log_wandb:bool = False,
                log_path:str = "",
                use_ground_truth_response: bool = False,
                wandb_prefix:str="",
                log_impr_path:str=""):
        """
        Initialize the environment where questions are represented and passed as embedding vectors.

        Parameters:
            question_embed_size (int): The size of the question embeddings. 
                                       It should match with the passed KT model.
            kt_model (CalibrationQDKTWrapper): An instance of the wrapper of the KT model to simulate the student.
            reward_func (RewardFunction): The reward function to be used.
            initial_history (dict): If starting from a none-empty history
            max_steps (int): Maximum number of steps per episode (default is 100)
            question_bank (QuestionBank): To be used to find the question in the question bank
                                            that is closest to the suggested question by the agent.
        """
        if pretrained_model_path == "":
            raise ValueError("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_que_path == "":
            raise ValueError("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise ValueError("Knowledge Concept embeddings path cannot be empty")
        
        if cluster_to_kc_path == "":
            raise ValueError("Cluster to Knowledge Concept mapping path cannot be empty")
        
        if cluster_to_que_path == "":
            raise ValueError("Cluster to Question IDs mapping path cannot be empty")
        
        super(FirstKCGroundTruthEnv, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.use_ground_truth_response = use_ground_truth_response
        self.log_wandb = log_wandb
        self.log_path = log_path
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_que_path = kc_to_que_path
        self.cluster_to_kc_path = cluster_to_kc_path
        self.cluster_to_que_path = cluster_to_que_path
        self._init_cluster_mappings()
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()

        self.batch_size = batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size, folds=dataset_folds, seed=seed, is_random=False)
        self._init_history()

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_n_steps = last_n_steps
        
        self.max_steps = max_steps
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.step_count = 0
        self.reward_scale = reward_scale if reward_scale else 1
        self.reward_func = reward_func
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.question_embed_size,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )

        self._internal_step_batch = -1 if self.batch_size <= 128 else 128

        self._is_reset = False
        self.wandb_prefix = wandb_prefix
        self.log_impr_path = log_impr_path
    
    def _init_history(self):

        all_qseqs = []
        all_rseqs = []
        all_qid_seqs = []
        total_batch = 0
        while total_batch < self.batch_size:
            qseqs, rseqs, qid_seqs = self.history.get_question_response_pairs()
            batch = qseqs.shape[0]
            total_batch += batch
            all_qseqs.append(qseqs)
            all_rseqs.append(rseqs)
            all_qid_seqs.append(qid_seqs)

        self._qseqs = torch.cat(all_qseqs, dim=0).to(device)
        self._rseqs = torch.cat(all_rseqs, dim=0).to(device)
        self._all_qid_seqs = torch.cat(all_qid_seqs, dim=0).to(device)
    
    def _init_cluster_mappings(self):
        map_dir = {}
        with open(self.cluster_to_kc_path, "r") as f:
            map_dir = json.load(f)
        self.cluster_to_kc_map = map_dir

        que_map_dir = {}
        with open(self.cluster_to_que_path, "r") as f:
            que_map_dir = json.load(f)
        self.cluster_to_que_map = que_map_dir
    
        self.num_clusters = len(self.cluster_to_que_map)

        que_to_cluster_ids = defaultdict(list)
        for cluster_id, qids in self.cluster_to_que_map.items():
            for qid in qids:
                que_to_cluster_ids[str(qid)].append(int(cluster_id))
            
        self.qid_to_cluster_ids = que_to_cluster_ids

    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_que_path, "r") as f:
            map_dir = json.load(f)
        self.kc_to_que_map = map_dir

        self.num_kcs = len(self.kc_to_que_map)

        self.kc_name_to_id = {}
        for i, kc_name in enumerate(self.kc_to_que_map.keys()):
            self.kc_name_to_id[kc_name] = i
        
        que_to_kc_ids = defaultdict(list)
        for kc, qids in self.kc_to_que_map.items():
            for qid in qids:
                que_to_kc_ids[str(qid)].append(self.kc_name_to_id[kc])
        self.qid_to_kc_ids = que_to_kc_ids

    def _init_kt_model(self):
        mini_batch_size = 64 if self.batch_size >= 64 else self.batch_size
        self._curr_student_states, self._qid_seqs = self.kt_model.init_states_with_data(
                                            mini_batch_size=mini_batch_size, 
                                            seq_size=self.init_seq_size,
                                            qseqs=self._qseqs,
                                            rseqs=self._rseqs,
                                            qid_seqs=self._all_qid_seqs)

        self._initial_student_states = self._curr_student_states

    
    def _init_cluster_statistics(self):
        self._cluster_matrix = np.zeros((self.batch_size, self.num_clusters), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_n_steps:]:
                qid = qid.item()
                for cluster_id in self.qid_to_cluster_ids[str(qid)]:
                    self._cluster_matrix[student_id, cluster_id] += 1
        
        self.max_seen_clusters = np.argmax(self._cluster_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)
        
        precomputed_kc_emb_tensors = torch.empty(self.num_kcs, self.question_embed_size)
        for key in list(emb_dir.keys()):
            index = self.kc_name_to_id[key]
            precomputed_kc_emb_tensors[index] = torch.tensor(emb_dir[key], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_cluster_embeddings(self):
        self._student_kc_map = torch.zeros((self.batch_size, self._kc_emb_dim), dtype=torch.float)
        for sid in range(self.batch_size):
            cluster_key = self.max_seen_clusters[sid]
            kc_keys = self.cluster_to_kc_map[str(cluster_key)]
            # TODO: Change it back to randomness
            sampled_kc = random.sample(kc_keys, k=1)[0]
            #sampled_kc = kc_keys[0]
            kc_id = self.kc_name_to_id[sampled_kc]
            kc_id = torch.tensor([kc_id], dtype=torch.int)
            _kc_emb = self.kc_emb(kc_id).squeeze(0) # Returns 1 by 768 -> squeezed to 768
            self._student_kc_map[sid] = _kc_emb
        self._student_kc_map = self._student_kc_map.to(device)
    
    def _init_preds(self):
        old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self._old_levels = old_levels
        self.all_kc_preds[:, 0] = old_levels

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.step_count = 0
        if not self._is_reset:
            self.all_kc_preds = torch.full((self.batch_size, self.max_steps+1), -1.0, dtype=torch.float)
            self._init_kt_model()
            self._init_cluster_statistics()
            self._assign_cluster_embeddings()
            self._init_preds()
            self._is_reset = True
        else:
            print("Environment ended!")
            """
            self._curr_student_states = self._initial_student_states
            self.kt_model.reset_in_rl()
            """
        obs = torch.cat((self._curr_student_states, self._student_kc_map), dim=1).to(device)
        info = {}
        return obs.detach().cpu().numpy(), info
    
    def _get_state(self):
        """
        Get the current state representation
        Returns:
            state (np.ndarray): The student's current knowledge state
        """
        if isinstance(self._curr_student_states, torch.Tensor):
            return self._curr_student_states.detach().cpu().numpy()
        else:
            raise ValueError("The student state must be a torch.Tensor.")
    
    def _calculate_reward(self, response):
        """
        """
        raise NotImplementedError
    
    def _get_history(self):
        return self.history
    
    def _get_info(self):
        return {
            "history": self.history,
            "states": self._curr_student_states
        }
    
    def log_improvements(self):
        old_preds = self.all_kc_preds[:, 0]
        new_preds = self.all_kc_preds[:, self.max_steps]
        improvement = new_preds - old_preds
        # Mean improvement: mean improvement of a student in 10 steps
        mean_improvement = improvement.mean()
        max_upper_bound = 1 - old_preds.mean()

        with open(os.path.join(self.log_impr_path, "end_results.txt"), "w+") as f:
            if self.use_ground_truth_response:
                print("Using Ground Truth Responses Results:", file=f, flush=True)
            else:
                print("NOT Using Ground Truth Responses Results:", file=f, flush=True)
            print({
                "GT Env Episode End - Mean Improvement": mean_improvement.mean().item(),
                "GT Env Episode End - Max Improvement": torch.max(mean_improvement).item(),
                "GT Env Episode End - Min Improvement": torch.min(mean_improvement).item(),
                "GT Env Episode End - Upper Bound Improvement": max_upper_bound.item()
            }, file=f, flush=True)
        if self.log_wandb:  
            wandb.log({
                "GT Env Episode End - Mean Improvement": mean_improvement,
                "GT Env Episode End - Max Improvement": torch.max(improvement),
                "GT Env Episode End - Min Improvement": torch.min(improvement),
                "GT Env Episode End - Upper Bound Improvement": max_upper_bound
            })

    def step(self, action):
        """
        Take a step in the environment with given questions and responses.

        The KC embeddings associated with each student is concatenated to the observation
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=device)
        if action.shape[0] != self.batch_size:
            raise ValueError(f"Got first dimension: {action.shape[0]}, expecting: {self.batch_size}")
        if action.shape[1] != self.question_embed_size:
            raise ValueError(f"Got second dimension {action.shape[1]}, expecting: {self.question_embed_size}")
        
        # The index of the next question from the dataset is self.init_seq_size + self.step_count
        # EX: init_seq_size = 100, step_count is 0 -> 100th question is the next question
        question_idx = self.init_seq_size + self.step_count
        new_que = self._qseqs[:, question_idx, :]
        # KC expertise level before
        if self.reward_func == "step_by_step":
            old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        # The following line turns the prediction to 0 or 1.
        if self.use_ground_truth_response:
            student_responses = self._rseqs[:, question_idx]
            with open(os.path.join(self.log_path, "student_responses.csv"), "a+") as f:
                np.savetxt(f, [student_responses.detach().cpu().numpy()], delimiter=",")
        else:
            with torch.no_grad():
                probs = self.kt_model.predict_in_rl(new_que)
                with open(os.path.join(self.log_path, "step_probs.csv"), "a+") as f:
                    np.savetxt(f, [probs.detach().cpu().numpy()], delimiter=",")
            student_responses = (torch.rand_like(probs) < probs).float()
            with open(os.path.join(self.log_path, "student_responses.csv"), "a+") as f:
                np.savetxt(f, [student_responses.detach().cpu().numpy()], delimiter=",")

        obs, _, __ = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        obs = torch.cat((obs, self._student_kc_map), dim=1)
        new_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        self.step_count += 1

        # Record the predictions in the all_kc_preds matrix
        self.all_kc_preds[:, self.step_count] = new_levels

        terminated = self.step_count >= self.max_steps
        if terminated:
            self.log_improvements()
        
        if self.reward_func == "step_by_step":
            reward = new_levels - old_levels
            reward = reward * self.reward_scale
        elif self.reward_func == "episode_end":
            if terminated:
                reward = new_levels - self._old_levels
                reward = reward * self.reward_scale
            else:
                reward = torch.zeros((self.batch_size), dtype=torch.float32)
        else:
            raise ValueError(f"Given reward_func: {self.reward_func} is not supported")
        if self.log_wandb:
            wandb.log({
                "Test Mean Reward": reward.mean()
                }, commit=True)
        truncated = False
        info = {}
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation
    


