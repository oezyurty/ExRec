import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import sys
from exercise_recommender.wrappers.calibrationqdkt_wrapper import CalibrationQDKTWrapper
from exercise_recommender.utils.data import get_history_generator
import json
from collections import defaultdict

# TODO: Write now, we store the embedding of the most seen KC along with each student
# Make this logic more modular
device = "cuda" if torch.cuda.is_available() else "cpu"
class KCVectorEnv(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                batch_size: int=1024,
                history_generation_batch_size=64,
                max_steps: int=10,
                max_steps_until_student_change: int=100,
                init_seq_size:int=200,
                last_kc_steps: int=200,
                kc_to_questions_path="",
                kc_emb_path="",
                student_state_size: int=300,
                question_bank = None,
                reward_scale = None,
                reward_func = None):
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
            raise("A path to a pretrained model should be given to use this environment.")
        
        if kc_to_questions_path == "":
            raise("A path to knowledge concepts to questions mapping should be provided")
        
        if kc_emb_path == "":
            raise("Knowledge Concept embeddings path cannot be empty")
        
        super(KCVectorEnv, self).__init__()

        
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size
        self.kc_to_questions_path = kc_to_questions_path
        self._init_kc_que_map()
        self.kc_emb_path = kc_emb_path
        self._init_kc_embeddings()

        self.batch_size = batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size)

        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self.last_kc_steps = last_kc_steps
        
        self.max_steps = max_steps
        # Episode step count is the number of steps until reaching max steps
        # Total step count is the ongoing number of steps until changing students
        self.episode_step_count = 0
        # If step_count reaches max_steps until student change, the a new batch of random students will be initialized
        self.max_steps_until_student_change = max_steps_until_student_change
        self.total_step_count = 0
        self._done = False
        self.question_bank = question_bank
        self.reward_scale = reward_scale if reward_scale else 1
        
        self._student_state_size = student_state_size
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.question_embed_size,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size + self._kc_emb_dim, ), dtype=np.float32
        )
    
    def _init_kc_que_map(self):
        map_dir = {}
        with open(self.kc_to_questions_path, "r") as f:
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
    
    def _init_kc_statistics(self):
        self._kc_matrix = np.zeros((self.batch_size, self.num_kcs), dtype=int)
        for student_id in range(self.batch_size):
            for qid in self._qid_seqs[student_id][-self.last_kc_steps:]:
                qid = qid.item()
                for kc_id in self.qid_to_kc_ids[str(qid)]:
                    self._kc_matrix[student_id, kc_id] += 1
        
        self.max_seen_kcs = np.argmax(self._kc_matrix, axis=1)

    def _init_kc_embeddings(self):
        emb_dir = {}
        with open(self.kc_emb_path, "r") as f:
            emb_dir = json.load(f)

        precomputed_kc_emb_tensors = torch.tensor([emb_dir[key] for key in emb_dir], dtype=torch.float)
        
        num_kc, emb_size = precomputed_kc_emb_tensors.shape
        self._kc_emb_dim = emb_size

        orig_norm = precomputed_kc_emb_tensors[0].norm()

        norms = precomputed_kc_emb_tensors.norm(p=2, dim=1, keepdim=True)
        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors/norms

        precomputed_kc_emb_tensors = precomputed_kc_emb_tensors * np.sqrt(emb_size)

        self.kc_emb = torch.nn.Embedding.from_pretrained(precomputed_kc_emb_tensors, freeze=True)
    
    def _assign_kc_embeddings(self):
        self._student_kc_map = torch.zeros((self.batch_size, self._kc_emb_dim), dtype=torch.float)
        for sid in range(self.batch_size):
            max_seen_kc_id = torch.tensor([self.max_seen_kcs[sid]], dtype=torch.int)
            _kc_emb = self.kc_emb(max_seen_kc_id).squeeze(0) # Returns 1 by 768 -> squeezed to 768
            self._student_kc_map[sid] = _kc_emb
        self._student_kc_map = self._student_kc_map.to(device)

    def reset(self):
        """
        Reset the environment to its initial state
        """
        if self.total_step_count == 0 or self.total_step_count >= self.max_steps_until_student_change:
            self._init_kt_model()
            self._init_kc_statistics()
            self._assign_kc_embeddings()
            self.total_step_count = 1
        else:
            self._curr_student_states = self._initial_student_states
        self.step_count = 0
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
        new_que = action # Shape becomes [batch_size, que_emb_size]
        # KC expertise level before
        old_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        with torch.no_grad():
            probs = self.kt_model.predict_in_rl(new_que)
        # The following line turns the prediction to 0 or 1.
        student_responses = (torch.rand_like(probs) < probs).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        obs = torch.cat((obs, self._student_kc_map), dim=1)
        # KC expertise level after
        new_levels = self.kt_model.predict_in_rl(self._student_kc_map)
        reward = new_levels - old_levels
        reward = reward * self.reward_scale
        self.step_count += 1
        self.total_step_count += 1
        terminated = self.step_count >= self.max_steps
        if not terminated:
            reward = torch.zeros_like(reward)
        truncated = False
        info = {}
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation
    


