import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import sys
from exercise_recommender.wrappers.calibrationqdkt_wrapper import CalibrationQDKTWrapper
from exercise_recommender.utils.data import get_history_generator


class QuestionVectorEnvironment(gym.Env):
    def __init__(self, 
                question_embed_size:int,
                pretrained_model_path="",
                reward_func = None,
                batch_size: int=1024,
                history_generation_batch_size=64,
                max_steps:int=100,
                init_seq_size:int=200,
                question_bank = None):
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
        
        super(QuestionVectorEnvironment, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reward_func = reward_func
        self.question_embed_size = question_embed_size
        self.init_seq_size = init_seq_size

        self.batch_size = batch_size
        self.history = get_history_generator(batch_size=history_generation_batch_size)
        self.kt_model = CalibrationQDKTWrapper(history_generator=self.history, pretrained_model_path=pretrained_model_path, total_batch_size=self.batch_size)
        self._init_kt_model()

        self.step_count = 0
        self.max_steps = max_steps
        self._done = False
        self.question_bank = question_bank
        

        if isinstance(self._curr_student_states, torch.Tensor):
            self._student_state_size = self._curr_student_states.shape[1]
        else:
            raise ValueError("The student state must be of type torch.Tensor")
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.question_embed_size,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._student_state_size, ), dtype=np.float32
        )
    
    def _init_kt_model(self):
        self._curr_student_states, *_ = self.kt_model.init_states(seq_size=self.init_seq_size)

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self._curr_student_states, *_ = self.kt_model.init_states(seq_size=self.init_seq_size)
        self.step_count = 0
        info = {}
        return self._curr_student_states.detach().cpu().numpy(), info
    
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
        """
        # TODO: Make the step functionality more modular
        """
            Before putting in the action, get the prediction of the student answering that question
            Then, from that prediction, decide if the student answers correctly or not
            Then, update the internal state with that response
            Then, return the different between the prev prediction and after the internal state is updates
        """        
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        new_que = action # Shape becomes [batch_size, que_emb_size]
        with torch.no_grad():
            old_prob = self.kt_model.predict_in_rl(new_que)
        old_pred = old_prob
        # The following line turns the prediction to 0 or 1.
        student_responses = (torch.rand_like(old_prob) < old_prob).float()
        obs = self.kt_model.update_hidden_state(new_que.unsqueeze(1), student_responses.unsqueeze(1))
        new_pred = self.kt_model.predict_in_rl(new_que)
        reward = new_pred - old_pred
        # TODO: Update terminated and truncated
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {}
        return obs.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info # TODO: Return additional info from KT like mastery or current latent representation
    


