from exercise_recommender.envs.cluster_vector_env import ClusterVectorEnv
from exercise_recommender.envs.test_cluster_vector_env import TestClusterVectorEnv
from exercise_recommender.envs.vector_env_wrapper import VectorEnvWrapper
from exercise_recommender.utils.question_bank import QuestionBank
from exercise_recommender.utils.data import get_history_generator
import torch
from exercise_recommender.agents.kc_actor import KCActor
from exercise_recommender.agents.kc_critic import KCCritic
from exercise_recommender.agents.critic_dkt import CriticDKT
from exercise_recommender.wrappers.ddpg_wrapper import DDPGPolicyWrapper
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.logger.wandb import WandbLogger
from tianshou.exploration.random import GaussianNoiseScheduler
from tianshou.utils.torch_utils import policy_within_training_step, torch_train_mode

import wandb
import uuid
import os

import argparse

def get_critic(params):
    """
        Params should have a parameter called critic_model
        that must be one of the followings:
            kc_critic
            critic_dkt
    """
    if params["critic_model"] == "critic_dkt":
        return CriticDKT(
            path_dkt=params["pretrained_model_path"],
            student_hidden_size=params["student_state_size"],
            reward_scale=params["train_reward_scale"]
        )
    elif params["critic_model"] == "kc_critic":
        return KCCritic(
            student_hidden_size=params["student_state_size"],
            kc_emb_size=params["kc_emb_size"],
            action_size=params["action_size"],
            hidden_size=params["critic_hidden_size"],
            up_projection_size=params["critic_up_projection_size"]
        )
    else:
        raise ValueError("Provided Critic model is not supported")

def train(params):

    folder_name = f"{str(uuid.uuid4())}"
    log_dir = params["log_dir"]
    if not os.path.exists(os.path.join(log_dir, folder_name)):
        os.makedirs(os.path.join(log_dir, folder_name))
    
    log_path = os.path.join(log_dir, folder_name)
    
    save_dir = params["save_dir"]
    if not os.path.exists(os.path.join(save_dir, folder_name)):
        os.makedirs(os.path.join(save_dir, folder_name))

    ckpt_path = os.path.join(save_dir, folder_name)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    params["checkpoint_path"] = ckpt_path
    params["log_path"] = log_path

    if params["use_wandb"]:
        wandb.init(
            project=params["wandb_project_name"],
            tags=["ddpg"],
            config=params
        )

    train_env = ClusterVectorEnv(
            question_embed_size=768, 
            pretrained_model_path=params["pretrained_model_path"],
            init_seq_size=params["train_init_seq_size"], 
            batch_size=params["train_batch_size"],
            max_steps=params["train_max_steps"],
            max_steps_until_student_change=params["train_max_steps_until_student_change"],
            student_state_size=params["student_state_size"],
            last_n_steps=params["train_last_n_steps"],
            dataset_folds={2,3,4},
            kc_to_que_path=params["kc_to_que_path"],
            kc_emb_path=params["kc_emb_path"],
            reward_scale=params["train_reward_scale"],
            cluster_to_kc_path=params["cluster_to_kc_path"],
            cluster_to_que_path=params["cluster_to_que_path"],
            reward_func=params["train_reward_func"])

    if params["use_wandb"]:
        wandb.config.update({
            "train_folds": "2-3-4"
        })

    old_question_bank = QuestionBank(
        que_emb_path="../data/XES3G5M_embeddings/qid2content_sol_avg_emb.json"
    )

    old_ques_valid_env = TestClusterVectorEnv(
            question_embed_size=768, 
            pretrained_model_path=params["pretrained_model_path"],
            init_seq_size=params["test_init_seq_size"],
            batch_size=params["test_batch_size"],
            max_steps=params["test_max_steps"],
            student_state_size=params["student_state_size"],
            last_n_steps=params["test_last_n_steps"],
            dataset_folds={0},
            kc_to_que_path=params["kc_to_que_path"],
            kc_emb_path=params["kc_emb_path"],
            reward_scale=params["test_reward_scale"],
            cluster_to_kc_path=params["cluster_to_kc_path"],
            cluster_to_que_path=params["cluster_to_que_path"],
            reward_func=params["test_reward_func"],
            enforce_corpus=True,
            log_wandb=params["test_log_wandb"],
            question_bank=old_question_bank)
    
    extended_question_bank = QuestionBank(
        que_emb_path="../data/all_questions_embeddings.json"
    )

    extended_ques_valid_env = TestClusterVectorEnv(
            question_embed_size=768, 
            pretrained_model_path=params["pretrained_model_path"],
            init_seq_size=params["test_init_seq_size"],
            batch_size=params["test_batch_size"],
            max_steps=params["test_max_steps"],
            student_state_size=params["student_state_size"],
            last_n_steps=params["test_last_n_steps"],
            dataset_folds={0},
            kc_to_que_path=params["kc_to_que_path"],
            kc_emb_path=params["kc_emb_path"],
            reward_scale=params["test_reward_scale"],
            cluster_to_kc_path=params["cluster_to_kc_path"],
            cluster_to_que_path=params["cluster_to_que_path"],
            reward_func=params["test_reward_func"],
            enforce_corpus=True,
            log_wandb=params["test_log_wandb"],
            question_bank=extended_question_bank)

    actor = KCActor(
        student_hidden_size=params["student_state_size"],
        kc_emb_size=params["kc_emb_size"],
        action_size=params["action_size"],
        hidden_size=params["hidden_size"]
    ).to(device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=params["actor_lr"])

    critic = get_critic(params)
    critic = critic.to(device)

    critic_optim = torch.optim.Adam(critic.parameters(), lr=params["critic_lr"])

    print(f"Actor and Critic Networks loaded into: {device}")

    exploration_noise = GaussianNoiseScheduler(sigma=params["initial_exploration_noise_sigma"], final_sigma=params["final_exploration_noise_sigma"], decay_steps=params["exploration_noise_decay_steps"])

    observation_space = train_env.observation_space
    action_space = train_env.action_space

    policy = DDPGPolicyWrapper(
        actor=actor,
        critic=critic,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        observation_space=observation_space,
        action_space=action_space,
        action_scaling=False,
        action_bound_method=None,
        tau=params["tau"],
        gamma=params["gamma"],
        exploration_noise=exploration_noise
    )

    train_vectorized_env = VectorEnvWrapper(train_env)
    buffer_size = params["train_replay_buffer_size"]
    train_replay_buffer = VectorReplayBuffer(buffer_size*train_vectorized_env.env_num, train_vectorized_env.env_num)
    train_collector = Collector(policy, train_vectorized_env, train_replay_buffer, exploration_noise=True)

    old_ques_valid_vectorized_env = VectorEnvWrapper(old_ques_valid_env)
    old_ques_valid_replay_buffer = VectorReplayBuffer(1*old_ques_valid_vectorized_env.env_num, old_ques_valid_vectorized_env.env_num)
    old_ques_valid_collector = Collector(policy, old_ques_valid_vectorized_env, old_ques_valid_replay_buffer, exploration_noise=True)

    extended_ques_valid_vectorized_env = VectorEnvWrapper(extended_ques_valid_env)
    extended_ques_valid_replay_buffer = VectorReplayBuffer(1*extended_ques_valid_vectorized_env.env_num, extended_ques_valid_vectorized_env.env_num)
    extended_ques_valid_collector = Collector(policy, extended_ques_valid_vectorized_env, extended_ques_valid_replay_buffer, exploration_noise=True)

    train_collector.reset()
    old_ques_valid_collector.reset()
    extended_ques_valid_collector.reset()

    train_replay_buffer.reset()
    old_ques_valid_collector.reset()
    extended_ques_valid_collector.reset()

    old_ques_best_mean_reward = float('-inf')
    old_ques_std_of_best_mean_reward = 0
    old_ques_min_of_best_mean_reward = 0
    old_ques_max_of_best_mean_reward = 0

    extended_ques_best_mean_reward = float("-inf")
    extended_ques_std_of_best_mean_reward = 0
    extended_ques_min_of_best_mean_reward = 0
    extended_ques_max_of_best_mean_reward = 0

    with torch_train_mode(policy, enabled=False):
        old_ques_evaluation_result = old_ques_valid_collector.collect(n_episode=params["test_n_episode"])
        extended_ques_evaluation_result = extended_ques_valid_collector.collect(n_episode=params["test_n_episode"])
        if params["use_wandb"]:
            wandb.log({
                "old_ques_valid/return_stats/mean": old_ques_evaluation_result.returns_stat.mean,
                "old_ques_valid/return_stats/std": old_ques_evaluation_result.returns_stat.std,
                "old_ques_valid/return_stats/max": old_ques_evaluation_result.returns_stat.max,
                "old_ques_valid/return_stats/min": old_ques_evaluation_result.returns_stat.min
            }, commit=True)
            wandb.log({
                "extended_ques_valid/return_stats/mean": extended_ques_evaluation_result.returns_stat.mean,
                "extended_ques_valid/return_stats/std": extended_ques_evaluation_result.returns_stat.std,
                "extended_ques_valid/return_stats/max": extended_ques_evaluation_result.returns_stat.max,
                "extended_ques_valid/return_stats/min": extended_ques_evaluation_result.returns_stat.min
            }, commit=True)
        
        old_ques_best_mean_reward = old_ques_evaluation_result.returns_stat.mean
        old_ques_std_of_best_mean_reward = old_ques_evaluation_result.returns_stat.std
        old_ques_min_of_best_mean_reward = old_ques_evaluation_result.returns_stat.min
        old_ques_max_of_best_mean_reward = old_ques_evaluation_result.returns_stat.max

        extended_ques_best_mean_reward = extended_ques_evaluation_result.returns_stat.mean
        extended_ques_std_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.std
        extended_ques_min_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.min
        extended_ques_max_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.max

        print(f"Old Ques Starting Mean Reward: {old_ques_best_mean_reward}")
        print(f"Extended Ques Starting Mean Reward: {extended_ques_best_mean_reward}")

    n_epoch = params["n_epoch"]
    sample_per_epoch = params["sample_per_epoch"]
    batch_size = params["batch_size"]

    for i in range(n_epoch):
        with policy_within_training_step(policy):
            train_res = train_collector.collect(n_episode=params["train_n_episode"])
            if params["use_wandb"]:
                wandb.log({
                    "train/return_stats/mean": train_res.returns_stat.mean,
                    "train/return_stats/std": train_res.returns_stat.std,
                    "train/return_stats/max": train_res.returns_stat.max,
                    "train/return_stats/min": train_res.returns_stat.min,
                }, commit=True)
            for _ in range(sample_per_epoch):
                with torch_train_mode(policy):
                    res = policy.update(sample_size=batch_size, buffer=train_collector.buffer)
                    if params["use_wandb"]:
                        wandb.log({
                            "actor_loss": res.actor_loss,
                            "critic_loss": res.critic_loss,
                        }, commit=True)
        with torch_train_mode(policy, enabled=False):
            old_ques_evaluation_result = old_ques_valid_collector.collect(n_episode=params["test_n_episode"])
            extended_ques_evaluation_result = extended_ques_valid_collector.collect(n_episode=params["test_n_episode"])
            if params["use_wandb"]:
                wandb.log({
                    "old_ques_valid/return_stats/mean": old_ques_evaluation_result.returns_stat.mean,
                    "old_ques_valid/return_stats/std": old_ques_evaluation_result.returns_stat.std,
                    "old_ques_valid/return_stats/max": old_ques_evaluation_result.returns_stat.max,
                    "old_ques_valid/return_stats/min": old_ques_evaluation_result.returns_stat.min
                }, commit=True)
                wandb.log({
                    "extended_ques_valid/return_stats/mean": extended_ques_evaluation_result.returns_stat.mean,
                    "extended_ques_valid/return_stats/std": extended_ques_evaluation_result.returns_stat.std,
                    "extended_ques_valid/return_stats/max": extended_ques_evaluation_result.returns_stat.max,
                    "extended_ques_valid/return_stats/min": extended_ques_evaluation_result.returns_stat.min
                }, commit=True)

            old_ques_mean_rew = old_ques_evaluation_result.returns_stat.mean
            if old_ques_mean_rew > old_ques_best_mean_reward + 1e-4:
                old_ques_best_mean_reward = old_ques_mean_rew
                old_ques_std_of_best_mean_reward = old_ques_evaluation_result.returns_stat.std
                old_ques_min_of_best_mean_reward = old_ques_evaluation_result.returns_stat.min
                old_ques_max_of_best_mean_reward = old_ques_evaluation_result.returns_stat.max
                torch.save(policy.state_dict(), os.path.join(ckpt_path, "old_ques_pretrained_model.ckpt"))
                print(f"Old Ques New Best Mean Reward: {old_ques_best_mean_reward}")
            
            extended_ques_mean_rew = extended_ques_evaluation_result.returns_stat.mean
            if extended_ques_mean_rew > extended_ques_best_mean_reward:
                extended_ques_best_mean_reward = extended_ques_mean_rew 
                extended_ques_std_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.std
                extended_ques_min_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.min
                extended_ques_max_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.max
                torch.save(policy.state_dict(), os.path.join(ckpt_path, "extended_ques_pretrained_model.ckpt"))
                print(f"Extended Ques New Best Reward: {extended_ques_best_mean_reward}")

    if params["use_wandb"]:
        wandb.log({
            "Final Old Ques Validation Mean Reward": old_ques_best_mean_reward,
            "Final Old Ques Validation Reward STD": old_ques_std_of_best_mean_reward,
            "Final Old Ques Validation Reward Max": old_ques_max_of_best_mean_reward,
            "Final Old Ques Validation Reward Min": old_ques_min_of_best_mean_reward,
        }, commit=True)
        wandb.log({
            "Final Extended Ques Validation Mean Reward": extended_ques_best_mean_reward,
            "Final Extended Ques Validation Reward STD": extended_ques_std_of_best_mean_reward,
            "Final Extended Ques Validation Reward Max": extended_ques_max_of_best_mean_reward,
            "Final Extended Ques Validation Reward Min": extended_ques_min_of_best_mean_reward,
        }, commit=True)

    # Remove valid_env and Question Bank for memory
    del old_question_bank
    del extended_question_bank
    del old_ques_valid_env
    del extended_ques_valid_env
    del old_ques_valid_vectorized_env
    del extended_ques_valid_vectorized_env
    del old_ques_valid_replay_buffer
    del extended_ques_valid_replay_buffer
    del old_ques_valid_collector
    del extended_ques_valid_collector
    del policy
    del train_env
    del train_vectorized_env
    del train_replay_buffer
    del train_collector

    question_bank_old = QuestionBank(
        que_emb_path="../data/XES3G5M_embeddings/qid2content_sol_avg_emb.json"
    )

    test_env_old_questions = TestClusterVectorEnv(
            question_embed_size=768, 
            pretrained_model_path=params["pretrained_model_path"],
            init_seq_size=params["test_init_seq_size"],
            batch_size=params["test_batch_size"],
            max_steps=params["test_max_steps"],
            student_state_size=params["student_state_size"],
            last_n_steps=params["test_last_n_steps"],
            dataset_folds={1},
            kc_to_que_path=params["kc_to_que_path"],
            kc_emb_path=params["kc_emb_path"],
            reward_scale=params["test_reward_scale"],
            cluster_to_kc_path=params["cluster_to_kc_path"],
            cluster_to_que_path=params["cluster_to_que_path"],
            reward_func=params["test_reward_func"],
            enforce_corpus=True,
            log_wandb=params["test_log_wandb"],
            question_bank=question_bank_old)
    
    old_ques_best_policy = DDPGPolicyWrapper(
        actor=actor,
        critic=critic,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        observation_space=observation_space,
        action_space=action_space,
        action_scaling=False,
        action_bound_method=None,
        tau=params["tau"],
        gamma=params["gamma"],
        exploration_noise=exploration_noise
    )
    
    old_ques_best_policy.load_state_dict(torch.load(os.path.join(ckpt_path, "old_ques_pretrained_model.ckpt")))

    test_vectorized_env = VectorEnvWrapper(test_env_old_questions)
    test_replay_buffer = VectorReplayBuffer(10*test_vectorized_env.env_num, test_vectorized_env.env_num)
    test_collector = Collector(old_ques_best_policy, test_vectorized_env, test_replay_buffer, exploration_noise=True)

    test_collector.reset()
    test_replay_buffer.reset()

    with torch_train_mode(old_ques_best_policy, enabled=False):
        test_result = test_collector.collect(n_episode=params["test_n_episode"])
        if params["use_wandb"]:
            wandb.log({
                "Old Ques Validated Old Questions test/return_stats/mean": test_result.returns_stat.mean,
                "Old Ques Validated Old Questions test/return_stats/std": test_result.returns_stat.std,
                "Old Ques Validated Old Questions test/return_stats/max": test_result.returns_stat.max,
                "Old Ques Validated Old Questions test/return_stats/min": test_result.returns_stat.min
            }, commit=True)
        print(f"Old Questions Validated Old Questions Test Mean Reward: {test_result.returns_stat.mean}")

    del question_bank_old
    del test_env_old_questions
    del test_vectorized_env
    del test_replay_buffer
    del test_collector

    question_bank_extended = QuestionBank(
        que_emb_path="../data/all_questions_embeddings.json"
    )

    test_env_extended_questions = TestClusterVectorEnv(
            question_embed_size=768, 
            pretrained_model_path=params["pretrained_model_path"],
            init_seq_size=params["test_init_seq_size"],
            batch_size=params["test_batch_size"],
            max_steps=params["test_max_steps"],
            student_state_size=params["student_state_size"],
            last_n_steps=params["test_last_n_steps"],
            dataset_folds={1},
            kc_to_que_path=params["kc_to_que_path"],
            kc_emb_path=params["kc_emb_path"],
            reward_scale=params["test_reward_scale"],
            cluster_to_kc_path=params["cluster_to_kc_path"],
            cluster_to_que_path=params["cluster_to_que_path"],
            reward_func=params["test_reward_func"],
            enforce_corpus=True,
            log_wandb=params["test_log_wandb"],
            question_bank=question_bank_extended)
    
    test_vectorized_env = VectorEnvWrapper(test_env_extended_questions)
    test_replay_buffer = VectorReplayBuffer(10*test_vectorized_env.env_num, test_vectorized_env.env_num)
    test_collector = Collector(old_ques_best_policy, test_vectorized_env, test_replay_buffer, exploration_noise=True)

    test_collector.reset()
    test_replay_buffer.reset()

    with torch_train_mode(old_ques_best_policy, enabled=False):
        test_result = test_collector.collect(n_episode=params["test_n_episode"])
        if params["use_wandb"]:
            wandb.log({
                "Old Ques Validated Extended Questions test/return_stats/mean": test_result.returns_stat.mean,
                "Old Ques Validated Extended Questions test/return_stats/std": test_result.returns_stat.std,
                "Old Ques Validated Extended Questions test/return_stats/max": test_result.returns_stat.max,
                "Old Ques Validated Extended Questions test/return_stats/min": test_result.returns_stat.min
            }, commit=True)
        print(f"Old Ques Validated Extended Test Mean Reward: {test_result.returns_stat.mean}")

    del question_bank_extended
    del test_env_extended_questions
    del test_vectorized_env
    del test_replay_buffer
    del test_collector
    del old_ques_best_policy

    extended_ques_best_policy = DDPGPolicyWrapper(
        actor=actor,
        critic=critic,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        observation_space=observation_space,
        action_space=action_space,
        action_scaling=False,
        action_bound_method=None,
        tau=params["tau"],
        gamma=params["gamma"],
        exploration_noise=exploration_noise
    )

    extended_ques_best_policy.load_state_dict(torch.load(os.path.join(ckpt_path, "extended_ques_pretrained_model.ckpt")))

    question_bank_old = QuestionBank(
        que_emb_path="../data/XES3G5M_embeddings/qid2content_sol_avg_emb.json"
    )

    test_env_old_questions = TestClusterVectorEnv(
            question_embed_size=768, 
            pretrained_model_path=params["pretrained_model_path"],
            init_seq_size=params["test_init_seq_size"],
            batch_size=params["test_batch_size"],
            max_steps=params["test_max_steps"],
            student_state_size=params["student_state_size"],
            last_n_steps=params["test_last_n_steps"],
            dataset_folds={1},
            kc_to_que_path=params["kc_to_que_path"],
            kc_emb_path=params["kc_emb_path"],
            reward_scale=params["test_reward_scale"],
            cluster_to_kc_path=params["cluster_to_kc_path"],
            cluster_to_que_path=params["cluster_to_que_path"],
            reward_func=params["test_reward_func"],
            enforce_corpus=True,
            log_wandb=params["test_log_wandb"],
            question_bank=question_bank_old)

    test_vectorized_env = VectorEnvWrapper(test_env_old_questions)
    test_replay_buffer = VectorReplayBuffer(10*test_vectorized_env.env_num, test_vectorized_env.env_num)
    test_collector = Collector(extended_ques_best_policy, test_vectorized_env, test_replay_buffer, exploration_noise=True)

    test_collector.reset()
    test_replay_buffer.reset()

    with torch_train_mode(extended_ques_best_policy, enabled=False):
        test_result = test_collector.collect(n_episode=params["test_n_episode"])
        if params["use_wandb"]:
            wandb.log({
                "Extended Ques Validated Old Questions test/return_stats/mean": test_result.returns_stat.mean,
                "Extended Ques Validated Old Questions test/return_stats/std": test_result.returns_stat.std,
                "Extended Ques Validated Old Questions test/return_stats/max": test_result.returns_stat.max,
                "Extended Ques Validated Old Questions test/return_stats/min": test_result.returns_stat.min
            }, commit=True)
        print(f"Extended Questions Validated Old Questions Test Mean Reward: {test_result.returns_stat.mean}")

    del question_bank_old
    del test_env_old_questions
    del test_vectorized_env
    del test_replay_buffer
    del test_collector

    question_bank_extended = QuestionBank(
        que_emb_path="../data/all_questions_embeddings.json"
    )

    test_env_extended_questions = TestClusterVectorEnv(
            question_embed_size=768, 
            pretrained_model_path=params["pretrained_model_path"],
            init_seq_size=params["test_init_seq_size"],
            batch_size=params["test_batch_size"],
            max_steps=params["test_max_steps"],
            student_state_size=params["student_state_size"],
            last_n_steps=params["test_last_n_steps"],
            dataset_folds={1},
            kc_to_que_path=params["kc_to_que_path"],
            kc_emb_path=params["kc_emb_path"],
            reward_scale=params["test_reward_scale"],
            cluster_to_kc_path=params["cluster_to_kc_path"],
            cluster_to_que_path=params["cluster_to_que_path"],
            reward_func=params["test_reward_func"],
            enforce_corpus=True,
            log_wandb=params["test_log_wandb"],
            question_bank=question_bank_extended)
    
    test_vectorized_env = VectorEnvWrapper(test_env_extended_questions)
    test_replay_buffer = VectorReplayBuffer(10*test_vectorized_env.env_num, test_vectorized_env.env_num)
    test_collector = Collector(extended_ques_best_policy, test_vectorized_env, test_replay_buffer, exploration_noise=True)

    test_collector.reset()
    test_replay_buffer.reset()

    with torch_train_mode(extended_ques_best_policy, enabled=False):
        test_result = test_collector.collect(n_episode=params["test_n_episode"])
        if params["use_wandb"]:
            wandb.log({
                "Extended Ques Validated Extended Questions test/return_stats/mean": test_result.returns_stat.mean,
                "Extended Ques Validated Extended Questions test/return_stats/std": test_result.returns_stat.std,
                "Extended Ques Validated Extended Questions test/return_stats/max": test_result.returns_stat.max,
                "Extended Ques Validated Extended Questions test/return_stats/min": test_result.returns_stat.min
            }, commit=True)
        print(f"Extended Ques Validated Extended Test Mean Reward: {test_result.returns_stat.mean}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training Environment Parameters
    parser.add_argument("--train_init_seq_size", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--train_max_steps", type=int, default=10)
    parser.add_argument("--train_max_steps_until_student_change", type=int, default=10)
    parser.add_argument("--train_last_n_steps", type=int, default=10)
    parser.add_argument("--train_reward_scale", type=int, default=1000)
    parser.add_argument("--train_reward_func", type=str, default="step_by_step")
    parser.add_argument("--train_replay_buffer_size", type=int, default=100)
    parser.add_argument("--train_n_episode", type=int, default=512)

    # Validation and Test Environment parameters
    parser.add_argument("--test_init_seq_size", type=int, default=100)
    parser.add_argument("--test_batch_size", type=int, default=2048)
    parser.add_argument("--test_max_steps", type=int, default=10)
    parser.add_argument("--test_last_n_steps", type=int, default=10)
    parser.add_argument("--test_reward_scale", type=int, default=1000)
    parser.add_argument("--test_reward_func", type=str, default="step_by_step")
    parser.add_argument("--test_log_wandb", action="store_true")
    parser.add_argument("--test_n_episode", type=int, default=2048)

    # Global Variables
    parser.add_argument("--pretrained_model_path", type=str, default="../data/pretrained_kt_model.ckpt")
    parser.add_argument("--student_state_size", type=int, default=300)
    parser.add_argument("--kc_to_que_path", type=str, default="../data/XES3G5M/metadata/kc_questions_map.json")
    parser.add_argument("--kc_emb_path", type=str, default="../data/XES3G5M_embeddings/kc_emb.json")
    parser.add_argument("--cluster_to_kc_path", type=str, default="../data/XES3G5M/metadata/kc_clusters.json")
    parser.add_argument("--cluster_to_que_path", type=str, default="../data/XES3G5M/metadata/cluster_to_que_ids_map.json")
    parser.add_argument("--kc_emb_size", type=int, default=768)
    parser.add_argument("--action_size", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--log_dir", type=str, default="./train_ddpg_logs")
    parser.add_argument("--save_dir", type=str, default="./ddpg_saved_models")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default="ddpg")
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--sample_per_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)

    # Critic Parameters:
    parser.add_argument("--critic_model", type=str, default="critic_dkt")
    parser.add_argument("--critic_lr", type=float, default=5e-5)

    # KCCritic Parameters
    parser.add_argument("--critic_hidden_size", type=int, default=300)
    parser.add_argument("--critic_up_projection_size", type=int, default=1200)

    # Actor Parameters
    parser.add_argument("--actor_lr", type=float, default=5e-5)

    # Policy Parameters:
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_exploration_noise_sigma", type=float, default=0.1)
    parser.add_argument("--final_exploration_noise_sigma", type=float, default=0.0001)
    parser.add_argument("--exploration_noise_decay_steps", type=float, default=10000)

    args = parser.parse_args()
    params = vars(args)
    train(params)
    