from exercise_recommender.envs.discrete_cluster_env import DiscreteClusterEnv
from exercise_recommender.envs.discrete_test_cluster_env import DiscreteTestClusterEnv
from exercise_recommender.envs.vector_env_wrapper import VectorEnvWrapper
from exercise_recommender.utils.question_bank import QuestionBank
from exercise_recommender.utils.data import get_history_generator
from exercise_recommender.wrappers.rainbow_wrapper import RainbowPolicyWrapper
import torch
from exercise_recommender.agents.c51_actor import C51Actor
import gymnasium as gym

device = "cuda" if torch.cuda.is_available() else "cpu"

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.torch_utils import policy_within_training_step, torch_train_mode

import wandb
import uuid
import os

import argparse

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
            tags=["rainbow"],
            config=params
        )

    old_question_bank = QuestionBank(
        que_emb_path="../data/XES3G5M_embeddings/qid2content_sol_avg_emb.json"
    )

    old_ques_train_env = DiscreteClusterEnv(
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
            reward_func=params["train_reward_func"],
            question_bank=old_question_bank)

    if params["use_wandb"]:
        wandb.config.update({
            "train_folds": "2-3-4",
            "valid_folds": "0",
            "test_folds": "1"
        })

    old_ques_valid_env = DiscreteTestClusterEnv(
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
            log_wandb=params["test_log_wandb"],
            question_bank=old_question_bank)

    actor = C51Actor(
        student_hidden_size=params["student_state_size"],
        kc_emb_size=params["kc_emb_size"],
        hidden_size=params["hidden_size"],
        num_questions=old_question_bank.num_q,
        num_atoms=params["num_atoms"]
    ).to(device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=params["actor_lr"])

    print(f"Actor Network loaded into: {device}")

    observation_space = old_ques_train_env.observation_space
    action_space = old_ques_train_env.action_space

    policy = RainbowPolicyWrapper(
        model=actor,
        optim=actor_optim,
        observation_space=observation_space,
        action_space=action_space,
        # Parameters specific to Rainbow:
        discount_factor=params["discount_factor"],
        num_atoms=params["num_atoms"],
        v_min=params["v_min"],
        v_max=params["v_max"],
        estimation_step=params["estimation_step"],
        target_update_freq=params["target_update_freq"],
        reward_normalization=params["reward_normalization"],
        is_double=params["is_double"],
        clip_loss_grad=params["clip_loss_grad"]
    )

    train_vectorized_env = VectorEnvWrapper(old_ques_train_env)
    buffer_size = params["train_replay_buffer_size"]
    train_replay_buffer = VectorReplayBuffer(buffer_size*train_vectorized_env.env_num, train_vectorized_env.env_num)
    train_collector = Collector(policy, train_vectorized_env, train_replay_buffer)

    old_ques_valid_vectorized_env = VectorEnvWrapper(old_ques_valid_env)
    old_ques_valid_replay_buffer = VectorReplayBuffer(1*old_ques_valid_vectorized_env.env_num, old_ques_valid_vectorized_env.env_num)
    old_ques_valid_collector = Collector(policy, old_ques_valid_vectorized_env, old_ques_valid_replay_buffer)

    train_collector.reset()
    old_ques_valid_collector.reset()

    train_replay_buffer.reset()
    old_ques_valid_collector.reset()

    old_ques_best_mean_reward = float('-inf')
    old_ques_std_of_best_mean_reward = 0
    old_ques_min_of_best_mean_reward = 0
    old_ques_max_of_best_mean_reward = 0

    with torch_train_mode(policy, enabled=False):
        old_ques_evaluation_result = old_ques_valid_collector.collect(n_episode=params["test_n_episode"])
        if params["use_wandb"]:
            wandb.log({
                "old_ques_valid/return_stats/mean": old_ques_evaluation_result.returns_stat.mean,
                "old_ques_valid/return_stats/std": old_ques_evaluation_result.returns_stat.std,
                "old_ques_valid/return_stats/max": old_ques_evaluation_result.returns_stat.max,
                "old_ques_valid/return_stats/min": old_ques_evaluation_result.returns_stat.min
            }, commit=True)
        
        old_ques_best_mean_reward = old_ques_evaluation_result.returns_stat.mean
        old_ques_std_of_best_mean_reward = old_ques_evaluation_result.returns_stat.std
        old_ques_min_of_best_mean_reward = old_ques_evaluation_result.returns_stat.min
        old_ques_max_of_best_mean_reward = old_ques_evaluation_result.returns_stat.max
        print(f"Old Ques Starting Mean Reward: {old_ques_best_mean_reward}")

    n_epoch = params["n_epoch"]
    sample_per_epoch = params["sample_per_epoch"]
    batch_size = params["batch_size"]

    for i in range(n_epoch):
        with policy_within_training_step(policy):
            train_res = train_collector.collect(n_episode=params["train_n_episode"])
            if params["use_wandb"]:
                wandb.log({
                    "old_ques_train/return_stats/mean": train_res.returns_stat.mean,
                    "old_ques_train/return_stats/std": train_res.returns_stat.std,
                    "old_ques_train/return_stats/max": train_res.returns_stat.max,
                    "old_ques_train/return_stats/min": train_res.returns_stat.min,
                }, commit=True)
            for _ in range(sample_per_epoch):
                with torch_train_mode(policy):
                    res = policy.update(sample_size=batch_size, buffer=train_collector.buffer)
                    if params["use_wandb"]:
                        wandb.log({
                            "loss": res.loss,
                        }, commit=True)
        with torch_train_mode(policy, enabled=False):
            old_ques_evaluation_result = old_ques_valid_collector.collect(n_episode=params["test_n_episode"])
            if params["use_wandb"]:
                wandb.log({
                    "old_ques_valid/return_stats/mean": old_ques_evaluation_result.returns_stat.mean,
                    "old_ques_valid/return_stats/std": old_ques_evaluation_result.returns_stat.std,
                    "old_ques_valid/return_stats/max": old_ques_evaluation_result.returns_stat.max,
                    "old_ques_valid/return_stats/min": old_ques_evaluation_result.returns_stat.min
                }, commit=True)

            old_ques_mean_rew = old_ques_evaluation_result.returns_stat.mean
            if old_ques_mean_rew > old_ques_best_mean_reward + 1e-4:
                old_ques_best_mean_reward = old_ques_mean_rew
                old_ques_std_of_best_mean_reward = old_ques_evaluation_result.returns_stat.std
                old_ques_min_of_best_mean_reward = old_ques_evaluation_result.returns_stat.min
                old_ques_max_of_best_mean_reward = old_ques_evaluation_result.returns_stat.max
                torch.save(policy.state_dict(), os.path.join(ckpt_path, "old_ques_pretrained_model.ckpt"))
                print(f"Old Ques New Best Mean Reward: {old_ques_best_mean_reward}")

    if params["use_wandb"]:
        wandb.log({
            "Final Old Ques Validation Mean Reward": old_ques_best_mean_reward,
            "Final Old Ques Validation Reward STD": old_ques_std_of_best_mean_reward,
            "Final Old Ques Validation Reward Max": old_ques_max_of_best_mean_reward,
            "Final Old Ques Validation Reward Min": old_ques_min_of_best_mean_reward,
        }, commit=True)

    # Remove valid_env and Question Bank for memory
    del old_question_bank
    del old_ques_valid_env
    del old_ques_valid_vectorized_env
    del old_ques_valid_replay_buffer
    del old_ques_valid_collector
    del policy
    del old_ques_train_env
    del train_vectorized_env
    del train_replay_buffer
    del train_collector

    question_bank_old = QuestionBank(
        que_emb_path="../data/XES3G5M_embeddings/qid2content_sol_avg_emb.json"
    )

    test_env_old_questions = DiscreteTestClusterEnv(
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
            log_wandb=params["test_log_wandb"],
            question_bank=question_bank_old)
    
    old_ques_best_policy = RainbowPolicyWrapper(
        model=actor,
        optim=actor_optim,
        observation_space=observation_space,
        action_space=action_space,
        # Parameters specific to Rainbow:
        discount_factor=params["discount_factor"],
        num_atoms=params["num_atoms"],
        v_min=params["v_min"],
        v_max=params["v_max"],
        estimation_step=params["estimation_step"],
        target_update_freq=params["target_update_freq"],
        reward_normalization=params["reward_normalization"],
        is_double=params["is_double"],
        clip_loss_grad=params["clip_loss_grad"]
    )
    
    old_ques_best_policy.load_state_dict(torch.load(os.path.join(ckpt_path, "old_ques_pretrained_model.ckpt")))

    test_vectorized_env = VectorEnvWrapper(test_env_old_questions)
    test_replay_buffer = VectorReplayBuffer(10*test_vectorized_env.env_num, test_vectorized_env.env_num)
    test_collector = Collector(old_ques_best_policy, test_vectorized_env, test_replay_buffer)

    test_collector.reset()
    test_replay_buffer.reset()

    with torch_train_mode(old_ques_best_policy, enabled=False):
        test_result = test_collector.collect(n_episode=params["test_n_episode"])
        if params["use_wandb"]:
            wandb.log({
                "Old Ques test/return_stats/mean": test_result.returns_stat.mean,
                "Old Ques test/return_stats/std": test_result.returns_stat.std,
                "Old Ques test/return_stats/max": test_result.returns_stat.max,
                "Old Ques test/return_stats/min": test_result.returns_stat.min
            }, commit=True)
        print(f"Old Ques Test Mean Reward: {test_result.returns_stat.mean}")

    del question_bank_old
    del test_env_old_questions
    del test_vectorized_env
    del test_replay_buffer
    del test_collector

    question_bank_extended = QuestionBank(
        que_emb_path="../data/all_questions_embeddings.json"
    )

    extended_ques_train_env = DiscreteClusterEnv(
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
            reward_func=params["train_reward_func"],
            question_bank=question_bank_extended
    )

    extended_ques_valid_env = DiscreteTestClusterEnv(
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
            log_wandb=params["test_log_wandb"],
            question_bank=question_bank_extended)
    
    observation_space=extended_ques_train_env.observation_space
    action_space=extended_ques_train_env.action_space

    actor = C51Actor(
        student_hidden_size=params["student_state_size"],
        kc_emb_size=params["kc_emb_size"],
        hidden_size=params["hidden_size"],
        num_questions=question_bank_extended.num_q,
        num_atoms=params["num_atoms"]
    ).to(device)
    
    actor_optim = torch.optim.Adam(actor.parameters(), lr=params["actor_lr"])
    
    policy = RainbowPolicyWrapper(
        model=actor,
        optim=actor_optim,
        observation_space=observation_space,
        action_space=action_space,
        # Parameters specific to Rainbow:
        discount_factor=params["discount_factor"],
        num_atoms=params["num_atoms"],
        v_min=params["v_min"],
        v_max=params["v_max"],
        estimation_step=params["estimation_step"],
        target_update_freq=params["target_update_freq"],
        reward_normalization=params["reward_normalization"],
        is_double=params["is_double"],
        clip_loss_grad=params["clip_loss_grad"]
    )

    train_vectorized_env = VectorEnvWrapper(extended_ques_train_env)
    buffer_size = params["train_replay_buffer_size"]
    train_replay_buffer = VectorReplayBuffer(buffer_size*train_vectorized_env.env_num, train_vectorized_env.env_num)
    train_collector = Collector(policy, train_vectorized_env, train_replay_buffer)

    extended_ques_valid_vectorized_env = VectorEnvWrapper(extended_ques_valid_env)
    extended_ques_valid_replay_buffer = VectorReplayBuffer(1*extended_ques_valid_vectorized_env.env_num, extended_ques_valid_vectorized_env.env_num)
    extended_ques_valid_collector = Collector(policy, extended_ques_valid_vectorized_env, extended_ques_valid_replay_buffer)

    train_collector.reset()
    extended_ques_valid_collector.reset()

    train_replay_buffer.reset()
    extended_ques_valid_collector.reset()

    extended_ques_best_mean_reward = float('-inf')
    extended_ques_std_of_best_mean_reward = 0
    extended_ques_min_of_best_mean_reward = 0
    extended_ques_max_of_best_mean_reward = 0

    with torch_train_mode(policy, enabled=False):
        extended_ques_evaluation_result = extended_ques_valid_collector.collect(n_episode=params["test_n_episode"])
        if params["use_wandb"]:
            wandb.log({
                "extended_ques_valid/return_stats/mean": extended_ques_evaluation_result.returns_stat.mean,
                "extended_ques_valid/return_stats/std": extended_ques_evaluation_result.returns_stat.std,
                "extended_ques_valid/return_stats/max": extended_ques_evaluation_result.returns_stat.max,
                "extended_ques_valid/return_stats/min": extended_ques_evaluation_result.returns_stat.min
            }, commit=True)
        
        extended_ques_best_mean_reward = extended_ques_evaluation_result.returns_stat.mean
        extended_ques_std_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.std
        extended_ques_min_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.min
        extended_ques_max_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.max
        print(f"Extended Ques Starting Mean Reward: {extended_ques_best_mean_reward}")

    n_epoch = params["n_epoch"]
    sample_per_epoch = params["sample_per_epoch"]
    batch_size = params["batch_size"]

    for i in range(n_epoch):
        with policy_within_training_step(policy):
            train_res = train_collector.collect(n_episode=params["train_n_episode"])
            if params["use_wandb"]:
                wandb.log({
                    "extended_ques_train/return_stats/mean": train_res.returns_stat.mean,
                    "extended_ques_train/return_stats/std": train_res.returns_stat.std,
                    "extended_ques_train/return_stats/max": train_res.returns_stat.max,
                    "extended_ques_train/return_stats/min": train_res.returns_stat.min,
                }, commit=True)
            for _ in range(sample_per_epoch):
                with torch_train_mode(policy):
                    res = policy.update(sample_size=batch_size, buffer=train_collector.buffer)
                    if params["use_wandb"]:
                        wandb.log({
                            "loss": res.loss,
                        }, commit=True)
        with torch_train_mode(policy, enabled=False):
            extended_ques_evaluation_result = extended_ques_valid_collector.collect(n_episode=params["test_n_episode"])
            if params["use_wandb"]:
                wandb.log({
                    "extended_ques_valid/return_stats/mean": extended_ques_evaluation_result.returns_stat.mean,
                    "extended_ques_valid/return_stats/std": extended_ques_evaluation_result.returns_stat.std,
                    "extended_ques_valid/return_stats/max": extended_ques_evaluation_result.returns_stat.max,
                    "extended_ques_valid/return_stats/min": extended_ques_evaluation_result.returns_stat.min
                }, commit=True)

            extended_ques_mean_rew = extended_ques_evaluation_result.returns_stat.mean
            if extended_ques_mean_rew > extended_ques_best_mean_reward + 1e-4:
                extended_ques_best_mean_reward = extended_ques_mean_rew
                extended_ques_std_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.std
                extended_ques_min_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.min
                extended_ques_max_of_best_mean_reward = extended_ques_evaluation_result.returns_stat.max
                torch.save(policy.state_dict(), os.path.join(ckpt_path, "extended_ques_pretrained_model.ckpt"))
                print(f"Extended Ques New Best Mean Reward: {extended_ques_best_mean_reward}")

    if params["use_wandb"]:
        wandb.log({
            "Final Extended Ques Validation Mean Reward": extended_ques_best_mean_reward,
            "Final Extended Ques Validation Reward STD": extended_ques_std_of_best_mean_reward,
            "Final Extended Ques Validation Reward Max": extended_ques_max_of_best_mean_reward,
            "Final Extended Ques Validation Reward Min": extended_ques_min_of_best_mean_reward,
        }, commit=True)

    del question_bank_extended
    del extended_ques_valid_env
    del extended_ques_valid_vectorized_env
    del extended_ques_valid_replay_buffer
    del extended_ques_valid_collector
    del policy
    del extended_ques_train_env
    del train_vectorized_env
    del train_replay_buffer
    del train_collector

    question_bank_extended = QuestionBank(
        que_emb_path="../data/all_questions_embeddings.json"
    )

    test_env_extended_questions = DiscreteTestClusterEnv(
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
            log_wandb=params["test_log_wandb"],
            question_bank=question_bank_extended)
    
    extended_ques_best_policy = RainbowPolicyWrapper(
        model=actor,
        optim=actor_optim,
        observation_space=observation_space,
        action_space=action_space,
        # Parameters specific to Rainbow:
        discount_factor=params["discount_factor"],
        num_atoms=params["num_atoms"],
        v_min=params["v_min"],
        v_max=params["v_max"],
        estimation_step=params["estimation_step"],
        target_update_freq=params["target_update_freq"],
        reward_normalization=params["reward_normalization"],
        is_double=params["is_double"],
        clip_loss_grad=params["clip_loss_grad"]
    )

    extended_ques_best_policy.load_state_dict(torch.load(os.path.join(ckpt_path, "extended_ques_pretrained_model.ckpt")))

    test_vectorized_env = VectorEnvWrapper(test_env_extended_questions)
    test_replay_buffer = VectorReplayBuffer(10*test_vectorized_env.env_num, test_vectorized_env.env_num)
    test_collector = Collector(extended_ques_best_policy, test_vectorized_env, test_replay_buffer)

    test_collector.reset()
    test_replay_buffer.reset()

    with torch_train_mode(extended_ques_best_policy, enabled=False):
        test_result = test_collector.collect(n_episode=params["test_n_episode"])
        if params["use_wandb"]:
            wandb.log({
                "Extended Ques test/return_stats/mean": test_result.returns_stat.mean,
                "Extended Ques test/return_stats/std": test_result.returns_stat.std,
                "Extended Ques test/return_stats/max": test_result.returns_stat.max,
                "Extended Ques test/return_stats/min": test_result.returns_stat.min
            }, commit=True)
        print(f"Extended Ques Test Mean Reward: {test_result.returns_stat.mean}")

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
    parser.add_argument("--log_dir", type=str, default="./train_rainbow_logs")
    parser.add_argument("--save_dir", type=str, default="./rainbow_saved_models")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default="rainbow")
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--sample_per_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)

    # Actor Parameters
    parser.add_argument("--actor_lr", type=float, default=5e-5)

    # Policy Parameters:
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--num_atoms", type=int, default=17)
    parser.add_argument("--v_min", type=float, default=-1000.0)
    parser.add_argument("--v_max", type=float, default=1000.0)
    parser.add_argument("--estimation_step", type=int, default=1)
    parser.add_argument("--target_update_freq", type=int, default=0)
    parser.add_argument("--reward_normalization", action="store_true")
    parser.add_argument("--is_double", action="store_true")
    parser.add_argument("--clip_loss_grad", action="store_true")

    args = parser.parse_args()
    params = vars(args)
    train(params)
    