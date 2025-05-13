#!/bin/bash
#SBATCH --account=es_sachan
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12288m
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=12288
#SBATCH --open-mode=truncate
#SBATCH --output=./.outs/new_calib_test.out
#SBATCH --error=./.errs/new_calib_test.err

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda

python3 calibration_train.py \
    --emb_path=/cluster/project/sachan/talmac/kt_rl/data/XES3G5M_embeddings/qid2content_sol_avg_emb.json \
    --kc_emb_path=/cluster/project/sachan/talmac/kt_rl/data/XES3G5M_embeddings/kc_emb.json \
    --pretrained_model_path=/cluster/project/sachan/talmac/KCQRL/pykt-toolkit/train_test/saved_model/8068901d-4c94-46fb-84d8-644430ef8e0d/pretrained_model.ckpt \
    --kc_to_questions_path=/cluster/project/sachan/talmac/kt_rl/data/XES3G5M/metadata/kc_questions_map.json \
    --clusters_to_kcs_path=/cluster/project/sachan/talmac/kt_rl/data/XES3G5M/metadata/kc_clusters_optics_xi_mins2.json \
    --clusters_to_qids_path=/cluster/project/sachan/talmac/kt_rl/data/XES3G5M/metadata/cluster_to_que_ids_map.json \
    --flag_joint_train \
    --flag_lstm_frozen \
    --flag_use_cluster \
    --learning_rate=0.00005 \
    --batch_size=128 \
    --train_num_q_pred=10 \
    --valid_num_q_pred=20 \
    --flag_emb_freezed --flag_load_emb \
    --wandb_project_name=dummy_calibration
