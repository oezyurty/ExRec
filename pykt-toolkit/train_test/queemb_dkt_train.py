import argparse
from wandb_train import main

# TODO: Add model saved name parameter
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--model_name", type=str, default="embedded_que_dkt")
    parser.add_argument("--emb_type", type=str, default="pretrained")
    parser.add_argument("--emb_path", type=str, default="", help="if not empty string, it will overwrite the emb path in data config.")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--emb_size", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--pred_hidden_size", type=int, default=300)
    parser.add_argument("--up_projection_size", type=int, default=1200)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--pred_dropout", type=float, default=0.0)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weighted_loss", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--train_subset_rate", type=float, default=1.0)

    
    # Important two model configs below
    parser.add_argument('--flag_load_emb', action='store_true', help="Explicitly control if the embeddings will be loaded from path")
    parser.add_argument('--flag_emb_freezed', action='store_true', help="Explicitly control if the embeddings will be freezed or trained")

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--wandb_project_name", type=str, default="", help="if not empty string, it will overwrite the default wandb project name")
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    main(params)