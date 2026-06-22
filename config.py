import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_steps', type=int, default=2500)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--eval_inter', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--save_path', type=str, default='model/model.pt')
    parser.add_argument('--root_train', type=str, default='/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/processed_dataset/hierarchical_dataset/processed_graphs_train_hcd_hierarchical_dummy')
    parser.add_argument('--root_val', type=str, default='/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/processed_dataset/hierarchical_dataset/processed_graphs_val_hcd_hierarchical_dummy')
    parser.add_argument('--root_test', type=str, default='/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/processed_dataset/hierarchical_dataset/processed_graphs_holdout_hcd_hierarchical_dummy')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--model_type', type=str, default='GAT')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_timesteps', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.)
    args = parser.parse_args()

    return args

