import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_inter', type=int, default=100)
    parser.add_argument('--eval_inter', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset_train', type=str, default='dataset/train_hcd.hdf5')
    parser.add_argument('--dataset_val', type=str, default='dataset/val_hcd.hdf5')
    parser.add_argument('--dataset_test', type=str, default='dataset/holdout_hcd.hdf5')
    parser.add_argument('--root_train', type=str, default='dataset/train_hcd_prec.pt')
    parser.add_argument('--root_val', type=str, default='dataset/val_hcd_prec.pt')
    parser.add_argument('--root_test', type=str, default='dataset/holdout_hcd_prec.pt')
    args = parser.parse_args()

    return args
