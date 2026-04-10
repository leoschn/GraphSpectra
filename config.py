import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_inter', type=int, default=100)
    parser.add_argument('--eval_inter', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset_train', type=str, default='dataset/traintest_hcd.hdf5')
    parser.add_argument('--dataset_val', type=str, default='dataset/holdout_hcd.hdf5')
    parser.add_argument('--dataset_test', type=str, default='dataset/holdout_hcd.hdf5')
    args = parser.parse_args()

    return args
