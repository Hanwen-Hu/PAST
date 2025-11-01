"""Main script to run the IEST model for training and testing."""

import argparse
import torch

from model import PAST


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    # Dataset Setting
    parser.add_argument('-dataset', type=str, default='MeTr-LA')
    parser.add_argument('-dim', type=int, default=1)
    parser.add_argument('-seq_len', type=int, default=100)
    parser.add_argument('-miss_rate', type=float, default=0.4)
    parser.add_argument('-miss_len', type=int, default=1)
    parser.add_argument('-miss_span', type=int, default=1)
    # Experiment Setting
    parser.add_argument('-cuda_id', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-lr', type=float, default=0.005)
    parser.add_argument('-load', type=bool, default=False)
    # Model Setting
    parser.add_argument('-hidden_dim', type=int, default=128)
    parser.add_argument('-layer_num', type=int, default=3)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-order', type=int, default=1)
    args = parser.parse_args()
    node_dict = {'MeTr-LA': 207, 'PeMS-Bay': 325, 'LargeST-SD': 716}
    args.node_num = node_dict[args.dataset]
    if torch.cuda.is_available():
        args.device = torch.device('cuda', args.cuda_id)
    else:
        args.device = torch.device('cpu')
    return args


def main():
    """Main function to run the IEST model."""
    args = parse_args()
    model = PAST(args)
    model.train()
    model.test()


def visual():
    """Function to visualize the model's predictions."""
    args = parse_args()
    model = PAST(args)
    node = 1  # Specify the node index to visualize
    model.visualize(80, node)


if __name__ == '__main__':
    visual()
