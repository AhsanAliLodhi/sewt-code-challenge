import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dynamic_ffn import FFNet
from utils import DataPoints
import argparse
from torch.autograd import Variable


def train(X, y,
          learning_rate=1e-3,
          batch_size=32,
          optimizer=torch.optim.Adam,
          loss=torch.nn.MSELoss,
          iters=500):
    data = DataPoints(X, y)
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=0)
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    loss_fn = loss(reduction='mean')

    # Training Loop
    for _iter in range(iters):
        for k, (_data, target) in enumerate(data_loader):
            # Definition of inputs as variables for the net.
            # requires_grad is set False because we do not need to compute
            # the derivative of the inputs.
            _data = Variable(_data, requires_grad=False)
            target = Variable(target, requires_grad=False)
            target = target.float()
            _data = _data.float()

            # Feed forward.
            pred = model(_data)

            # Loss calculation.
            loss = loss_fn(pred, target)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss every 10 iterations.
        if _iter % 10 == 0:
            print('Loss {:.4f} at iter {:d}'.format(loss.item(), _iter))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Train a simple dynamically generated feed forward network on any
    data which looks like X (vector with features) and y (vector with
    targets) where predicting targets is a regression task.
    ''')
    parser.add_argument('-d', '--data_pickle', type=str,
                        help='Path for pandas pickle file containing data.',
                        default="data/train/features.pkl")
    parser.add_argument('-X', '--features', type=str,
                        help='Name of the column containing feature vectors',
                        default="features")
    parser.add_argument('-y', '--target', type=str,
                        help='Name of the column containing target vectors.',
                        default="bbox")
    parser.add_argument('-l', '--learning_rate', type=float,
                        help='Learning rate.',
                        default=1e-3)
    parser.add_argument('-b', '--batch_size', type=float,
                        help='Batch size.',
                        default=512)
    parser.add_argument('-i', '--iterations', type=float,
                        help='Iterations.',
                        default=100)
    parser.add_argument('-f', '--divide_factor', type=float,
                        help='''The factor with which we reduce the size of
                                dynamic FFN. i.e. If input size is 2048 and
                                divide_factor is 2 then the next hidden layer
                                would be of size 1024''',
                        default=3)
    parser.add_argument('-m', '--max_layers', type=float,
                        help='''Maximum number of hidden layers allowed in
                                the dynamic FFN.
                             ''',
                        default=5)
    parser.add_argument('-o', '--out_file', type=str,
                        help='Name by which the model needs to be saved.',
                        default="trained_ffn.pkl")
    args = parser.parse_args()
    data = pd.read_pickle(args.data_pickle)
    X = np.array(data[args.features])
    y = np.array(data[args.target])
    model = FFNet(input_size=len(X[0]),
                  output_size=len(y[0]),
                  divide_factor=args.divide_factor,
                  max_layers=args.max_layers)
    train(X, y,
          learning_rate=args.learning_rate,
          batch_size=args.batch_size,
          optimizer=torch.optim.Adam,
          loss=torch.nn.MSELoss,
          iters=args.iterations)
    torch.save(model.state_dict(), args.out_file)
