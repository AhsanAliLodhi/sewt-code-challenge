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
from sklearn.metrics import mean_squared_error


def evaluate(df, feature_col, batch_size=32):
    X = df[feature_col]
    df["predicted_bbox"] = None
    data = DataPoints(X, extras=df["image_name"])
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=0)
    # Test Loop
    for k, (_data, image_names) in enumerate(data_loader):
        # Definition of inputs as variables for the net.
        # requires_grad is set False because we do not need to compute
        # the derivative of the inputs.
        _data = Variable(_data, requires_grad=False)
        _data = _data.float()

        # Feed forward.
        pred = model(_data)
        pred = (pred.data).cpu().numpy()
        for j, image_name in enumerate(image_names):
            idx = df.index[df["image_name"] == image_name][0]
            df.at[idx, "predicted_bbox"] = pred[j]
    return df


def eval_mse(df, real, preicted):
    if real not in df.columns or preicted not in df.columns:
        return "Can't Evaluate, prediction or targets missing."
    # convert any tensors to numpy arrays
    df[real] = [item.data.numpy() if type(item) == torch.Tensor
                else item for item in df[real]]
    df[preicted] = [item.data.numpy() if type(item) == torch.Tensor
                    else item for item in df[preicted]]
    return mean_squared_error(np.stack(df[real]), np.stack(df[preicted]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Train a simple dynamically generated feed forward network on any
    data which looks like X (vector with features) and y (vector with
    targets) where predicting targets is a regression task.
    ''')
    parser.add_argument('-d', '--data_pickle', type=str,
                        help='Path for pandas pickle file containing data.',
                        default="data/test/features.pkl")
    parser.add_argument('-X', '--features', type=str,
                        help='Name of the column containing feature vectors',
                        default="features")
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Batch size.',
                        default=512)
    parser.add_argument('-f', '--divide_factor', type=int,
                        help='''The factor with which we reduce the size of
                                dynamic FFN. i.e. If input size is 2048 and
                                divide_factor is 2 then the next hidden layer
                                would be of size 1024''',
                        default=3)
    parser.add_argument('-m', '--max_layers', type=int,
                        help='''Maximum number of hidden layers allowed in
                                the dynamic FFN.
                             ''',
                        default=5)
    parser.add_argument('-o', '--output_size', type=int,
                        help='size of final layer.',
                        default=4)
    args = parser.parse_args()
    df = pd.read_pickle(args.data_pickle)
    model = FFNet(input_size=len(df[args.features][0]),
                  output_size=args.output_size,
                  divide_factor=args.divide_factor,
                  max_layers=args.max_layers)
    model.eval()
    df = evaluate(df, args.features, batch_size=args.batch_size)
    print(" MSE Evaluation : ", eval_mse(df, "bbox", "predicted_bbox"))
    df.to_pickle(args.data_pickle)
