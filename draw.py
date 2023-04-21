import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd

import os

def plot(args):
    path = os.path.join("experiment", args.exp_name, "loss.csv")
    df = pd.read_csv(path)
    plt.plot(df["Epoch"], df["training loss"], label="training loss")
    plt.plot(df["Epoch"], df["validation loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.ylim(0, args.ylim)
    plt.legend()
    plt.savefig(os.path.join("experiment", args.exp_name, "loss.png"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Quantization')

    parser.add_argument('--exp_name', type= str, help='Experiment name')
    parser.add_argument('--ylim', type= float, help='Y limits')

    args = parser.parse_args()
    plot(args)
