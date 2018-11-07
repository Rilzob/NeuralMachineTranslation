# encoding:utf-8

# @Author: Rilzob
# @Time: 2018/11/6 下午4:46

import argparse

def get_args():
    # Basics
    parser = argparse.ArgumentParser()

    # Data file
    parser.add_argument('--train_file',
                        type=str,
                        default=None,
                        help='Training file')

    parser.add_argument('--dev_file',
                        type=str,
                        default=None,
                        help='Development file')

    # Optimization details
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size')

    parser.add_argument('--num_epoches',
                        type=int,
                        default=100,
                        help='Number of epoches')

    parser.add_argument('--learning_rate', 'lr',
                        type=float,
                        default=0.1,
                        help='Learning rate for Adam')
