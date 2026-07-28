# %%
import gc
import os
import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# 原代码：直接运行本文件时没有把仓库根目录加入 sys.path，可能找不到 data/models 顶层包。
# 修改代码：根据当前文件位置定位仓库根目录，保证可从项目根目录直接执行 trainer。
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.loader import Sleep_Loader
from models.raw.model import RawModel
import datetime

# %%

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=str(REPOSITORY_ROOT / r'datasets\sleep-edf-153-processed-v1'))
    parser.add_argument('--save-path', default=str(REPOSITORY_ROOT / r'source_reproduction\full_outputs'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', type=int, default=0)
    parser.add_argument('--no-balance', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.datetime.now()
    device = torch.device("cuda:0" if args.device == 'auto' and torch.cuda.is_available() else ("cpu" if args.device == 'auto' else args.device))
    # 原代码：
    # data_path = r'D:\BJM\testdata'
    # save_path = r'D:\BJM\test_outputs'
    #
    # 修改代码：
    # 路径从命令行传入，默认指向本项目已经预处理好的 Sleep-EDF-153 数据和输出目录。
    data_path = args.data_path
    save_path = args.save_path
    model_name = '1D-CNN'

    ratio = [0.8, 0.1, 0.1]
    seed = args.seed

    model = RawModel(device, save_path, name=model_name)
    print('=========================================pretrain=========================================')
    # 原代码：
    # batch_size = 128
    # n_epoch = 20
    # learn_rate = 1e-5
    # gamma = 0.95
    #
    # 修改代码：
    # 默认值保持原仓库设置，但允许用命令行做短时验证或正式复现配置。
    batch_size = args.batch_size
    n_epoch = args.epochs
    learn_rate = args.learning_rate
    gamma = args.gamma
    train_loader = Sleep_Loader(data_path, 'train', batch_size, ratio=ratio, seed=seed, balance=not args.no_balance, size=args.size)
    valid_loader = Sleep_Loader(data_path, 'valid', batch_size, ratio=ratio, seed=seed, balance=False, size=args.size)
    model.pretrain(train_loader, valid_loader, n_epoch, learn_rate, gamma)
    model.save()

    end_time = datetime.datetime.now()
    time_cost = end_time-start_time
    print(f'time cost: {time_cost}')

# %%