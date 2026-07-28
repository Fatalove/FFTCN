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
from models.merge.model import MergeModel
import datetime

# %%

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=str(REPOSITORY_ROOT / r'datasets\sleep-edf-153-processed-v1'))
    parser.add_argument('--save-path', default=str(REPOSITORY_ROOT / r'source_reproduction\full_outputs'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--lamb', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', type=int, default=0)
    parser.add_argument('--seq-len', type=int, default=50)
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
    # 路径从命令行传入，默认读取本项目已预处理的 Sleep-EDF-153 split 目录。
    data_path = args.data_path
    save_path = args.save_path
    model_name = 'FFTCN'

    raw_net_path = os.path.join(save_path, 'RawModel', '1D-CNN', 'network.pth')
    wavelet_net_path = os.path.join(save_path, 'WaveletModel', '2D-CNN', 'network.pth')

    model = MergeModel(device, save_path, raw_net_path, wavelet_net_path, name=model_name)

    ratio = [0.8, 0.1, 0.1]
    seed = args.seed
    size = args.size
    waveshape = (30, 60)

    print('=========================================finetune=========================================')
    # 原代码：
    # batch_size = 32
    # seq_len = 50
    # n_epoch = 50
    # learn_rate = 1e-5
    # lamb = 1e-2
    # gamma = 0.95
    # alpha = 0
    #
    # 修改代码：
    # 默认值保持原仓库设置，但允许命令行覆盖，便于本地正式复现和短时验证。
    batch_size = args.batch_size
    seq_len = args.seq_len
    n_epoch = args.epochs
    learn_rate = args.learning_rate
    lamb = args.lamb
    gamma = args.gamma
    alpha = args.alpha
    train_loader = Sleep_Loader(data_path, 'train', batch_size, seq_len, ratio=ratio, seed=seed, wave='with',
                                waveshape=waveshape, size=size)
    valid_loader = Sleep_Loader(data_path, 'valid', batch_size, seq_len, ratio=ratio, seed=seed, wave='with',
                                waveshape=waveshape, size=size)
    model.finetune(train_loader, valid_loader, n_epoch, learn_rate, lamb, gamma, alpha, True)
    model.save()

    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(f'time cost: {time_cost}')

    # 测试阶段：
    # 原仓库没有单独写 test.py，而是在完整 FFTCN 微调结束后，
    # 直接用 test split 分别评估 best checkpoint 和 final checkpoint。
    # 这一步只做评估，不再更新模型参数。
    test_loader = Sleep_Loader(data_path, 'test', batch_size, seq_len, ratio=ratio, seed=seed, wave='with',
                               waveshape=waveshape, size=size)
    print('best:')
    model.test(test_loader, 'best_network.pth')
    print('final:')
    model.test(test_loader)

# %%
