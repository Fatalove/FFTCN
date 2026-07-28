# %%
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter
from tqdm import tqdm
from data.resample import copy_resample, offset_resample
from data.wavelet_torch import cwt


# %%
class Sleep_Data(Dataset):

    def __init__(self, root_path, set_name, seq_len,
                 fold_i, fold_n, ratio=None,
                 balance=False,
                 size=0,
                 seed=0,
                 wave="not", waveshape=(30, 60),
                 ):
        super().__init__()
        self.root_path = root_path
        self.seq_len = seq_len

        # 设置随机数种子
        np.random.seed(seed)

        # 原代码：
        # self.dirs = os.listdir(self.root_path)
        #
        # 修改代码：
        # 论文复现时优先读取已经按受试者划分好的 train/validation/test 子目录。
        # 如果这些子目录不存在，则回退到原仓库的“单层目录 + ratio/k-fold 切分”行为。
        split_dir_names = {
            'train': ('train',),
            'valid': ('valid', 'validation'),
            'test': ('test',),
        }
        split_root = None
        for dirname in split_dir_names.get(set_name, ()):
            candidate = os.path.join(self.root_path, dirname)
            if os.path.isdir(candidate):
                split_root = candidate
                break

        if split_root is not None:
            self.root_path = split_root
            self.dirs = os.listdir(self.root_path)
            self.dirs.sort()
            if size != 0:
                self.dirs = self.dirs[:min(size, len(self.dirs))]
            dirs = self.dirs
        else:
            # 原仓库行为：读取单层目录下的所有文件
            self.dirs = os.listdir(self.root_path)
            # 根据size随机选择部分数据
            if size != 0:
                self.dirs = np.random.choice(self.dirs, size, False)
            else:
                size = len(self.dirs)
                # np.random.shuffle(self.dirs)
            self.dirs.sort()
            # ratio为none表示k折交叉验证
            if ratio == None:
                fold_num = int(np.round(len(self.dirs) / fold_n))
                if fold_i == fold_n - 1:
                    self.valid_dirs = self.dirs[fold_i * fold_num:]
                else:
                    self.valid_dirs = self.dirs[fold_i * fold_num: (fold_i + 1) * fold_num]
                self.train_dirs = list(set(self.dirs) - set(self.valid_dirs))
            # ratio不为none则根据比例划分数据集
            else:
                assert len(ratio) == 3 and ratio[0] + ratio[1] + ratio[2] == 1
                self.train_dirs = self.dirs[:int(size * ratio[0])]
                self.valid_dirs = self.dirs[int(size * ratio[0]):int(size * (ratio[0] + ratio[1]))]
                self.test_dirs = self.dirs[int(size * (ratio[0] + ratio[1])):]

            if set_name == 'train':
                dirs = self.train_dirs
            elif set_name == 'valid':
                dirs = self.valid_dirs
            elif set_name == 'test':
                dirs = self.test_dirs

        self.wave = wave
        self.raw = []
        self.targets = []
        self.spectrum = []
        self.wavelet = cwt(1 / 100, 3000)
        self.cwt_chunk_epochs = int(os.environ.get('FFTCN_CWT_CHUNK_EPOCHS', '128'))
        self.all_counter = Counter()
        sys.stdout.flush()
        pbar = tqdm(dirs, desc=set_name, ncols=0)
        for d in pbar:
            p = os.path.join(self.root_path, d)
            if os.path.exists(p):
                with np.load(p) as f:
                    data = f['x']  # [N, 1, 3000]
                    label = f['y']  # [N]
                if balance:
                    assert self.seq_len == 1
                    data, label = offset_resample(data, label, 300)
                sample_n = len(label) // self.seq_len
                pbar.set_postfix({'name': d, 'samples': sample_n})
                counter = Counter(label)
                self.all_counter += counter
                if self.seq_len != 1:
                    data = data[:sample_n * self.seq_len].reshape(
                        (-1, self.seq_len, *data.shape[1:]))  # [n, seqlen, 1, 3000]
                    label = label[:sample_n * self.seq_len].reshape((-1, self.seq_len, *label.shape[1:]))  # [n, seqlen]
                data = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.long)
                if wave != "only":
                    for r in data:
                        self.raw.append(r)
                if wave != "not":
                    # 原代码：
                    # s = self.wavelet(data, waveshape[1])  # [n, seqlen, 1, f, t]
                    #
                    # 修改代码：
                    # 4GB 显存下，整条记录一次性计算 CWT 会展开出巨大的
                    # [n, seq_len, 1, 30, 3000] 复数中间张量，容易 CUDA OOM。
                    # 这里按“约多少个 30 秒时段”为上限分块计算，输出顺序和形状不变。
                    epochs_per_item = int(np.prod(data.shape[1:-1]))
                    chunk_items = max(1, self.cwt_chunk_epochs // max(1, epochs_per_item))
                    with torch.no_grad():
                        for begin in range(0, len(data), chunk_items):
                            chunk = data[begin:begin + chunk_items]
                            s = self.wavelet(chunk, waveshape[1])  # [n, seqlen, 1, f, t]
                            for si in s:
                                self.spectrum.append(si.cpu().to(torch.float16))
                            del s
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                for l in label:
                    self.targets.append(l)
        # if len(self.raw) != 0:
        #     self.raw = torch.cat(self.raw)
        # if len(self.spectrum) != 0:
        #     self.spectrum = torch.cat(self.spectrum)
        # if len(self.targets) != 0:
        #     self.targets = torch.cat(self.targets)
        print('total:', self.all_counter)

    def __getitem__(self, index):
        if self.wave == "with":
            rtn = [self.raw[index], self.spectrum[index].float()]
        elif self.wave == "only":
            rtn = [self.spectrum[index].float()]
        elif self.wave == "not":
            rtn = [self.raw[index]]
        rtn.append(self.targets[index])
        return rtn

    def __len__(self):
        return len(self.targets)


def Sleep_Loader(root_path, set_name='train', batch_size=16, seq_len=1, fold_i=0, fold_n=10, ratio=None, balance=False,
                 size=0, seed=0, wave="not", waveshape=(30, 60)):
    # 交叉验证时 确保ratio=None
    # 当ratio不为None时 fold_i和fold_n不再有效
    dataset = Sleep_Data(root_path, set_name, seq_len, fold_i, fold_n, ratio, balance, size, seed, wave, waveshape)
    return DataLoader(dataset, batch_size, shuffle=True)


# %%
if __name__ == "__main__":
    root_path = r'E:\wty\data\shhs_cut\c3'
    set_name = 'train'
    loader = Sleep_Loader(root_path, set_name, 128, seq_len=10, ratio=[0.8, 0.1, 0.1], balance=False, wave="only",
                          waveshape=(30, 750))

# %%
