import json
import os
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class CVAE(Dataset):
    def __init__(self, mode, base_path = '/home/karljackab/coco_CVAE_paraphrase/data', with_img = False):
        super().__init__()
        self.with_img = with_img
        if mode == 'train':
            with open(os.path.join(base_path, 'coco_train.json')) as f:
                self.data = json.load(f)
        elif mode == 'test':
            with open(os.path.join(base_path, 'coco_val.json')) as f:
                self.data = json.load(f)
        
        with open(os.path.join(base_path, 'coco_word2num.json')) as f:
            self.word2num_map = json.load(f)
            self.word_len = len(self.word2num_map)
        
        self.S_onehot = torch.zeros(1, self.word_len)\
                .scatter_(1, torch.tensor([[self.word2num_map['Start']]]), 1)
        self.E_onehot = torch.zeros(1, self.word_len)\
                .scatter_(1, torch.tensor([[self.word2num_map['End']]]), 1)

        if mode=='train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
    def __getitem__(self, idx):
        data = self.data[idx]

        words = []
        word_list = []
        target_idx = np.random.randint(2, size=1)[0]
        for idx, package in enumerate(data):
            fill = []
            for word in package:
                fill.append([self.word2num_map[word]])
            pack_len = len(package)
            if pack_len != 0:
                word_set = torch.zeros(pack_len, self.word_len)\
                        .scatter_(1, torch.tensor(fill), 1)
                words.append(word_set)
            else:
                word_set = torch.tensor([])
                words.append(torch.zeros(1, self.word_len))

            if idx == target_idx:
                B_len = word_set.shape[0]
                Dec_input_words = torch.zeros((B_len+1, self.word_len))

                Dec_input_words[0, :] = self.S_onehot
                if B_len != 0:
                    Dec_input_words[1:1+B_len, :] = word_set

                ground_truth = torch.zeros((B_len+1))
                for i, word in enumerate(package):
                    ground_truth[i] = self.word2num_map[word]
                ground_truth[B_len] = self.word2num_map['End']

        return words[1-target_idx], words[target_idx], Dec_input_words, ground_truth

    def __len__(self):
        return len(self.data)
