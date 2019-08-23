import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from torch.optim.lr_scheduler import StepLR

import model.coco_dataset as dataset
from model.coco_cvae import CVAE

device = torch.device('cuda')
save_dir = '/home/karljackab/coco_CVAE_paraphrase/demo_res/coco_res.json'
weight_dir = '/home/karljackab/coco_CVAE_paraphrase/coco_weight/23_2.4385976791381836.pkl'
num2word_path = '/home/karljackab/coco_CVAE_paraphrase/data/coco_num2word.json'


def collat(data):
    return data
def preprocess(data, device):
    batch_size = len(data)
    result = []
    ## For terms
    for i in range(3):
        max_len = max([len(sub_data[i]) for sub_data in data])
        res = torch.zeros((batch_size, max_len, data[0][i].shape[1]))
        for idx, sub_data in enumerate(data):
            res[idx, :sub_data[i].shape[0], :] = sub_data[i]
        result.append(res.to(device))
    
    ## For ground truth
    max_len = max([len(sub_data[3]) for sub_data in data])
    res = torch.zeros((batch_size, max_len))
    for idx, sub_data in enumerate(data):
        res[idx, :len(sub_data[3])] = sub_data[3]
    result.append(res.to(device))

    return result

test_loader = DataLoader(dataset=dataset.CVAE('test'),
    batch_size=32,
    num_workers=2,
    collate_fn = collat)

model = CVAE(device).to(device)
model.load_state_dict(torch.load(weight_dir))
with open(num2word_path, 'r') as f:
    num2term_map = json.load(f)

with torch.no_grad():
    test_len = len(test_loader)
    final = []
    for idx, data in enumerate(test_loader):
        print(f'{idx}/{test_len}')
        termsA, termsB, Dec_input_terms, ground_truth\
            = preprocess(data, device)
        output = model.inference((termsA, termsB), Dec_input_terms\
            , Share_enc=False)

        for pred, ground in zip(output, ground_truth):
            gold_terms_set = []
            predict_terms_set = []
            for i in range(len(ground)):
                term = num2term_map[str(int(ground[i]))]
                if term == 'End':
                    break
                gold_terms_set.append(term)
                
            for i in range(len(pred)):
                key = pred[i].max(0)[-1]
                key = str(int(key))
                term = num2term_map[key]
                if term == 'End':
                    break
                predict_terms_set.append(term)
            final.append({
                "Gold": ' '.join(gold_terms_set),
                "Predict": ' '.join(predict_terms_set)
            })

with open(save_dir, 'w') as f:
    json.dump(final, f)