import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from torch.optim.lr_scheduler import StepLR

import model.coco_dataset as dataset
from model.coco_cvae import CVAE

# ====================================================
device = torch.device('cuda')
Epoch = 100
Show_Iter = 100
LR = 0.001
save_dir = '/home/karljackab/coco_CVAE_paraphrase/coco_weight'
weight_dir = '/home/karljackab/coco_CVAE_paraphrase/coco_weight/23_2.4385976791381836.pkl'
num2word_path = '/home/karljackab/coco_CVAE_paraphrase/data/coco_num2word.json'
# ====================================================
# ====================================================
TRAIN = True
TEST = False
# ====================================================

def show_result(ground_truth, predict, num2term_map):
    print('Ground Truth Terms: ')
    for i in range(len(ground_truth)):
        term = num2term_map[str(int(ground_truth[i]))]
        print(term, end=' ')
        if term  == 'End':
            cut = i
            break
    print()
    print('Predict Terms: ')
    for i in range(len(predict)):
        val, key = predict[i].max(0)
        key = str(int(key))
        term = num2term_map[key]
        print(term, end=' ')
        if term  == 'End':
            break
    print()

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


train_loader = DataLoader(dataset=dataset.CVAE('train'),
    batch_size=32,
    shuffle=True,
    num_workers=2,
    collate_fn = collat)
test_loader = DataLoader(dataset=dataset.CVAE('test'),
    batch_size=32,
    num_workers=2,
    collate_fn = collat)
    
print(f'train len {len(train_loader)}')
print(f'test len {len(test_loader)}')

model = CVAE(device).to(device)
# model.load_state_dict(torch.load(weight_dir))

# model = torch.load(weight_dir).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
with open(num2word_path, 'r') as f:
    num2term_map = json.load(f)
Loss = nn.CrossEntropyLoss()
kl_ratio = 0
word_drop_ratio = 0.3
kl_ratio_progress = 0.001
recon_ratio = 80

for epoch in range(Epoch):
    if TRAIN:
        train_loss = 0
        train_temp_loss = 0
        KL_temp_loss = 0
        KL_loss = 0
        keep = False
        cnt = 0
        iters = 0
        for idx, data in enumerate(train_loader):
            iters += 1
            termsA, termsB, Dec_input_terms, ground_truth = preprocess(data, device)
            output, KLD_loss = model((termsA, termsB), Dec_input_terms,\
                word_drop_ratio=word_drop_ratio, Share_enc=False)
            
            ground_truth = ground_truth.view(-1).type(torch.LongTensor).to(device)
            bat_size = output.shape[0]
            orig_len = output.shape[1]
            output = output.view(-1, output.shape[2])
            cal_idx = ground_truth!=0
            loss = Loss(output[cal_idx], ground_truth[cal_idx]) * recon_ratio + KLD_loss * kl_ratio
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## KL annealing
            if iters % 100 == 0:
                if not keep:
                    kl_ratio += kl_ratio_progress
                if not keep and kl_ratio > 0.01:
                    keep = True
                    cnt = 0.01/kl_ratio_progress
                if keep:
                    cnt -= 1
                    if cnt <= 0:
                        kl_ratio = 0
                        keep = False

            with torch.no_grad():
                loss = Loss(output[cal_idx], ground_truth[cal_idx]).detach() + KLD_loss.detach()
                KL_temp_loss += KLD_loss.detach()
                train_temp_loss += loss

            if (idx-1) % Show_Iter == 0:
                print(f'Training Epoch {epoch} iter {idx}: loss {(train_temp_loss-KL_temp_loss)/Show_Iter}, KL loss {KL_temp_loss/Show_Iter}')
                train_loss += train_temp_loss
                KL_loss += KL_temp_loss
                train_temp_loss = 0
                KL_temp_loss = 0
                show_result(ground_truth.detach(), output.detach(), num2term_map)
                print('============')
            
            with torch.no_grad():
                output = output.view(bat_size, orig_len, -1)
                ground_truth = ground_truth.view(bat_size, -1)

        train_loss /= len(train_loader)
        KL_loss /= len(train_loader)

        with open(os.path.join(save_dir, 'train_tot_loss'), 'a') as f:
            f.write(str(float(train_loss))+'\n')
        with open(os.path.join(save_dir, 'train_KL_loss'), 'a') as f:
            f.write(str(float(KL_loss))+'\n')
        torch.save(model, os.path.join(save_dir, f'{epoch}_{train_loss}.pkl'))

    if TEST:
        for idx, data in enumerate(test_loader):
            with torch.no_grad():
                termsA, termsB, Dec_input_terms, ground_truth\
                    = preprocess(data, device)
                output = model.inference((termsA, termsB), Dec_input_terms\
                    , Share_enc=False)

                if idx % 10 == 0:
                    print(f'Testing Epoch {epoch} iter {idx}:')
                    ground_truth = ground_truth.view(-1).type(torch.LongTensor).to(device)
                    output = output.view(-1, output.shape[2])
                    show_result(ground_truth.detach(), output.detach(), num2term_map)
                    print('============')