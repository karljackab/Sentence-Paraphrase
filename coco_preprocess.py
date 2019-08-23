import json
import os
import re

json_folder = '/home/karljackab/coco_CVAE_paraphrase/coco'
modes = ['train', 'val']
store_fold = '/home/karljackab/coco_CVAE_paraphrase/data'

def clean_str(string):
    '''
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    '''
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?:;.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r'\W+', ' ', string)
    string = string.lower()
    return string.strip()

word_to_num = {' ':0, 'Start':1, 'End':2, 'UNK':3}
num_to_word = {0:' ', 1:'Start', 2:'End', 3:'UNK'}
idx = 4
for mode in modes:
    with open(os.path.join(json_folder, f'captions_{mode}2014.json')) as f:
        data = json.load(f)
    li = {}
    for anno in data['annotations']:
        if anno['image_id'] not in li:
            li[anno['image_id']] = []
        cap = clean_str(anno['caption'])
        cap = cap.split(' ')
        for word in cap:
            if word not in word_to_num:
                word_to_num[word] = idx
                num_to_word[idx] = word
                idx += 1
        li[anno['image_id']].append(cap)
    res = []
    for img_id in li:
        res.append((li[img_id][0], li[img_id][1]))
        res.append((li[img_id][2], li[img_id][3]))
    with open(os.path.join(store_fold, f'coco_{mode}.json'), 'w') as f:
        json.dump(res, f)

print(idx)
print(len(word_to_num))
print(len(num_to_word))
with open(os.path.join(store_fold, f'coco_word2num.json'), 'w') as f:
        json.dump(word_to_num, f)
with open(os.path.join(store_fold, f'coco_num2word.json'), 'w') as f:
        json.dump(num_to_word, f)