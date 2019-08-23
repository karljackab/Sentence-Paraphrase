import torch
import torch.nn as nn
import random
import json
import torch.nn.functional as F

import model.config as conf
from model.coco_encoder import Encoder

class CVAE(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.word_leng = 27312
        self.word_embedding = nn.Embedding(self.word_leng, conf.term_embed)
    
        self.encoder = Encoder(self.device)
        
        ## Decoding
        self.dec_encoder = nn.LSTM(input_size=conf.term_embed,
            hidden_size=conf.latent_vec,
            num_layers=1,
            batch_first=True)
        self.dec_decoder = nn.LSTM(input_size=conf.term_embed+conf.z_dim,
            hidden_size=conf.latent_vec,
            num_layers=conf.decoder_layers,
            batch_first=True)
        self.gen_c_init = nn.Linear(conf.latent_vec*2, conf.decoder_layers * conf.latent_vec)
        self.gen_h_init = nn.Linear(conf.latent_vec*2, conf.decoder_layers * conf.latent_vec)
        self.fc = nn.Linear(conf.latent_vec, self.word_leng)

        with open('data/coco_word2num.json', 'r') as f:
            self.term2num = json.load(f)
        with open('data/coco_num2word.json', 'r') as f:
            self.num2word = json.load(f)

    def embed_term(self, term_set):
        result = torch.zeros(term_set.shape[0], term_set.shape[1], 300).to(self.device)
        for bat, data in enumerate(term_set):
            _, key = data.max(1)
            key = key.tolist()
            words = [self.num2word[str(i)] for i in key]
            embed = self.word_embedding[words]
            result[bat] = embed
        return result

    def inference(self, terms, target, Share_enc=False):
        ## Term Embedding
        ## batch_size*k*300
        terms_embedA = self.word_embedding(terms[0].max(-1)[-1])
        terms_embedB = self.word_embedding(terms[1].max(-1)[-1])
        target_embed = self.word_embedding(target.max(-1)[-1])

        batch_size = terms_embedA.shape[0]
        term_dim_len = terms_embedA.shape[2]

        input_emb_A = terms_embedA
        input_emb_B = terms_embedB

        ## Encoding side
        _, KLD_loss, b_init = self.encoder(input_emb_A, input_emb_B)
        z = torch.randn((batch_size, conf.z_dim)).unsqueeze(1).to(self.device)

        ## Decoding side
        if Share_enc:
            bh, bc = b_init
        else:
            _, (bh, bc) = self.dec_encoder(input_emb_A)

        next_input = torch.cat((target_embed[:,0,:].unsqueeze(1), z), dim=2)
        END = False
        max_decode_len = target_embed.shape[1]
        for time in range(max_decode_len):
            hid_out, (bh, bc) = self.dec_decoder(next_input, (bh, bc))
            hid_out = hid_out.view(batch_size, -1)
            out = self.fc(hid_out)

            if time==0:
                result = out.unsqueeze(1)
            else:
                result = torch.cat((result, out.unsqueeze(1)), dim=1)

            ## generate next input and teacher forcing
            _, max_idx = out.max(dim=1)
            max_idx = max_idx.view(-1, 1).to(self.device)
            next_input = self.word_embedding(max_idx).view(batch_size, 1, -1)
            next_input = torch.cat((next_input, z), dim=2)

        return result

    def forward(self, terms, target, word_drop_ratio = 0.3, Share_enc=False):

        ## Term Embedding
        ## batch_size*k*300
        terms_embedA = self.word_embedding(terms[0].max(-1)[-1])
        terms_embedB = self.word_embedding(terms[1].max(-1)[-1])
        target_embed = self.word_embedding(target.max(-1)[-1])

        batch_size = terms_embedA.shape[0]
        term_dim_len = terms_embedA.shape[2]

        input_emb_A = terms_embedA
        input_emb_B = terms_embedB

        ## Encoding side
        z, KLD_loss, b_init = self.encoder(input_emb_A, input_emb_B)
        z = torch.randn((batch_size, conf.z_dim)).to(self.device)

        ## Decoding side
        if Share_enc:
            bh, bc = b_init
        else:
            _, (bh, bc) = self.dec_encoder(input_emb_A)

        words_len = target_embed.shape[1]
        z = z.unsqueeze(1).repeat(1, words_len, 1)
        dec_input = torch.cat((target_embed, z), dim=2)
        for bat in range(dec_input.shape[0]):
            word_dropout = random.random() < word_drop_ratio
            if word_dropout:
                dec_input[bat, :, :conf.term_embed] = torch.zeros((1, words_len, conf.term_embed))

        hid_out, (bh, bc) = self.dec_decoder(dec_input, (bh, bc))
        out = self.fc(hid_out)

        return out, KLD_loss