import torch
import torch.nn as nn
import os

import model.config as conf

class Encoder(nn.Module):
    def __init__(self, device, with_img = True):
        super().__init__()
        self.enc_encoder = nn.LSTM(input_size=conf.term_embed,
            hidden_size=conf.latent_vec,
            num_layers=1,
            batch_first=True)
        self.b_encoder = nn.LSTM(input_size=conf.term_embed,
            hidden_size=conf.latent_vec,
            num_layers=1,
            batch_first=True)

        self.gen_c_init = nn.Linear(conf.latent_vec*2, conf.latent_vec)
        self.gen_h_init = nn.Linear(conf.latent_vec*2, conf.latent_vec)
        
        self.gen_mu = nn.Linear(conf.latent_vec*2, conf.z_dim)
        self.gen_logvar = nn.Linear(conf.latent_vec*2, conf.z_dim)
        self.device = device
    
    def forward(self, input_emb_A, input_emb_B):
        batch_size = input_emb_A.shape[0]
        _, (h_a, c_a) = self.enc_encoder(input_emb_A)
        _, (h_b, c_b) = self.b_encoder(input_emb_B, (h_a, c_a))
        
        h_b = h_b.permute(1,0,2).contiguous().view(batch_size, -1)
        c_b = c_b.permute(1,0,2).contiguous().view(batch_size, -1)
        hidden = torch.cat((h_b, c_b), 1)
        mu = self.gen_mu(hidden)
        logvar = self.gen_logvar(hidden)

        std = torch.exp(0.5 * logvar)
        z = torch.randn((1, conf.z_dim)).to(self.device)
        z = z*std + mu

        kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1))\
                .mean().squeeze()

        return z, kld, (h_a, c_a)