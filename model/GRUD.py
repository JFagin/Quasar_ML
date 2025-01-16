#Joshua Fagin
#Adaptived from: https://github.com/zhiyongc/GRU-D/blob/master/GRUD.py

import torch.nn.functional as F
import torch
import torch.nn as nn
        
class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUD, self).__init__()

        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        
        self.gamma_x_l = nn.Linear(self.delta_size, self.delta_size)
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size)

    
    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        
        delta_x = torch.exp(-F.relu(self.gamma_x_l(delta)))
        delta_h = torch.exp(-F.relu(self.gamma_h_l(delta)))
        
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h
        
        combined = torch.cat((x, h, mask), 1)
        z = torch.sigmoid(self.zl(combined))
        r = torch.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 1)
        h_tilde = torch.tanh(self.hl(combined_r))
        h = (1 - z) * h + z * h_tilde
        
        return h
    
    def forward(self, x, h=None):
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        input_dim = x.size(2)

        # Get the mean of the observed values
        x_mean = torch.sum(x, dim=1) / torch.sum((x != 0.0).type_as(x), dim=1).clamp(min=1e-6)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size).type_as(x).to(x.device)
        else:
            h = h.to(x.device).squeeze(0)

        x_tm1 = torch.zeros(batch_size,input_dim).type_as(x).to(x.device)
        delta_t = torch.zeros(batch_size,input_dim).type_as(x).to(x.device)
        
        time_step = 1.0/seq_len
        obs_mask_t = 0.0
        outputs = None
        for t in range(seq_len):

            x_t = x[:,t] 
            
            if t > 0:
                delta_t = (1.0-obs_mask_t)*delta_t + time_step
            
            obs_mask_t = (x_t != 0.0).type_as(x_t) 

            x_tm1 = torch.where(obs_mask_t>0.0,x_t,x_tm1) 

            h = self.step(x_t, x_tm1, x_mean, h, obs_mask_t, delta_t)
                
            if outputs is None:
                outputs = h.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, h.unsqueeze(1)), 1)

        return outputs,  h.unsqueeze(0)