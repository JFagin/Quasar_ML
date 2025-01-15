# Author Joshua Fagin

import torch 
from torch import nn
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint
import torchsde
from model.GRUD import GRUD
import torch.nn.functional as F
import numpy as np
import gc
import logging
from scipy import constants as const
from astropy import constants as const_astropy
from model.TF import GenerateTFModule
import math 

# Just for debugging
import matplotlib.pyplot as plt

# For debugging, can set the anomaly detection to see where the gradients are going to NaN. Will slow down the training a lot.
#torch.autograd.set_detect_anomaly(True)

# See example of SDEs in: https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py
# See example of Recurrent Inferece Machine in: https://github.com/pputzky/invertible_rim/blob/master/irim/rim/rim.py

class RNN_Encoder(nn.Module):
    """
    Function to encode the input time series using a bidirectional RNN including layer normalization and skip connections.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size: int, the dimension of the input data for time series of shape [B, T, input_size]
        hidden_size: int, the dimension of the hidden layers
        output_size: int, the dimension of the output data so that the output is of shape [B, T, output_size]

        return: output tensor with the encoded time series, shape [B, T, output_size]
        """
        super(RNN_Encoder, self).__init__()
        self.rnn_forward1 = GRUD(input_size, hidden_size//2)
        self.rnn_backward1 = GRUD(input_size, hidden_size//2)

        self.norm1 = nn.LayerNorm(hidden_size)

        self.rnn_forward2 = nn.GRU(hidden_size, hidden_size//2, batch_first=True)
        self.rnn_backward2 = nn.GRU(hidden_size, hidden_size//2, batch_first=True)


        self.norm2 = nn.LayerNorm(hidden_size)
        self.rnn_forward3 = nn.GRU(hidden_size, hidden_size//2, batch_first=True)
        self.rnn_backward3 = nn.GRU(hidden_size, hidden_size//2, batch_first=True)
        
        self.norm3 = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = torch.cat((self.rnn_forward1(x)[0], torch.flip(self.rnn_backward1(torch.flip(x, (1,)))[0], (1,))), dim=-1) # [B, T, hidden_size]

        skip = x

        x = self.norm1(x)
        x = torch.cat((self.rnn_forward2(x)[0], torch.flip(self.rnn_backward2(torch.flip(x, (1,)))[0], (1,))), dim=-1) # [B, T, hidden_size]
        x = x + skip

        skip = x

        x = self.norm2(x)
        x = torch.cat((self.rnn_forward3(x)[0], torch.flip(self.rnn_backward3(torch.flip(x, (1,)))[0], (1,))), dim=-1) # [B, T, hidden_size]
        x = x + skip


        x = self.norm3(x)
        x = self.linear(x)

        return x

# Positional Encoding function
def positional_encoding(x):
    """
    Add positional encoding to the input tensor. This is used in the Transformer model.

    x: input tensor, shape [B, T, D]
    return: output tensor with positional encoding added, shape [B, T, D]
    """
    B, T, D = x.shape
    pos = torch.arange(T, device=x.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2, device=x.device) * -(math.log(10000.0) / D))
    
    pos_enc = torch.zeros(T, D, device=x.device)
    pos_enc[:, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, 1::2] = torch.cos(pos * div_term)
    
    pos_enc = pos_enc.unsqueeze(0)
    return x + pos_enc

class Context_From_Transformer(nn.Module):
    """
    Function to get the context from the Transformer model. This is used in the RNN_Transformer model.
    """
    def __init__(self, hidden_size):
        """
        hidden_size: int, the dimension of the hidden layers

        return: output tensor with the context, shape [B, hidden_size]
        """
        super(Context_From_Transformer, self).__init__()
        self.linear = nn.Linear(4*hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)

    
    def forward(self, x):
        return self.linear(torch.cat((self.norm1(x.mean(dim=1)), self.norm2(x.std(dim=1)), self.norm3(x[:,0]), self.norm4(x[:, -1])), dim=1))

class RNN_Transformer(nn.Module):
    """
    Function that combines an RNN with a Transformer model to encode the input time series.
    """
    def __init__(self, input_dim, model_dim, output_dim, num_heads, num_layers, dropout=0.0):
        """
        input_dim: int, the dimension of the input data for time series of shape [B, T, input_dim]
        model_dim: int, the dimension of the hidden layers
        output_dim: int, the dimension of the output data so that the output is of shape [B, T, output_dim]
        num_heads: int, the number of heads in the multi-head attention of the Transformer model
        num_layers: int, the number of layers in the Transformer model
        dropout: float, the dropout rate, should be between 0 and 1, default is 0.0

        return: output tensor with the encoded time series, shape [B, T, output_dim]
        """
        super(RNN_Transformer, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.transformer_model_dim = 2*model_dim
        self.rnn_embedding = RNN_Encoder(input_size=input_dim, hidden_size=model_dim, output_size=self.transformer_model_dim)

        encoder_layers = nn.TransformerEncoderLayer(self.transformer_model_dim, 
                                                    num_heads, 
                                                    dim_feedforward=4*self.transformer_model_dim,
                                                    activation='gelu',
                                                    dropout=dropout, 
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.layer_norm = nn.LayerNorm(2*self.transformer_model_dim)
        self.decoder = nn.Sequential(nn.Linear(2*self.transformer_model_dim, model_dim),
                                    nn.LeakyReLU(),
                                    nn.LayerNorm(model_dim),
                                    nn.Linear(model_dim, output_dim),
                                    )
        
        self.get_context = Context_From_Transformer(output_dim)

    def forward(self, x):

        x = self.rnn_embedding(x)
        skip = x

        x = x * math.sqrt(self.transformer_model_dim)
        x = positional_encoding(x)
        x = self.transformer_encoder(x)

        x = torch.cat((x, skip), dim=-1) # we can skip the transformer if we want

        x = self.layer_norm(x)
        x = self.decoder(x)

        context = self.get_context(x)

        return x, context

class RNN_Projector(nn.Module):
    """
    Function to project the output of the SDE to the desired output dimension.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size: int, the dimension of the input data for time series of shape [B, T, input_size]
        hidden_size: int, the dimension of the hidden layers
        output_size: int, the dimension of the output data so that the output is of shape [B, T, output_size]

        return: output tensor with the projected time series, shape [B, T, output_size]
        """
        super(RNN_Projector, self).__init__()
        self.rnn_forward1 = nn.GRU(input_size, hidden_size//2, batch_first=True)
        self.rnn_backward1 = nn.GRU(input_size, hidden_size//2, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_size)

        self.rnn_forward2 = nn.GRU(hidden_size, hidden_size//2, batch_first=True)
        self.rnn_backward2 = nn.GRU(hidden_size, hidden_size//2, batch_first=True)

        self.linear_skip = nn.Linear(input_size, output_size)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        skip = self.linear_skip(x)

        x = torch.cat((self.rnn_forward1(x)[0], torch.flip(self.rnn_backward1(torch.flip(x, (1,)))[0], (1,))), dim=-1) # [B, T, hidden_size]
        x = self.norm1(x)

        x = torch.cat((self.rnn_forward2(x)[0], torch.flip(self.rnn_backward2(torch.flip(x, (1,)))[0], (1,))), dim=-1) # [B, T, hidden_size]
        x = self.norm2(x)
        x = self.linear(x)

        x = x + skip
        
        return x

# Not used currently
class Linear_projector(nn.Module):
    """
    Function to project the output of the SDE to the desired output dimension.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size: int, the dimension of the input data for time series of shape [B, T, input_size]
        hidden_size: int, the dimension of the hidden layers
        output_size: int, the dimension of the output data so that the output is of shape [B, T, output_size]

        return: output tensor with the projected time series, shape [B, T, output_size]
        """
        super(Linear_projector, self).__init__()
        self.skip = nn.Linear(input_size, output_size)
        self.linear = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.LeakyReLU(),
                                    nn.LayerNorm(hidden_size),
                                    nn.Linear(hidden_size, output_size),
                                    )
    def forward(self, x):
        skip = self.skip(x)
        x = self.linear(x)
        x = x + skip
        return x

class SDE(nn.Module):
    """
    Define the latent SDE model. This is the model that will be used to generate the driving signal of our NN.
    """

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, input_size, latent_size, hidden_size, context_size, output_dim, num_layers_encoder, num_heads,
                 dt=1e-3, dropout_rate=0.0, method="euler", logqp=False, device="cpu"):
        super(SDE, self).__init__()
        """
        input_size : int, dimension of the input data
        latent_size: int, dimension of the latent space
        hidden_size: int, dimension of hidden layers
        context_size: int, dimension of the context vector
        output_dim: int, dimension of the output data
        num_layers_encoder: int, number of RNN layers in the encoder
        num_heads: int, number of heads in the multi-head attention of the Transformer model
        dt: float, the time step size, must be positive
        dropout_rate: float, the dropout rate, should be between 0 and 1, default is 0.0
        method: str, the method to use for the SDE solver, default is "euler"
        logqp: bool, whether to use the logqp method which would sample from the latent space, default is False
        device: str, the device to use, default is "cpu"
        """
        self.latent_size = latent_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.dropout_rate = dropout_rate
        self.method = method

        self.logqp = logqp

        self.encoder = RNN_Transformer(input_dim=input_size, 
                                model_dim=hidden_size,
                                output_dim=context_size,  
                                num_heads=num_heads, 
                                num_layers=num_layers_encoder,
                                dropout=dropout_rate,
                            )

        #self.projector = Linear_projector(latent_size, hidden_size, output_dim)
        self.projector = RNN_Projector(latent_size, hidden_size, output_dim)

        self.f_net = Linear_Skip_new(latent_size + context_size, latent_size, hidden_size, layer_norm_input=True, dropout_rate=self.dropout_rate)
            
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.LeakyReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(self.dropout_rate),

                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(self.dropout_rate),

                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )

        # We only need the h_net if we are using the logqp method to sample from the latent space
        if self.logqp:
            self.h_net = Linear_Skip_new(latent_size, latent_size, hidden_size, layer_norm_input=True, dropout_rate=self.dropout_rate)

    def f(self, t, y):
        """Network that decodes the latent and context
        (posterior drift function)
        Time-inhomogeneous (see paper sec 9.12)

        """
        # ts ~ [T]
        # ctx ~ [T, B, context_size]
        # searchsorted output: if t were inserted into ts, what would the
        # indices have to be to preserve ordering, assuming ts is sorted
        # training time: t is tensor with no size (scalar)
        # inference time: t ~ [num**2, 1] from the meshgrid

        i = min(torch.searchsorted(self.ts, t.min(), right=True), len(self.ts) - 1)

        # training time: y ~ [B, latent_dim]
        # inference time: y ~ [num**2, 1] from the meshgrid
        
        f_in = torch.cat((y, self.ctx[i]), dim=1) 
        
        # Training time (for each time step)
        # t ~ []
        # y ~ [B, latent_dim]
        # f_in ~ [B, latent_dim + context_dim]
        f_out = self.f_net(f_in)
        # f_out ~ [B, latent_dim]
        return f_out

    def g(self, t, y):
        """Network that decodes each time step of the latent
        (diagonal diffusion)

        """
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        g_out = torch.cat(out, dim=1)  # [B, latent_dim]
        return g_out     

    def h(self, t, y):
        """Network that decodes the latent
        (prior drift function)

        """
        if self.logqp:
            return self.h_net(y)
        else:
            return 0.0

    def encode(self, xs, ts):
        # xs ~ [B, T, 2*input_dim]
        self.ctx, context = self.encoder(xs) # [B, T, context_size], [B, context_size]
        self.ctx = self.ctx.transpose(0, 1) # [B, T, context_size] to [T, B, context_size]
        self.ts = ts

        return context
        

    def forward(self, z0):

        if self.logqp:
            zs, log_ratio = torchsde.sdeint(self, z0, self.ts, dt=self.dt, logqp=True, method=self.method)
            log_ratio = log_ratio.mean()
        else:
            zs = torchsde.sdeint(self, z0, self.ts, dt=self.dt, logqp=False, method=self.method)
            log_ratio = 0.0
        zs = zs.transpose(0, 1) # [T, B, latent_dim] to [B, T, latent_dim]
        zs = self.projector(zs)  # project ourput of SDE from [T, B, latent_dim] to [T, B, output_dim]

        return zs, log_ratio

class RNN_Cell(nn.Module):
    """
    Used in the recurrent inference machine procedure to iteratively update the hidden state of the RNN cell.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        """
        input_size: int, the dimension of the input data for [B, input_size]
        hidden_size: int, the dimension of the hidden layers
        output_size: int, the dimension of the output data so that the output is of shape [B, output_size]
        dropout_rate: float, the dropout rate, should be between 0 and 1, default is 0.0

        return: output tensor with the updated hidden state, shape [B, output_size]
        """
        super(RNN_Cell, self).__init__()

        self.rnn_cell = nn.GRUCell(input_size, hidden_size)
        self.linear = nn.Sequential(nn.LayerNorm(hidden_size),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.LeakyReLU(),
                                    nn.LayerNorm(hidden_size),
                                    nn.Dropout(dropout_rate),
                                    nn.Linear(hidden_size, output_size),
                                    )
    def forward(self, x, h):
        h = self.rnn_cell(x, h)
        x = self.linear(x)

        return x, h

class Linear_Skip_new(nn.Module):
    """
    Function defining a MLP with skip connections and layer normalization.
    """
    def __init__(self, input_size, output_size, hidden_size, layer_norm_input=False, dropout_rate=0.0):
        """
        input_size: int, the dimension of the input data for [B, input_size] or [B, T, input_size]
        output_size: int, the dimension of the output data so that the output is of shape [B, output_size]
        hidden_size: int, the dimension of the hidden layers
        layer_norm_input: bool, whether to use layer normalization on the input, default is False
        dropout_rate: float, the dropout rate, should be between 0 and 1, default is 0.0

        return: output tensor with the updated hidden state, shape [B, output_size] or [B, T, output_size]
        """
        super(Linear_Skip_new, self).__init__()

        self.layer_norm_input = layer_norm_input
        if layer_norm_input:
            self.norm_input = nn.LayerNorm(input_size)

        self.linear1 = nn.Sequential(
                                    nn.Linear(input_size, hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout_rate),
                                    )
        
        self.linear2 = nn.Sequential(
                                    nn.LayerNorm(hidden_size),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout_rate),
                                    )
        
        self.linear3 = nn.Sequential(
                                    nn.LayerNorm(hidden_size),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout_rate),
                                    )
        
        self.linear4 = nn.Sequential(
                                    nn.LayerNorm(hidden_size),
                                    nn.Linear(hidden_size, output_size),
                                    )
        

    def forward(self, x):
        if self.layer_norm_input:
            x = self.norm_input(x)
        x = self.linear1(x)
        x = self.linear2(x) + x  # skip connection 
        x = self.linear3(x) + x  # skip connection
        x = self.linear4(x)
        return x

# Currently not used. Could compare the performance to our full model.
class RNN_baseline(nn.Module):
    """
    RNN baseline model to predict the parameters of the light curve. 
    """
    def __init__(self, input_size, hidden_size, num_bands, num_layers, num_Gaussian_parameterization, n_params, dropout_rate=0.0, num_heads=4):
        """
        input_size: int, the dimension of the input data for time series of shape [B, T, input_size]
        hidden_size: int, the dimension of the hidden layers
        num_bands: int, the number of bands in the light curve
        num_layers: int, the number of layers in the RNN model
        num_Gaussian_parameterization: int, the number of Gaussian parameterizations to use
        n_params: int, the number of parameters to predict
        dropout_rate: float, the dropout rate, should be between 0 and 1, default is 0.0
        num_heads: int, the number of heads in the multi-head attention of the Transformer model
        """
        super(RNN_baseline, self).__init__()
        encoder_hidden_size = hidden_size//2 # since we include the flip, we want the hidden size to be half of the original size

        self.encoder_network = RNN_Transformer(input_dim=input_size, 
                                model_dim=hidden_size,
                                output_dim=hidden_size,  
                                num_heads=num_heads, 
                                num_layers=num_layers,
                                dropout=dropout_rate,
                            )

        
        self.num_bands = num_bands
        self.n_params = n_params
        self.out_dim = num_bands
        self.num_Gaussian_parameterization = num_Gaussian_parameterization
        self.num_samples = 50_000
        self.params_mean_net = nn.Sequential(Linear_Skip_new(input_size=hidden_size+2*num_bands,
                                                        output_size=num_Gaussian_parameterization*n_params,
                                                        hidden_size=hidden_size,
                                                        dropout_rate=dropout_rate),
                                            )
        
        self.gaussian_mixture_coefficients_net = nn.Sequential(
                                                        Linear_Skip_new(input_size=hidden_size+2*num_bands+num_Gaussian_parameterization*n_params,
                                                                    output_size=num_Gaussian_parameterization,
                                                                    hidden_size=hidden_size,
                                                                    dropout_rate=dropout_rate),
                                                        nn.Softmax(dim=1),
                                                    )
    
        num_output = num_Gaussian_parameterization*self.n_params*(self.n_params+1)//2  
        self.L_params_net = Linear_Skip_new(input_size=hidden_size+2*num_bands+num_Gaussian_parameterization*n_params+num_Gaussian_parameterization,
                                        output_size=num_output,
                                        hidden_size=hidden_size,
                                        dropout_rate=dropout_rate)

        self.max_magnitude = 27.0
        self.max_magnitude_std = 1.0
        self.epsilon = 1e-6

    def get_weighted_mean_std(self, xs):
        """
        Gets the weighted mean and standard deviation of the predicted light curves

        xs ~ [B, T, 2*num_bands], we have num_bands and uncertainty for each band

        returns: mean ~ [B, 1, num_bands], std ~ [B, 1, num_bands], mean_mask ~ [B, 1, num_bands]
        """

        mask = (xs[:,:,:self.out_dim] != 0.0).type_as(xs)

        # get weighted mean of the light curve
        # the mean is weighted by the inverse of the variance
        # https://en.wikipedia.org/wiki/Inverse-variance_weighting
        mean = torch.sum(xs[:,:,:self.out_dim]/(xs[:,:,self.out_dim:]+self.epsilon)**2,dim=1)/(torch.sum(mask/(xs[:,:,self.out_dim:]+self.epsilon)**2,dim=1)+self.epsilon)

        #now get the std masking all the zeros of xs
        mean_diff = mask*(xs[:,:,:self.out_dim] - mean.unsqueeze(1))**2 # mask out the zeros again
        std = torch.sqrt(torch.sum(mean_diff/(xs[:,:,self.out_dim:]+self.epsilon)**2+self.epsilon,dim=1)/(torch.sum(mask/(xs[:,:,self.out_dim:]+self.epsilon)**2,dim=1)+self.epsilon))

        mean = mean.unsqueeze(1) # mean ~ [B, 1, num_bands]
        std = std.unsqueeze(1)   # std ~ [B, 1, num_bands]

        # Get the mask where the mean is less than self.min_magnitude or greater than self.max_magnitude. 
        # Nominally the mean and std should already be masked though from not being observed in the first place in which case they will be zero for that band. 
        #mean_mask = (mean > self.min_magnitude) & (mean < self.max_magnitude)
        # mean_mask = mean_mask.type_as(mean)
        mean_mask = ((mean > 0.0) & (mean < self.max_magnitude)).type_as(mean)
        mean = mean*mean_mask
        std = std*mean_mask
        
        # Now replace the zero values with the maximum values allowed for the mean and std
        mean = mean + (1-mean_mask)*self.max_magnitude
        std = std + (1-mean_mask)*self.max_magnitude_std # 1.0 magnitude is the maximum std allowed

        return mean, std, mean_mask
    
    def expit(self, x):
        """
        expit function. Turns x~[-inf, inf] to x~[0, 1]
        """
        return 1.0/(1.0+torch.exp(-x))
    
    
    def logit(self, x, eps=1e-5):
        """
        logit function. Turns x~[0, 1] to x~[-inf, inf]
        """
        assert torch.max(x) <= 1. and torch.min(x) >= 0.
        x = torch.clip(x, eps, 1-eps) # clip to avoid infinities
        return torch.log(x/(1.-x))

    def fill_lower_triangular(self, L):
        """
        Fills a lower triangular matrix with values from a flat vector.
        We then take the softplus of the diagonal elements because they must be positive.

        L: Tensor of shape [B, n_dims*(n_dims+1)/2], the lower triangular matrix
        return: Tensor of shape [B, n_dims, n_dims], the filled lower triangular matrix
        """
        # Calculate n_dims by solving the quadratic equation n*(n + 1)/2 = L.shape[1]
        n_dims = int((np.sqrt(1 + 8 * L.shape[1]) - 1) / 2)
        
        # Create indices for lower triangular part
        idxs = torch.tril_indices(row=n_dims, col=n_dims)
        
        # Create zero matrix with the same type and device as L
        mat = torch.zeros(L.shape[0], n_dims, n_dims, dtype=L.dtype, device=L.device)
        
        # Fill lower triangular part of mat with values from L
        mat[:, idxs[0], idxs[1]] = L

        # take exp of the diagonal elements because they must be positive
        # Get the diagonal elements of each matrix in the batch
        diagonal_elements = mat.diagonal(dim1=1, dim2=2)

        # Compute the exponential of the diagonal elements
        pos_diagonal = F.softplus(diagonal_elements)

        # Create diagonal matrices with the exponentials
        pos_diagonal_matrices = torch.diag_embed(pos_diagonal)

        # Now, subtract the current diagonal from the original matrices and add the new diagonal
        mat = mat - torch.diag_embed(diagonal_elements) + pos_diagonal_matrices

        return mat
    
    def sample_mixture(self, num_samples, dists, weights):
        """
        Sample from a mixture of Gaussians

        num_samples ~ int, the number of samples to draw
        dists ~ list of distributions, the distributions to sample from. Here we use MultivariateNormal but it can be any distribution. Each Gaussian should have mean shape [B, features] and covariance shape [B, features, features].
        weights ~ [B, N], the weights of the distributions where N is the number of Gaussians. torch.sum(weights, dim=1) should be 1.

        return: samples ~ [num_samples, B, features], the samples drawn from the mixture of Gaussians
        """
        B = weights.shape[0]
        N = len(dists)  # number of Gaussians
        features = dists[0].mean.shape[-1]  # number of features
        
        # 1. Sample distribution indices based on weights
        cat = Categorical(weights)
        indices = cat.sample((num_samples,)).transpose(0, 1)  # [B, num_samples]
        
        # 2. Sample from all distributions
        all_samples_list = [dist.sample_n(num_samples).transpose(0, 1) for dist in dists]
        all_samples = torch.stack(all_samples_list, dim=2)  # [B, num_samples, N, features]

        # 3. Use torch.gather to select the appropriate samples
        indices = indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, features)
        samples = torch.gather(all_samples, 2, indices).squeeze(2)

        return samples.transpose(0, 1)  # [num_samples, B, features]


    def forward(self, xs, param_true=None):
        mean, std, mean_mask = self.get_weighted_mean_std(xs)

        xs[:, :, :self.num_bands] = mean_mask * (xs[:, :, :self.num_bands] - mean)/std
        xs[:, :, self.num_bands:] = mean_mask * xs[:, :, self.num_bands:]/std
        _, X = self.encoder_network(xs)
        del xs
        X = torch.cat((X, mean.squeeze(1), std.squeeze(1)), dim=1)

        params_mean = self.params_mean_net(X)
        gaussian_mixture_coefficients = self.gaussian_mixture_coefficients_net(torch.cat((X, params_mean), dim=1))
        L_params = self.L_params_net(torch.cat((X, params_mean, gaussian_mixture_coefficients), dim=1))
    
        params_mean_vector = torch.zeros(params_mean.shape[0], self.n_params, self.num_Gaussian_parameterization, dtype=params_mean.dtype, device=params_mean.device)
        for i in range(self.num_Gaussian_parameterization):
            params_mean_vector[:,:,i] = params_mean[:,i*self.n_params:(i+1)*self.n_params]

        param_pred_L_matrix = torch.zeros(L_params.shape[0], self.n_params, self.n_params, self.num_Gaussian_parameterization, dtype=params_mean.dtype, device=params_mean.device)
        for i in range(self.num_Gaussian_parameterization):
            param_pred_L_matrix[:,:,:,i] = self.fill_lower_triangular(L_params[:,i*self.n_params*(self.n_params+1)//2:(i+1)*self.n_params*(self.n_params+1)//2])

        # Now we can calculate the loss
        loss = 0.0
        log_probs = []
        if param_true is not None:

            # convert from [0, 1] to [-inf, inf]
            param_true_logit = self.logit(param_true)

            for i in range(self.num_Gaussian_parameterization):
                param_dist = MultivariateNormal(loc=params_mean_vector[:,:,i], scale_tril=param_pred_L_matrix[:,:,:,i])
                log_prob = param_dist.log_prob(param_true_logit)

                weighted_log_prob = log_prob + torch.log(gaussian_mixture_coefficients[:, i])
                log_probs.append(weighted_log_prob)

            log_probs = torch.stack(log_probs, dim=1)
            log_sum_exp = torch.logsumexp(log_probs, dim=1)
            param_loss = -log_sum_exp.mean() / self.n_params



            samples = self.sample_mixture(self.num_samples, [MultivariateNormal(loc=params_mean_vector[:,:,i], scale_tril=param_pred_L_matrix[:,:,:,i]) for i in range(self.num_Gaussian_parameterization)], gaussian_mixture_coefficients)
            samples = self.expit(samples) # convert back to [0, 1]
            params_mean = torch.mean(samples, dim=0) # params_mean ~ [B, n_params], now in the range from 0 to 1

            # calculate metrics
            rmse = torch.sqrt(torch.mean((param_true - params_mean)**2))
            mae = torch.mean(torch.abs(param_true - params_mean))

            NGLL_batch = -log_sum_exp / self.n_params
            rmse_batch = torch.sqrt(torch.mean((param_true - params_mean)**2, dim=1))
            mae_batch = torch.mean(torch.abs(param_true - params_mean), dim=1)

        else:
            param_loss = 0.0

        return param_loss, rmse, mae, params_mean, samples, NGLL_batch, rmse_batch, mae_batch
    
    @torch.no_grad()
    def predict(self, xs, param_true=None):
        return self.forward(xs, param_true)

class LatentSDE(nn.Module):
    """
    This is the main model that combines the Transformer/RNN encoder, latent SDE driving signal, and auto-differential simulation of the transfer functions.
    The convolution of the driving signal with the transfer functions is done in the forward pass. 
    """
    def __init__(self, input_size, hidden_size, driving_latent_size, context_size, num_bands, num_iterations, driving_resolution, device, freq_effective, lambda_effective_Angstrom,
                 n_params_accretion, n_params_variability, min_max_array, kernel_num_days, kernel_resolution, parameters_keys, dt=1e-3, method="euler",logqp=False,
                 num_layers=3, num_heads=5, param_loss_weight=1.0, log_pxs_weight=1.0, log_pxs2_weight=1.0, log_pxs2_leeway=0.005, num_Gaussian_parameterization=1, time_delay_loss_weight=1.0,dropout_rate=0.0, 
                 give_redshift=False, KL_anneal_epochs=0, param_anneal_epochs=0, relative_mean_time=True, reference_band=3, min_magnitude=13, max_magnitude=27,):
        """
        input_size: int, the dimension of the input data for light curve of shape [B, T, input_size]
        hidden_size: int, the dimension of the hidden layers
        driving_latent_size: int, the dimension of the latent driving signal
        context_size: int, the dimension of the context vector
        num_bands: int, the number of bands in the light curve
        num_iterations: int, the number of iterations to run the RIM block
        driving_resolution: floar, the resolution of the driving signal in days
        device: PyTorch device, the device to use
        freq_effective: np.array, the effective frequencies of the bands in Hz (i.e. the effective frequency of the ugrizy bands for LSST)
        lambda_effective_Angstrom: np.array, the effective wavelengths of the bands in Angstroms (i.e. the effective wavelength of the ugrizy bands for LSST)
        n_params_accretion: int, the number of parameters to predict for the accretion disk and black holes (this is what is used in the auto-differential simulation of the transfer functions)
        n_params_variability: int, the number of parameters to predict for the driving variability
        min_max_array: np.array, the minimum and maximum values for the parameters to predict used for normalization, we sampled the parameters uniformly from this range
        kernel_num_days: float, the number of days in the kernel
        kernel_resolution: float, the resolution of the kernel in days
        parameters_keys: list of strings, the keys of the parameters to predict
        dt: float, the time step size of the SDE solver, must be between 0 and 1, default is 1e-3
        method: str, the method to use for the SDE solver, default is "euler"
        logqp: bool, whether to use the logqp method which would sample from the latent space, default is False
        num_layers: int, the number of layers in the RNN encoder model, default is 3
        num_heads: int, the number of heads in the multi-head attention of the Transformer model, default is 4
        param_loss_weight: float, the weight of the parameter loss, default is 1.0
        log_pxs_weight: float, the weight of the log_pxs loss, default is 1.0
        log_pxs2_weight: float, the weight of the log_pxs2 loss, default is 1.0
        log_pxs2_leeway: float, the leeway for the log_pxs2 loss, default is 0.005, this is mostly just for numerical stability
        num_Gaussian_parameterization: int, the number of Gaussian parameterizations to use for Gaussian mixture model, default is 1
        time_delay_loss_weight: float, the weight of the time delay loss, default is 1.0
        dropout_rate: float, the dropout rate, should be between 0 and 1, default is 0.0
        give_redshift: bool, whether to give the redshift as an input to the model, default is False, Not used at this time
        KL_anneal_epochs: int, the number of epochs to anneal the KL loss, default is 0, only used if logqp is True
        param_anneal_epochs: int, the number of epochs to anneal the parameter loss, default is 0
        relative_mean_time: bool, whether to use the relative mean time, default is True
        reference_band: int, the reference band to use for the relative mean time, default is 3 (the i-band)
        min_magnitude: float, the minimum magnitude of observations in the light curve, default is 13
        max_magnitude: float, the maximum magnitude of observations in the light curve, default is 27
        """
        super(LatentSDE, self).__init__()
        self.hidden_size = hidden_size #driving, kernels, bias, mult
        self.context_size = context_size
        self.num_bands = num_bands
        self.out_dim = num_bands
        self.num_iterations = num_iterations
        self.num_layers = num_layers
        self.give_redshift = give_redshift # Not used at this time

        self.log_pxs_weight = log_pxs_weight
        self.log_pxs2_weight = log_pxs2_weight
        self.log_pxs2_leeway = log_pxs2_leeway 
        # the resolution of the input data in days.
        self.driving_resolution = driving_resolution 
        
        self.kernel_num_days = kernel_num_days
        # the resolution of the kernel in days.
        self.kernel_resolution = kernel_resolution
        self.parameters_keys = parameters_keys
        # this is the number of points in our output kernel
        self.kernel_size = int(kernel_num_days/kernel_resolution)

        # Extra time needed in the driving signal to account for the kernel in the convolution.
        self.extra_time_from_kernel = int(kernel_num_days/driving_resolution)-1

        self.param_loss_weight = param_loss_weight
        self.num_Gaussian_parameterization = num_Gaussian_parameterization
        self.time_delay_loss_weight = time_delay_loss_weight
        self.freq_effective = torch.from_numpy(freq_effective).to(device).float()
        self.lambda_effective_Angstrom = torch.from_numpy(lambda_effective_Angstrom).to(device).float()
        # This is just for numerical stability. Usually just to avoid division by zero.
        self.epsilon = 1e-6
        self.relative_mean_time = relative_mean_time
        self.reference_band = reference_band

        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        # Max value allowed in photometric error
        self.max_magnitude_std = 1.0 

        self.driving_latent_size = driving_latent_size

        self.logqp = logqp

        self.N_mean_band = self.num_bands-1 if self.relative_mean_time else self.num_bands

        # This is the number of segments we break up the power spectrum of our driving signal into. Used to help predict the variability parameters.
        self.freq_segments = 5

        self.epoch = 0
        self.KL_anneal_epochs = KL_anneal_epochs
        self.param_anneal_epochs = param_anneal_epochs

        self.n_params_accretion = n_params_accretion
        self.n_params_variability = n_params_variability
        self.n_params = n_params_accretion+n_params_variability
        self.min_max_array = torch.from_numpy(min_max_array).to(device).float()

        # Number of samples to draw from the Gaussian mixture model to get the mean parameters. We want at least 10,000 to get a good estimate of the mean.
        self.num_samples = 50_000 

        ### Some tests to make sure the functions are working correctly ###
        # test the interpolation
        self.test_interpolate()
        # test the fill lower triangular
        self.test_fill_lower_triangular()
        # test the convolution with this test funciton
        self.test_convolution_driving_kernel()
        
        # This is the auto-differential model for generating the transfer functions.
        self.generate_tf = GenerateTFModule(self.parameters_keys, self.lambda_effective_Angstrom, self.kernel_num_days, self.kernel_resolution)
        
        # This can save GPU memory but slow down training. Could be useful depending on hardware.
        self.use_checkpoints = False

        # Encoder for RIM to adjust the accretion disk / black hole parameters and latent space of the SDE
        self.grad_Transformer = RNN_Transformer(input_dim=6*self.out_dim,
                                            model_dim=hidden_size,
                                            output_dim=hidden_size,
                                            num_heads=num_heads,
                                            num_layers=num_layers,
                                            dropout=dropout_rate,
                                        )

        # Define the latent SDE model for the driving signal. This also contains an RNN/Transformer encoder.
        self.SDE = SDE(input_size=2*self.out_dim, 
                        latent_size=driving_latent_size,
                        hidden_size=hidden_size, 
                        context_size=context_size, 
                        output_dim=1,                         # Output dimension of the SDE is 1 (the x-ray corona driving signal)
                        num_layers_encoder=num_layers, 
                        num_heads=num_heads,
                        dt=dt,
                        dropout_rate=dropout_rate,
                        method=method,
                        logqp=logqp,
                        device=device,
                    )
        
        # RIM block to iteratively update the accretion disk/black hole parameters and latent space of the SDE
        self.RIM_block = RNN_Cell(input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout_rate=dropout_rate)
        
        # Latent space of the SDE
        z0_delta_input_size = hidden_size + driving_latent_size + n_params_accretion*num_Gaussian_parameterization + num_Gaussian_parameterization + context_size + 2*num_bands + 1
        self.z0_delta = Linear_Skip_new(input_size=z0_delta_input_size,
                                        output_size=driving_latent_size,
                                        hidden_size=hidden_size,
                                        dropout_rate=dropout_rate)

        # Used to suppress the iteration of the parameters in the RIM block. Helps to stabilize the training.
        self.param_iteration_supression_net = nn.Sequential(
                                                            nn.LayerNorm(hidden_size+1), # Delta and the iteration number
                                                            nn.Linear(hidden_size+1, hidden_size),
                                                            nn.LeakyReLU(),
                                                            nn.Dropout(dropout_rate),
                                                            nn.LayerNorm(hidden_size),
                                                            nn.Linear(hidden_size, 1),
                                                            nn.Softplus(),
                                                            )

        # Used to suppress the iteration of the latent space in the RIM block. Helps to stabilize the training.    
        self.z0_iteration_supression_net = nn.Sequential(
                                                    nn.LayerNorm(hidden_size+1), # Delta and the iteration number
                                                    nn.Linear(hidden_size+1, hidden_size),
                                                    nn.LeakyReLU(),
                                                    nn.Dropout(dropout_rate),
                                                    nn.LayerNorm(hidden_size),
                                                    nn.Linear(hidden_size, 1),
                                                    nn.Softplus(),
                                                    )
                   
        # RNN to produce uncertainty in the reconstructed driving signal and light curve
        self.uncertainty_net = RNN_Encoder(input_size=4*num_bands+1, hidden_size=hidden_size, output_size=num_bands+1)


        time_delay_uncertainty_net_input_size = self.n_params+context_size+4*num_bands
        time_delay_uncertainty_net_hidden_size = hidden_size//2
        if self.relative_mean_time:
            num_bands_time_delay = num_bands - 1 # We don't need to predict the time delay for the reference band
        else:
            num_bands_time_delay = num_bands
        time_delay_uncertainty_net_output_size = num_bands_time_delay * (num_bands_time_delay + 1) // 2 # the number of unique elements in a symmetric NxN matrix is N*(N+1)/2.
        # Produce the uncertainty in the time delay between the bands
        self.time_delay_uncertainty_net = Linear_Skip_new(input_size=time_delay_uncertainty_net_input_size,
                                                        output_size=time_delay_uncertainty_net_output_size,
                                                        hidden_size=time_delay_uncertainty_net_hidden_size,
                                                        dropout_rate=dropout_rate)

        param_delta_input_size = hidden_size + context_size + num_Gaussian_parameterization*n_params_accretion + 2*num_bands + 1 + num_Gaussian_parameterization
        # Predict the mean parameters of the accretion disk and black hole
        self.param_delta = Linear_Skip_new(input_size=param_delta_input_size, 
                                            output_size=num_Gaussian_parameterization*n_params_accretion+num_Gaussian_parameterization, 
                                            hidden_size=hidden_size, 
                                            dropout_rate=dropout_rate)


        param_variability_input_size = 2*self.freq_segments+5+context_size+3*driving_latent_size+2*num_bands+n_params_accretion*num_Gaussian_parameterization+num_Gaussian_parameterization 
        if self.logqp:
            param_variability_input_size += driving_latent_size
        # Predict the mean variability parameters of the driving signal
        self.param_variability_net = Linear_Skip_new(input_size=param_variability_input_size,
                                                    output_size=num_Gaussian_parameterization*n_params_variability,
                                                    hidden_size=hidden_size,
                                                    dropout_rate=dropout_rate)

        num_output = num_Gaussian_parameterization*self.n_params*(self.n_params+1)//2  
        L_params_net_input_size = 2*self.freq_segments+5+num_Gaussian_parameterization+num_Gaussian_parameterization*self.n_params+3*driving_latent_size+context_size+2*num_bands
        if self.logqp:
            L_params_net_input_size += driving_latent_size
        # Predict the lower triangular matrix of the covariance matrix of the accretion disk / black hole and driving variability parameters
        self.L_params_net = Linear_Skip_new(input_size=L_params_net_input_size,
                                            output_size=num_output,
                                            hidden_size=hidden_size,
                                            dropout_rate=dropout_rate)
                                        
        bias_mult_net_input_size = 2+6*num_bands+context_size+self.n_params
        # Predict the normalization parameters for the reconstructed light curve
        self.bias_mult_net = Linear_Skip_new(input_size=bias_mult_net_input_size,
                                            output_size=3*(num_bands+1), # also reconstruct the driving signal
                                            hidden_size=hidden_size,
                                            dropout_rate=dropout_rate)


    def step_epoch(self):
        """
        This function is called at the end of each epoch to update self.epoch.
        """
        self.epoch += 1

    def linear_anneal_weight(self, anneal_epoch, min_weight=1e-2):
        """
        This function returns the weight from 0 to 1 to linearally anneal the loss weight from 0 to 1.

        anneal_epoch: int, the number of epochs to anneal the weight
        min_weight: float, the minimum weight to use, default is 1e-2

        return: float, the weight to use between 0 and 1
        """
        if anneal_epoch == 0:
            return 1.0
        else:
            return min(1.0, self.epoch/anneal_epoch+min_weight)

    
    def flux_to_mag(self,flux):
        """
        This function converts a flux in units of  to a magnitude in units of AB mag.

        flux: flux per frequency in units of erg/s/cm^2/Hz 
        return: magnitude in units of AB mag
        """
        flux = torch.clip(flux,1e-50,1e10)
        mag = -2.5*torch.log10(flux) - 48.60
        return torch.clip(mag,0.0,50.0)

    
    def mag_to_flux(self,mag):
        """
        This function converts a magnitude in units of AB mag to a flux in units of erg/s/cm^2/Hz.

        mag: magnitude in units of AB mag
        returns: flux per frequency in units of erg/s/cm^2/Hz
        """
        return 10.0**(-0.4*(mag+48.60))                                             

    # Only used if the driving signal and transfer functions are not the same resolution.
    # Found to have issues when using this method, so we keep both at the same resolution (1 day).
    def interpolate(self, x, xp, fp):
        """
        One-dimensional linear interpolation for monotonically increasing sample points.

        Adapted from: https://github.com/pytorch/pytorch/issues/50334

        x: the :math:`x`-coordinates at which to evaluate the interpolated values. Shape [B, T2] or [T2]
        xp: the :math:`x`-coordinates of the data points, must be increasing. Shape [B, T1] or [T1]
        fp: the :math:`y`-coordinates of the data points, same length as `xp`. Shape [B, T1, features], [B, T1], or [T1]

        returns: the interpolated values, same size as `x`. shape [B, T2, features], [B, T2], or [T2]
        """
        assert len(xp.shape) in (1,2)
        assert len(x.shape) in (1,2)
        assert len(fp.shape) in (1,2,3)

        # Reshape inputs if necessary
        x = x if len(x.shape) == 2 else x.unsqueeze(0)
        xp = xp if len(xp.shape) == 2 else xp.unsqueeze(0)
        if len(fp.shape) == 1:
            fp = fp.unsqueeze(0).unsqueeze(-1)
        elif len(fp.shape) == 2:
            fp = fp.unsqueeze(-1)

        m = (fp[:, 1:, :] - fp[:, :-1, :]) / (xp[:, 1:] - xp[:, :-1]).unsqueeze(-1)
        b = fp[:, :-1, :] - m * xp[:, :-1].unsqueeze(-1)

        indices = torch.sum(x[:, :, None] >= xp[:, None, :], dim=-1) - 1
        indices = torch.clamp(indices, 0, m.shape[1] - 1)

        output = m[torch.arange(m.shape[0]).unsqueeze(-1), indices]*x[:, :, None] + b[torch.arange(b.shape[0]).unsqueeze(-1), indices]
        output = output.squeeze(-1)

        return output

    def test_interpolate(self):
        """
        Function to test the interpolate function. Makes sure it is working correctly.
        """
        # test case 1
        x = torch.tensor([[0.5, 1.5, 2.5], [0.75, 1.75, 2.75]])
        xp = torch.tensor([[0., 1., 2., 3.], [0., 1., 2., 3.]])
        fp = torch.tensor([[0., 1., 2., 3.], [0., 2., 4., 6.]])

        output = self.interpolate(x, xp, fp)
        expected_output = torch.tensor([[0.5, 1.5, 2.5], [1.5, 3.5, 5.5]])  # Line y=x for the first batch and y=2x for the second batch
        assert torch.allclose(output, expected_output), f'Output {output} does not match expected output {expected_output}'

        # test case 2
        x = torch.tensor([0.5, 1.5, 2.5])
        xp = torch.tensor([0., 1., 2., 3.])
        fp = torch.tensor([0., 1., 2., 3.])

        output = self.interpolate(x, xp, fp)
        expected_output = torch.tensor([0.5, 1.5, 2.5])  # Line y=x
        assert torch.allclose(output, expected_output), f'Output {output} does not match expected output {expected_output}'

        # test case 3
        x = torch.tensor([1.5, 2.5, 3.5])
        xp = torch.tensor([1.0, 2.0, 3.0, 4.0])
        fp = torch.tensor([1.0, 2.0, 3.0, 4.0])

        expected_output = torch.tensor([1.5, 2.5, 3.5])
        output = self.interpolate(x, xp, fp)
        assert torch.allclose(output, expected_output, atol=1e-7), f'Output {output} does not match expected output {expected_output}'

        # test case 4
        fp = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])

        expected_output = torch.tensor([[1.5, 2.5, 3.5], [3.0, 5.0, 7.0]])
        output = self.interpolate(x, xp, fp)
        assert torch.allclose(output, expected_output, atol=1e-7), f'Output {output} does not match expected output {expected_output}'

        # Case when fp has dimension 3
        fp = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], 
                        [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10.0]]])

        expected_output = torch.tensor([[[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]], 
                                        [[3.0, 5.0], [5.0, 7.0], [7.0, 9.0]]])
        output = self.interpolate(x, xp, fp)

        assert torch.allclose(output, expected_output, atol=1e-7), f'Output {output} does not match expected output {expected_output}'

    def interpolate_driving(self, driving, plot_interpolation=False):
        """
        Function to interpolate the driving signal to the resolution of the convolution.

        driving: driving signal. shape [B, T+padding]
        convolution_resolution: resolution of the convolution in days, float
        plot_interpolation: whether to plot the interpolation or not, bool. Just for debugging purposes.
        """

        t1 = torch.linspace(0.0, 1.0, driving.shape[1], dtype=driving.dtype, device=driving.device)
        t2 = torch.linspace(0.0, 1.0, int(self.driving_resolution*driving.shape[1]/self.kernel_resolution), dtype=driving.dtype, device=driving.device)

        if plot_interpolation:
            plt.plot(t1.detach().cpu().numpy(), driving[0].detach().cpu().numpy(), label="original")

        driving = self.interpolate(t2, t1, driving)

        if plot_interpolation:
            plt.plot(t2.detach().cpu().numpy(), driving[0].detach().cpu().numpy(), label="interpolated")
            plt.legend()
            plt.show()

        return driving

    def interpolate_output(self, output, plot_interpolation=False):
        """
        Function to interpolate the light curve output to the resolution of the convolution.

        output: output of the model. shape [B, T, num_bands]
        convolution_resolution: resolution of the convolution in days, float
        plot_interpolation: whether to plot the interpolation or not, bool. Just for debugging purposes.
        """
        t1 = torch.linspace(0.0, 1.0, output.shape[1], dtype=output.dtype, device=output.device)
        new_length = int(self.kernel_resolution * output.shape[1] / self.driving_resolution)
        t2 = torch.linspace(0.0, 1.0, new_length, dtype=output.dtype, device=output.device)

        interpolated_output = self.interpolate(t2, t1, output)

        if plot_interpolation:
            plt.plot(t1.detach().cpu().numpy(), output[0,:,0].detach().cpu().numpy(), label="original")
            plt.plot(t2.detach().cpu().numpy(), interpolated_output[0,:,0].detach().cpu().numpy(), label="interpolated")
            plt.legend()
            plt.show()

        return interpolated_output

    def convolution_driving_kernel(self, driving, kernels):
        """
        Performs the convolution of the driving signal with the kernels.
        This is a 1D convolution, but we have multiple bands, so we need to reshape the driving signal and the kernels.

        driving: [batch_size, T+kernel_size-1]
        kernels: [batch_size, kernel_size, bands]
        returns: [batch_size, T, bands] 
        """
        batch_size = kernels.shape[0]
        kernel_size = kernels.shape[1]
        bands = kernels.shape[2]
        T = driving.shape[1] - (kernel_size - 1)

        driving_new = torch.zeros(batch_size*bands, T+kernel_size-1, dtype=driving.dtype, device=driving.device)
        for band in range(bands):
            driving_new[band::bands, :] = driving[:, :]
        
        kernels_new = torch.zeros(batch_size*bands, kernel_size, dtype=kernels.dtype, device=kernels.device)
        for band in range(bands):
            kernels_new[band::bands, :] = kernels[:, :, band]
        
        kernels_new = kernels_new.flip(dims=(1,)).unsqueeze(dim=1)
        output = F.conv1d(driving_new, kernels_new, groups=batch_size*bands)

        output_new = torch.zeros(batch_size, T, bands, dtype=output.dtype, device=output.device)
        for band in range(bands):
            output_new[:, :, band] = output[band::bands, :]
        
        return output_new

    def test_convolution_driving_kernel(self):
        """"
        Simple test that our convolution in pytorch matches the numpy convolution
        """
        T = 10
        kernel_size = 3
        batch_size = 2
        bands = 2

        driving = torch.ones(batch_size, T+kernel_size-1)
        kernels = torch.zeros(batch_size, kernel_size, bands)

        driving[0, 0] = 0
        driving[0, 1] = 1
        driving[0, 2] = 2
        driving[1, 0] = -1
        driving[1, 1] = -2
        driving[1, 2] = -3
        driving[0, 5:] = -4
        driving[1, 5:] = 4

        kernels[0, 1, 0] = 1
        kernels[0, 1, 1] = 2
        kernels[0, 2, 0] = 3
        kernels[1, 0, 0] = 5
        kernels[1, 0, 1] = 6
        kernels[1, 1, 0] = 7
        kernels[1, 1, 1] = 8
        kernels[1, 2, 0] = 8
        kernels[1, 2, 1] = 9

        driving_np = driving.numpy()
        kernels_np = kernels.numpy()

        output = self.convolution_driving_kernel(driving, kernels)

        # Convert tensors to numpy arrays for comparison
        output_np = np.zeros((batch_size, T, bands))

        # Perform the convolution using numpy for each item in the batch and each band
        for b in range(batch_size):
            for band in range(bands):
                output_np[b, :, band] = np.convolve(driving_np[b, :], kernels_np[b, :, band], mode='valid')

        assert np.allclose(output.numpy(), output_np), f"Output {output} did not match numpy output {output_np}"

    def convolve(self, driving, kernels, plot_convolution=False):
        """ 
        convolve the driving signal with the kernels. No padding.

        driving ~ [B, T, 1], xray driving signal. Should be in magnitude.
        kernels ~ [B, kernel_size, num_bands], the kernel or transfer function

        returns ~ [B, T, num_bands], the convolution of the driving signal with the kernels
        """
        # interpolate the transfer function from log time to linear time. Should be of shape [B, kernel_size]

        # convert driving to flux in units of erg/s/cm^2/Hz
        driving = self.mag_to_flux(driving)
        # convert from erg/s/cm^2/Hz to erg/s/cm^2 using the u-band effective frequency
        #driving = self.freq_effective[self.reference_band]*driving 
        driving = driving.squeeze(-1) # remove the last dimension
        # upsample driving to convolution resolution
        if self.driving_resolution != self.kernel_resolution:
            driving = self.interpolate_driving(driving)

        # convolve with transfer function kernels
        # driving ~ [B, T+kernel_size-1, 1]
        #kernels ~ [B, kernel_size, num_bands]

        if plot_convolution:
            time = torch.linspace(0, driving.shape[1], driving.shape[1], dtype=driving.dtype, device=driving.device)
            # plot driving vs convolution
            plt.plot(time.detach().cpu().numpy(), driving[0].detach().cpu().numpy(), label="driving")

        output = self.convolution_driving_kernel(driving, kernels)

        if plot_convolution:
            # plot the convolution
            for band in range(output.shape[2]):
                plt.plot(torch.linspace(self.kernel_num_days/self.kernel_resolution, self.kernel_num_days/self.kernel_resolution+output.shape[1], output.shape[1]).type_as(output).to(output.device).detach().cpu().numpy(), output[0,:,band].detach().cpu().numpy(), label=f"{band}-band")
            plt.legend()
            plt.show()

        # want driving ~ [B, T, num_bands]
        # downsample to driving resolution
        # Interpolate down 
        if self.driving_resolution != self.kernel_resolution:
            output = self.interpolate_output(output)
        # convert back from erg/s/cm^2 to erg/s/cm^2/Hz
        #output = output/self.freq_effective
        # convert back to magnitudes
        output = self.flux_to_mag(output)

        return output

    def KL_div(self, P,Q):
        """
        Compute the KL divergence between two distributions:

        P ~ [B, seq_len, features], the target distribution
        Q ~ [B, seq_len, features], the predicted distribution
        
        KL_div = sum(P*log(P/Q)

        returns: the KL divergence between the two distributions of shape [B]
        """
        return torch.mean(torch.sum(P*torch.log(torch.clip(P,1e-7,1.0)/torch.clip(Q,1e-7,1.0)), dim=1)) # average across batch and features

    # Not used
    def JS_div(self, P,Q):
        """
        Compute the Jensen-Shannon divergence between two distributions:

        P ~ [B, seq_len, features], the target distribution
        Q ~ [B, seq_len, features], the predicted distribution
        
        JS_div = 0.5*KL(P||M) + 0.5*KL(Q||M), where M = 0.5*(P+Q) and KL(P||Q) = sum(P*log(P/Q))

        returns: the Jensen-Shannon divergence between the two distributions of shape [B]
        """
        M = 0.5*(P+Q)
        return 0.5*(self.KL_div(P,M)+self.KL_div(Q,M))
    
    def get_mean_time(self, kernels):
        """
        Get the mean time of the kernels. We add the kernel resolution to avoid numerical issues so we never predict/have true means less than one pixel.

        kernels ~ [B, kernel_size, num_bands], the kernel or transfer function. Should already be normalized to a probability distribution.
        returns ~ [B, num_bands], the mean time of the kernels
        """
        # We add one extra day in the time windo so that the log mean doesn't go to -inf every
        time_steps = torch.linspace(0, self.kernel_num_days, self.kernel_size, dtype=kernels.dtype, device=kernels.device)
        mean = torch.sum(kernels*time_steps.unsqueeze(0).unsqueeze(2), dim=1)
        return mean
    
    def get_stdev_kernel(self, kernels):
        """
        Get the standard deviation of the kernels

        kernels ~ [B, kernel_size, num_bands], the kernel or transfer function. Should already be normalized to a probability distribution.
        returns ~ [B, num_bands], the standard deviation of the kernels
        """
        mean = self.get_mean_time(kernels)
        time_steps = torch.linspace(0, self.kernel_num_days, self.kernel_size, dtype=kernels.dtype, device=kernels.device)
        stdev = torch.sqrt(torch.sum(kernels*(time_steps.unsqueeze(0).unsqueeze(2)-mean.unsqueeze(1))**2, dim=1))
        return stdev

    
    def fill_lower_triangular(self, L):
        """
        Fills a lower triangular matrix with values from a flat vector.
        We then take the exp of the diagonal elements because they must be positive.

        L: Tensor of shape [B, n_dims*(n_dims+1)/2], the lower triangular matrix
        returns: Tensor of shape [B, n_dims, n_dims], the lower triangular matrix 
        """
        # Calculate n_dims by solving the quadratic equation n*(n + 1)/2 = L.shape[1]
        n_dims = int((np.sqrt(1 + 8 * L.shape[1]) - 1) / 2)
        
        # Create indices for lower triangular part
        idxs = torch.tril_indices(row=n_dims, col=n_dims)
        
        # Create zero matrix with the same type and device as L
        mat = torch.zeros(L.shape[0], n_dims, n_dims, dtype=L.dtype, device=L.device)
        
        # Fill lower triangular part of mat with values from L
        mat[:, idxs[0], idxs[1]] = L

        # take exp of the diagonal elements because they must be positive
        # Get the diagonal elements of each matrix in the batch
        diagonal_elements = mat.diagonal(dim1=1, dim2=2)

        # Compute the exponential of the diagonal elements
        pos_diagonal = F.softplus(diagonal_elements)

        # Create diagonal matrices with the exponentials
        pos_diagonal_matrices = torch.diag_embed(pos_diagonal)

        # Now, subtract the current diagonal from the original matrices and add the new diagonal
        mat = mat - torch.diag_embed(diagonal_elements) + pos_diagonal_matrices

        return mat

    def numpy_softplus(self, x):
        """
        numpy softplus function

        x: input to the softplus function
        returns: softplus(x) = log(1 + exp(x))
        """
        return np.log(1 + np.exp(x))

    def test_fill_lower_triangular(self):
        """
        Function to test the fill_lower_triangular function. Makes sure it is working correctly.
        """
        # Test Case 1: 
        L1 = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)
        expected_output1 = torch.tensor([[self.numpy_softplus(1), 0], [2, self.numpy_softplus(3)]]).type_as(L1)
        assert torch.allclose(self.fill_lower_triangular(L1), expected_output1), "Test Case 1 Failed"

        # Test Case 2:
        L2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unsqueeze(0)
        expected_output2 = torch.tensor([[self.numpy_softplus(1), 0, 0], [2, self.numpy_softplus(3), 0], [4, 5, self.numpy_softplus(6)]]).type_as(L1)
        assert torch.allclose(self.fill_lower_triangular(L2), expected_output2), "Test Case 2 Failed"

        # Test Case 3: Testing batch input
        L3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_output3 = torch.tensor([[[self.numpy_softplus(1), 0], [2, self.numpy_softplus(3)]], [[self.numpy_softplus(4), 0], [5, self.numpy_softplus(6)]]]).type_as(L1)
        assert torch.allclose(self.fill_lower_triangular(L3), expected_output3), "Test Case 3 Failed"

    def reparameterize(self, q_z_mean_kernels, q_z_log_std_kernels):
        """
        draws the latent vector from the posterior distribution

        q_z_mean_kernels ~ [B, latent_size], the mean of the posterior distribution
        q_z_log_std_kernels ~ [B, latent_size], the log standard deviation of the posterior distribution
        returns ~ [B, latent_size], the sampled latent vector
        """
        return q_z_mean_kernels + torch.exp(q_z_log_std_kernels) * torch.randn_like(q_z_log_std_kernels)  

    
    def get_weighted_mean_std(self, xs):
        """
        Gets the weighted mean and standard deviation of the predicted light curves

        xs ~ [B, T, 2*num_bands], we have num_bands and uncertainty for each band
        returns: mean ~ [B, 1, num_bands], std ~ [B, 1, num_bands], mean_mask ~ [B, 1, num_bands]
        """

        mask = (xs[:,:,:self.out_dim] != 0.0).type_as(xs)

        # get weighted mean of the light curve
        # the mean is weighted by the inverse of the variance
        # https://en.wikipedia.org/wiki/Inverse-variance_weighting
        mean = torch.sum(xs[:,:,:self.out_dim]/(xs[:,:,self.out_dim:]+self.epsilon)**2,dim=1)/(torch.sum(mask/(xs[:,:,self.out_dim:]+self.epsilon)**2,dim=1)+self.epsilon)

        #now get the std masking all the zeros of xs
        mean_diff = mask*(xs[:,:,:self.out_dim] - mean.unsqueeze(1))**2 # mask out the zeros again
        std = torch.sqrt(torch.sum(mean_diff/(xs[:,:,self.out_dim:]+self.epsilon)**2+self.epsilon,dim=1)/(torch.sum(mask/(xs[:,:,self.out_dim:]+self.epsilon)**2,dim=1)+self.epsilon))

        mean = mean.unsqueeze(1) # mean ~ [B, 1, num_bands]
        std = std.unsqueeze(1)   # std ~ [B, 1, num_bands]

        # Get the mask where the mean is less than self.min_magnitude or greater than self.max_magnitude. 
        # Nominally the mean and std should already be masked though from not being observed in the first place in which case they will be zero for that band. 
        mean_mask = ((mean > 0.0) & (mean < self.max_magnitude)).type_as(mean)
        mean = mean*mean_mask
        std = std*mean_mask
        
        # Now replace the zero values with the maximum values allowed for the mean and std
        mean = mean + (1-mean_mask)*self.max_magnitude
        std = std + (1-mean_mask)*self.max_magnitude_std # 1.0 magnitude is the maximum std allowed

        return mean, std, mean_mask
    
    
    def expit(self, x):
        """
        expit function. Turns x~[-inf, inf] to x~[0, 1]

        x: input to the expit function
        returns: expit(x) = 1/(1+exp(-x))
        """
        return 1.0/(1.0+torch.exp(-x))
    
    
    def logit(self, x, eps=1e-5):
        """
        logit function. Turns x~[0, 1] to x~[-inf, inf]

        x: input to the logit function
        eps: small number to avoid infinities
        returns: logit(x) = log(x/(1-x))
        """
        assert torch.max(x) <= 1. and torch.min(x) >= 0.
        x = torch.clip(x, eps, 1-eps) # clip to avoid infinities
        return torch.log(x/(1.-x))

    def get_physical_params(self, params):
        """
        This function converts the predicted parameters to physical parameters

        params ~ [B, n_params], the predicted parameters, should be in the range [0, 1]
        returns: [B, n_params], the physical parameters in their original range
        """
        assert torch.max(params) <= 1. and torch.min(params) >= 0., f"params should be in the range [0, 1] but got {params}"
        return (self.min_max_array[:params.shape[1],1] - self.min_max_array[:params.shape[1],0])*params + self.min_max_array[:params.shape[1],0] # convert to the original range

    
    def analyze_driving_signal(self, driving):
        """
        This function gets relevent information from the driving signal
        
        driving ~ [B, T+kernel_size-1, 1], the driving signal
        returns: [B, 2*self.freq_segments+5], some relevant information from the driving signal that can be used as features in the network
        """

        driving_without_kernel = driving[:, self.extra_time_from_kernel:]

        driving_mean = torch.mean(driving_without_kernel, dim=1)
        driving_std = torch.std(driving_without_kernel, dim=1)
        driving_mean_absolute_deviation = torch.mean(torch.abs(driving_without_kernel - driving_mean.unsqueeze(1)), dim=1)

        driving_total_variation = torch.sum(torch.abs(driving_without_kernel[:,1:]-driving_without_kernel[:,:-1]), dim=1)
        driving_total_square_variation = torch.sqrt(torch.sum((driving_without_kernel[:,1:]-driving_without_kernel[:,:-1])**2, dim=1))

        power_spectrum = torch.abs(torch.fft.fft(driving_without_kernel.squeeze(2), dim=1))**2
        power_spectrum = power_spectrum[:, :power_spectrum.shape[1]//2]  # consider only the first half for real signals

        # Get frequencies for each FFT bin
        frequencies = torch.fft.fftfreq(driving_without_kernel.shape[1])[:driving_without_kernel.shape[1]//2].type_as(driving_without_kernel).to(driving_without_kernel.device)
        # Skip the first frequency (0) (since it's log will be -inf)
        frequencies = frequencies[1:]
        power_spectrum = power_spectrum[:, 1:]

        segments = self.freq_segments 
        spectral_flatness = torch.zeros(driving_without_kernel.shape[0], segments, dtype=driving_without_kernel.dtype, device=driving_without_kernel.device)
        spectral_slope = torch.zeros(driving_without_kernel.shape[0], segments, dtype=driving_without_kernel.dtype, device=driving_without_kernel.device)
        for i in range(segments):

            # Get the segment of the power spectrum
            frequencies_segment = frequencies[i * (len(frequencies) // segments):(i + 1) * (len(frequencies) // segments)]
            power_spectrum_segment = power_spectrum[:, i * (len(frequencies) // segments):(i + 1) * (len(frequencies) // segments)]

            log_frequencies_segment = torch.log(frequencies_segment+self.epsilon)
            log_power_spectrum_segment = torch.log(power_spectrum_segment+self.epsilon)

            ## Calculate the spectral flatness ##
            geometric_mean = torch.exp(torch.mean(log_power_spectrum_segment, dim=1))
            arithmetic_mean = torch.mean(power_spectrum_segment, dim=1)
            spectral_flatness[:, i] = geometric_mean / arithmetic_mean.clamp(min=self.epsilon)
            
            ## Calculate the spectrual slope ##
            # Subtract the mean
            log_frequencies_segment = log_frequencies_segment - torch.mean(log_frequencies_segment)
            log_power_spectrum_segment = log_power_spectrum_segment - torch.mean(log_power_spectrum_segment, dim=1).unsqueeze(1)

            # Calculate the spectral slope (which is the least squares solution)
            spectral_slope[: ,i] = torch.sum(log_frequencies_segment.unsqueeze(0) * log_power_spectrum_segment, dim=1) / torch.sum(log_frequencies_segment ** 2).clamp(min=self.epsilon)

        return torch.cat((self.normalized_mean(driving_mean),
                          self.normalized_std(driving_std),
                          driving_mean_absolute_deviation,
                          driving_total_variation/2.0, # divide by 2.0 to set to similar scale as other inputs
                          driving_total_square_variation,
                          spectral_flatness,
                          spectral_slope/3.0), # divide by 3.0 to set to similar scale as other inputs
                          dim=1)

    def normalized_mean(self, mean):
        """
        Function to normalize the mean of the driving signal to a similar scale as the other inputs

        mean ~ [B, num_bands], the mean of the driving signal
        returns: [B, num_bands], the normalized mean
        """
        return (mean - 20.0)/5.0
    
    def normalized_std(self, std):
        """
        Function to normalize the standard deviation of the driving signal to a similar scale as the other inputs

        std ~ [B, num_bands], the standard deviation of the driving signal
        returns: [B, num_bands], the normalized standard deviation
        """
        return 2*std

    def apply_normalization(self, _xs, driving_reconstructed, mean, std, bias_norm, bias_flux_norm, mult_norm):
        """
        Apply normalization to the predicted light curve and driving signal

        _xs ~ [B, T, num_bands], the predicted light curve
        driving_reconstructed ~ [B, T, 1], the reconstructed driving signal
        mean ~ [B, num_bands], the mean of the light curve
        std ~ [B, num_bands], the standard deviation of the light curve
        bias_norm ~ [B, num_bands+1], the bias of the normalization
        bias_flux_norm ~ [B, num_bands+1], the bias of the flux normalization
        mult_norm ~ [B, num_bands+1], the multiplicative factor of the normalization

        returns: _xs ~ [B, T, num_bands], driving_reconstructed ~ [B, T, 1], The normalized light curve and driving signal
        """

        # Add the bias to the flux. This is important due to host galaxy flux
        additional_flux =  self.mag_to_flux(bias_flux_norm[:, :, 1:]*mean) - self.mag_to_flux(mean)
        additional_flux = torch.nan_to_num(additional_flux, nan=0.0, posinf=0.0, neginf=0.0)
        # unnormalize the light curve back to the original scale in magnitude to apply the normalization
        _xs = std*_xs + mean
        _xs = self.flux_to_mag(self.mag_to_flux(_xs) + additional_flux)
        # reapply the normalization
        _xs = (_xs - mean)/std

        _xs = mult_norm[:, :, 1:] * _xs + bias_norm[:, :, 1:]

        _xs = torch.nan_to_num(_xs, nan=20.0, posinf=20.0, neginf=20.0)

        driving_reconstructed = driving_reconstructed * std[:, :, self.reference_band].unsqueeze(-1) + mean[:, :, self.reference_band].unsqueeze(-1)
        additional_flux_driving = self.mag_to_flux(bias_flux_norm[:, :, 0].unsqueeze(-1)*mean[:, :, self.reference_band].unsqueeze(-1)) - self.mag_to_flux(mean[:, :, self.reference_band].unsqueeze(-1))
        additional_flux_driving = torch.nan_to_num(additional_flux_driving, nan=0.0, posinf=0.0, neginf=0.0)
        driving_reconstructed = self.flux_to_mag(self.mag_to_flux(driving_reconstructed) + additional_flux_driving)
        driving_reconstructed = (driving_reconstructed - mean[:, :, self.reference_band].unsqueeze(-1))/std[:, :, self.reference_band].unsqueeze(-1)
        driving_reconstructed = mult_norm[:, :, 0].unsqueeze(-1) * driving_reconstructed + bias_norm[:, :, 0].unsqueeze(-1)
        driving_reconstructed = torch.nan_to_num(driving_reconstructed, nan=20.0, posinf=20.0, neginf=20.0)        

        return _xs, driving_reconstructed

    
    def sample_mixture(self, num_samples, dists, weights):
        """
        Sample from a mixture of Gaussians

        num_samples ~ int, the number of samples to draw
        dists ~ list of distributions, the distributions to sample from. Here we use MultivariateNormal but it can be any distribution. Each Gaussian should have mean shape [B, features] and covariance shape [B, features, features].
        weights ~ [B, N], the weights of the distributions where N is the number of Gaussians. torch.sum(weights, dim=1) should be 1.
        """
        B = weights.shape[0]
        N = len(dists)  # number of Gaussians
        features = dists[0].mean.shape[-1]  # number of features
        
        # 1. Sample distribution indices based on weights
        cat = Categorical(weights)
        indices = cat.sample((num_samples,)).transpose(0, 1)  # [B, num_samples]
        
        # 2. Sample from all distributions
        all_samples_list = [dist.sample_n(num_samples).transpose(0, 1) for dist in dists]
        all_samples = torch.stack(all_samples_list, dim=2)  # [B, num_samples, N, features]

        # 3. Use torch.gather to select the appropriate samples
        indices = indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, features)
        samples = torch.gather(all_samples, 2, indices).squeeze(2)

        return samples.transpose(0, 1)  # [num_samples, B, features]

    def forward(self, xs, true_LC = None, driving_true=None, mean_time_true=None, param_true=None, save_iterations = False, redshift=None, batch_fraction_sim=1.0, print_metrics=False):
        """
        xs ~ [B, T, 2*num_bands], we have num_bands and uncertainty for each band
        true_LC ~ [B, T, num_bands], the true light curve
        driving_true ~ [B, T+kernel_size-1, 1], the true driving signal
        mean_time_true ~ [B, num_bands], the time delays of each band with respect to the driving signal
        param_true ~ [B, n_params], the true parameters of our LC
        save_iterations ~ bool, if true, save the kernels of each iteration 
        redshift ~ [B], the redshift of each light curve
        batch_fraction_sim ~ float, fraction of the batch that is used by the transfer function sim at a time to save GPU memory.
        print_metrics ~ bool, if true, print the metrics of the model

        returns output_dict containing many outputs
        """
        dtype = xs.dtype
        device = xs.device

        if redshift is not None:
            assert self.give_redshift == True, "redshift is not None but give_redshift is False"
            redshift = redshift.unsqueeze(-1) # redshift ~ [B, 1]
        else:
            assert self.give_redshift == False, "redshift is None but give_redshift is True"

        if self.relative_mean_time and mean_time_true is not None:
            # subtract the time from the reference band. Only the relative time delays are important.
            mean_time_true = mean_time_true - mean_time_true[:, self.reference_band].unsqueeze(1) 
            # Get rid of the first time delay, since it is always 0.0
            mean_time_true = torch.cat((mean_time_true[:,:self.reference_band], mean_time_true[:,self.reference_band+1:]), dim=1)
            mean_time_true = torch.nan_to_num(mean_time_true, nan=0.0, posinf=0.0, neginf=0.0) 
            
        mean, std, mean_mask = self.get_weighted_mean_std(xs)

        # save the kernels for each iteration if save_iterations is True
        kernel_list = [] 
        # save the predicted light curves for each iteration if save_iterations is True
        _xs_list = []
        # save the driving signal for each iteration if save_iterations is True
        driving_list = []

        param_pred_mean_list = []
        param_pred_L_list = []
        gaussian_mixture_coefficients_list = []
        mean_time_pred_list = []

        # the extra padding because we need to get rid of extra times from the convolution
        padding_size = int(self.kernel_num_days/self.driving_resolution)-1

        params_accretion = torch.zeros(xs.shape[0], self.num_Gaussian_parameterization*self.n_params_accretion, dtype=dtype, device=device)
        gaussian_mixture_coefficients_not_normalized = torch.zeros(xs.shape[0], self.num_Gaussian_parameterization, dtype=dtype, device=device)

        # Initialize the reconstruction to zero
        _xs_pad = torch.zeros(xs.shape[0], xs.shape[1]+padding_size, self.num_bands, dtype=dtype, device=device)

        # initialize the driving signal latent space to zero
        z0 = torch.zeros(xs.shape[0], self.driving_latent_size, dtype=dtype, device=device)

        # initialize the hidden state of the RIM
        h = torch.zeros(xs.shape[0], self.hidden_size, dtype=dtype, device=device)

        # Loss that we will optimize each iteration
        loss = 0.0
        loss_list = []
        
        for iteration in range(self.num_iterations):
            # normalize the input
            mask = (xs[:,:,:self.out_dim] != 0.0).type_as(xs)
            xs[:,:,:self.out_dim] = mask*((xs[:,:,:self.out_dim] - mean)/std) # normalize the context points
            xs[:,:,self.out_dim:] = mask*(xs[:,:,self.out_dim:]/std)      # normalize the uncertainties of the context points
            
            del mask

            xs_pad = torch.zeros(xs.shape[0], xs.shape[1]+padding_size, xs.shape[2], dtype=dtype, device=device)
            xs_pad[:, padding_size:] = xs

            mask = (xs_pad[:,:,:self.out_dim] != 0.0).type_as(xs_pad)

            if iteration == 0: 
                # encode the context points, we only need to do this once 
                ts = torch.linspace(0, 1, xs.shape[1]+padding_size, dtype=dtype, device=device)
                context = self.SDE.encode(xs_pad, ts) # [B, context_size]
                context = torch.nan_to_num(context, nan=0.01, posinf=3.0, neginf=-3.0)
                del ts

            # masked mean of the noise
            mean_noise_var = torch.sum(mask*xs_pad[:,:,self.out_dim:]**2, dim=1)/(torch.sum(mask, dim=1)+self.epsilon)

            grad = 2*mask*((_xs_pad-xs_pad[:,:,:self.out_dim])/(xs_pad[:,:,self.out_dim:]+self.epsilon)**2) * mean_noise_var.unsqueeze(1)
            z_score = mask*((_xs_pad-xs_pad[:,:,:self.out_dim])/(xs_pad[:,:,self.out_dim:]+self.epsilon)) * torch.sqrt(mean_noise_var).unsqueeze(1)
            NGLL_context = 0.5 * mask *((_xs_pad-xs_pad[:,:,:self.out_dim])**2/(xs_pad[:,:,self.out_dim:]+self.epsilon)**2) * torch.sqrt(mean_noise_var).unsqueeze(1)

            grad = torch.nan_to_num(grad, nan=0.01, posinf=3.0, neginf=-3.0)
            z_score = torch.nan_to_num(z_score, nan=0.01, posinf=3.0, neginf=-3.0)
            NGLL_context = torch.nan_to_num(NGLL_context, nan=0.01, posinf=3.0, neginf=-3.0)


            _, delta = self.grad_Transformer(torch.cat((xs_pad, _xs_pad, NGLL_context,  z_score, grad), dim=2)) # we only need the context of the transformer
            delta = torch.nan_to_num(delta, nan=0.01, posinf=3.0, neginf=-3.0)

            del mask, xs_pad, _xs_pad, grad, z_score, NGLL_context
            gc.collect()
            torch.cuda.empty_cache()

            delta, h = self.RIM_block(delta, h)
            delta = torch.nan_to_num(delta, nan=0.01, posinf=3.0, neginf=-3.0)
            h = torch.nan_to_num(h, nan=0.01, posinf=3.0, neginf=-3.0)


            # Adjust the parameters of the accretion disk
            iteration_torch = 1.0+torch.tensor(iteration, dtype=delta.dtype, device=delta.device)
            iteration_torch = iteration_torch.unsqueeze(0).unsqueeze(1).expand(delta.shape[0], 1)

            param_delta = torch.nan_to_num(self.param_delta(torch.cat((
                                                                    delta,
                                                                    context, 
                                                                    params_accretion, 
                                                                    gaussian_mixture_coefficients_not_normalized,
                                                                    self.normalized_mean(mean.squeeze(1)), 
                                                                    self.normalized_std(std.squeeze(1)),
                                                                    iteration_torch),
                                                                    dim=1)),
                                                                    nan=0.01, posinf=3.0, neginf=-3.0) 


            param_delta = param_delta / iteration_torch ** (1.0 + torch.nan_to_num(self.param_iteration_supression_net(torch.cat((delta, iteration_torch),dim=1)), nan=0.01, posinf=3.0, neginf=-3.0))
            
            params_accretion_delta = param_delta[:,:self.num_Gaussian_parameterization*self.n_params_accretion]
            params_accretion = params_accretion + params_accretion_delta

            # Gaussian mixture coefficients, they must be positive and sum to 1 (we use the softmax in the network)
            gaussian_mixture_coefficients_delta = param_delta[:,self.num_Gaussian_parameterization*self.n_params_accretion:]
            gaussian_mixture_coefficients_not_normalized = gaussian_mixture_coefficients_not_normalized + gaussian_mixture_coefficients_delta
            gaussian_mixture_coefficients = F.softmax(gaussian_mixture_coefficients_not_normalized, dim=1) # This is now normalized such that the sum of the coefficients is 1
            
            del param_delta, gaussian_mixture_coefficients_delta

            params_accretion_reshape = torch.zeros(params_accretion.shape[0], self.n_params_accretion, self.num_Gaussian_parameterization, dtype=dtype, device=device)
            for i in range(self.num_Gaussian_parameterization):
                params_accretion_reshape[:,:,i] = params_accretion[:,i*self.n_params_accretion:(i+1)*self.n_params_accretion]
            

            gaussian_mixture_coefficients_list.append(gaussian_mixture_coefficients.detach().cpu().numpy())

            z0_delta = torch.nan_to_num(self.z0_delta(torch.cat((delta,  
                                                                z0,
                                                                params_accretion, 
                                                                gaussian_mixture_coefficients,
                                                                context, 
                                                                self.normalized_mean(mean.squeeze(1)),
                                                                self.normalized_std(std.squeeze(1)),
                                                                iteration_torch),
                                                                dim=1)), 
                                                                nan=0.01, posinf=3.0, neginf=-3.0) 
            z0_delta = z0_delta / iteration_torch ** (1.0 + torch.nan_to_num(self.z0_iteration_supression_net(torch.cat((delta, iteration_torch), dim=1)), nan=0.01, posinf=3.0, neginf=-3.0))
            z0 = z0 + z0_delta
            del z0_delta
            # get the driving signal from the SDE
            driving, log_ratio = self.SDE(z0) # if logqp is False then log_ratio is zero
            driving = torch.nan_to_num(driving, nan=0.0, posinf=1.0, neginf=-1.0)
            driving = std[:,:,self.reference_band].unsqueeze(-1) * driving + mean[:,:,self.reference_band].unsqueeze(-1) # put the driving signal back in the original scale

            # Analyze_driving_signal to improve our variability parameter predictions
            if self.use_checkpoints:
                driving_info = checkpoint(self.analyze_driving_signal, driving)
            else:
                driving_info = self.analyze_driving_signal(driving)
            driving_info = torch.nan_to_num(driving_info, nan=0.01, posinf=3.0, neginf=-3.0)
            # driving_info ~ [B, 2*self.freq_segments+5]

            f_net_out = self.SDE.f(self.SDE.ts, z0)
            g_net_out = self.SDE.g(None, z0)
            if self.logqp:
                h_net_out = self.SDE.h(None, z0)
            else:
                h_net_out = None

            if self.logqp:
                params_variability = self.param_variability_net(torch.cat((driving_info,
                                                                        params_accretion,
                                                                        gaussian_mixture_coefficients,
                                                                        context, 
                                                                        f_net_out, 
                                                                        g_net_out,
                                                                        h_net_out,
                                                                        z0,
                                                                        self.normalized_mean(mean.squeeze(1)),
                                                                        self.normalized_std(std.squeeze(1))),
                                                                        dim=1))
            else:
                params_variability = self.param_variability_net(torch.cat((driving_info,
                                                                        params_accretion,
                                                                        gaussian_mixture_coefficients,
                                                                        context, 
                                                                        f_net_out,
                                                                        g_net_out,
                                                                        z0,
                                                                        self.normalized_mean(mean.squeeze(1)),
                                                                        self.normalized_std(std.squeeze(1))),
                                                                        dim=1))
                
            params_variability = torch.nan_to_num(params_variability, nan=0.01, posinf=3.0, neginf=-3.0)

            # Get the uncertainty on the parameter predictions (both the acression disk and variability parameters)
            if self.logqp:
                param_pred_L = self.L_params_net(torch.cat((params_accretion,  
                                                        gaussian_mixture_coefficients,
                                                        params_variability,
                                                        z0,
                                                        context,
                                                        driving_info,
                                                        f_net_out,
                                                        g_net_out,
                                                        h_net_out,
                                                        self.normalized_mean(mean.squeeze(1)),
                                                        self.normalized_std(std.squeeze(1))),
                                                        dim=1))
            else:
                param_pred_L = self.L_params_net(torch.cat((params_accretion,  
                                                            gaussian_mixture_coefficients,
                                                            params_variability,
                                                            z0,
                                                            context,
                                                            driving_info,
                                                            f_net_out,
                                                            g_net_out,
                                                            self.normalized_mean(mean.squeeze(1)),
                                                            self.normalized_std(std.squeeze(1))),
                                                            dim=1))
            
            del driving_info, f_net_out, g_net_out, h_net_out

            param_pred_L = torch.nan_to_num(param_pred_L, nan=0.01, posinf=3.0, neginf=-3.0)
            param_pred_L_output_size = param_pred_L.shape[1]//self.num_Gaussian_parameterization
                                                        
            params_variability_reshape = torch.zeros(params_variability.shape[0], self.n_params_variability, self.num_Gaussian_parameterization, dtype=dtype, device=device)
            param_pred_L_reshape = torch.zeros(param_pred_L.shape[0], param_pred_L_output_size, self.num_Gaussian_parameterization, dtype=dtype, device=device)
            for Gauss_indx in range(self.num_Gaussian_parameterization):
                params_variability_reshape[:,:,Gauss_indx] = params_variability[:,Gauss_indx*self.n_params_variability:(Gauss_indx+1)*self.n_params_variability]
                param_pred_L_reshape[:,:,Gauss_indx] = param_pred_L[:,Gauss_indx*param_pred_L_output_size:(Gauss_indx+1)*param_pred_L_output_size]

            # combined the parameter predictions to include both the accretion and variability parameters
            param_pred_reshape = torch.cat((params_accretion_reshape, params_variability_reshape), dim=1)

            # Reshape the L matrix to be [B, n_params, n_params, num_Gaussian_parameterization], the lower triangular matrix of the covariance matrix such that Sigma = L*L^T for each Gaussian.
            param_pred_L_matrix = torch.zeros(param_pred_L_reshape.shape[0], self.n_params, self.n_params, self.num_Gaussian_parameterization, dtype=dtype, device=device)
            for Gauss_indx in range(self.num_Gaussian_parameterization):
                param_pred_L_matrix[:, :, :, Gauss_indx] = self.fill_lower_triangular(param_pred_L_reshape[:,:,Gauss_indx])


            # Save the mean predictions and uncertainty of the parameters for each iteration to see how they change
            if save_iterations:

                param_pred_mean_list.append(param_pred_reshape.detach().cpu().numpy())
                param_pred_L_list.append(param_pred_L_matrix.detach().cpu().numpy())

            # Now we sample from the Gaussian mixture to get the mean parameters in the physical space
            samples = self.sample_mixture(self.num_samples, [MultivariateNormal(loc=param_pred_reshape[:,:,i], scale_tril=param_pred_L_matrix[:,:,:,i]) for i in range(self.num_Gaussian_parameterization)], gaussian_mixture_coefficients)
            samples = torch.nan_to_num(samples, nan=0.01, posinf=3.0, neginf=-3.0)
            params_mean = torch.mean(self.expit(samples), dim=0) # params_mean ~ [B, n_params], now in the range from 0 to 1

            del samples
            gc.collect()
            torch.cuda.empty_cache()

            # Generate kernels from the mean accretion disk parameters
            if self.use_checkpoints:
                # First reshape the parameters to be [B, num_Gaussian_parameterization, n_params_accretion]
                kernels = checkpoint(self.generate_tf, self.get_physical_params(params_mean), batch_fraction_sim)
            else:
                kernels = self.generate_tf(self.get_physical_params(params_mean), batch_fraction_sim=batch_fraction_sim)

            # Get mean time delays from kernels

            mean_time_pred = self.get_mean_time(kernels)
            mean_time_pred = torch.nan_to_num(mean_time_pred, nan=0.01, posinf=3.0, neginf=-3.0)

            time_delay = torch.clone(mean_time_pred)/self.kernel_num_days  # time_delay ~ [B, num_bands], between 0 and 1

            # If we want the relative time delay between bands or the absolute time delay
            if self.relative_mean_time:
                # Subtract the time from the reference band. Only the relative time delays are important.
                mean_time_pred = mean_time_pred - mean_time_pred[:, self.reference_band].unsqueeze(1)
                # Get rid of the reference band time delay, since it is always 0.0
                mean_time_pred = torch.cat((mean_time_pred[:,:self.reference_band], mean_time_pred[:,self.reference_band+1:]), dim=1)

            mean_time_pred_list.append(mean_time_pred.detach().cpu().numpy())

            stdev_kernel = self.get_stdev_kernel(kernels)/self.kernel_num_days # stdev_kernel ~ [B, num_bands], between 0 and 1
            stdev_kernel = torch.nan_to_num(stdev_kernel, nan=0.01, posinf=3.0, neginf=-3.0)
            # Get the uncertainty in the time delay measurements

            L_time_delay = self.time_delay_uncertainty_net(torch.cat((params_mean,  # n_params+context_size+4*num_bands
                                                                    time_delay,
                                                                    stdev_kernel,
                                                                    context, 
                                                                    self.normalized_mean(mean.squeeze(1)),
                                                                    self.normalized_std(std.squeeze(1))),
                                                                    dim=1))
            L_time_delay = torch.nan_to_num(L_time_delay, nan=0.01, posinf=3.0, neginf=-3.0)
            
            # Do the convolution and normalize the light curve
            if self.use_checkpoints:
                _xs = checkpoint(self.convolve, driving, kernels) # _xs ~ [B, T, num_bands]
            else:
                _xs = self.convolve(driving, kernels)
            _xs = _xs + (mean-mean[:,:,self.reference_band].unsqueeze(-1)) # add the mean of the reference_band band back to the light curve

            _xs = (_xs-mean)/std # normalize the light curve
            driving_reconstructed = (driving - mean[:,:,self.reference_band].unsqueeze(-1))/std[:,:,self.reference_band].unsqueeze(-1) # normalize the driving signal

            # Get the mean and standard deviation of the predicted light curve
            mean_LC_pred = torch.mean(_xs, dim=1)
            std_LC_pred = torch.std(_xs, dim=1)

            # Get the mean and standard deviation of the driving signal
            mean_driving_pred = torch.mean(driving_reconstructed[:, :self.extra_time_from_kernel], dim=1)
            std_driving_pred = torch.std(driving_reconstructed[:, :self.extra_time_from_kernel], dim=1)

            # bias and multiplicative term for the light curve
            bias, bias_flux, mult = self.bias_mult_net(torch.cat((
                                                    context, 
                                                    time_delay,
                                                    stdev_kernel,
                                                    params_mean,
                                                    mean_LC_pred,
                                                    std_LC_pred,
                                                    mean_driving_pred,
                                                    std_driving_pred,
                                                    self.normalized_mean(mean.squeeze(1)),
                                                    self.normalized_std(std.squeeze(1))),
                                                    dim=1)).chunk(3, dim=1)
            
            bias = torch.nan_to_num(bias, nan=0.01, posinf=1.0, neginf=-1.0)
            bias_flux = torch.nan_to_num(bias_flux, nan=0.001, posinf=0.001, neginf=0.001)
            mult = torch.nan_to_num(mult, nan=0.01, posinf=1.0, neginf=-1.0)

            del mean_LC_pred, std_LC_pred, params_mean

            bias_norm = bias.unsqueeze(1) # bias_norm ~ [B, 1, num_bands]
            bias_flux_norm = torch.sigmoid(bias_flux).unsqueeze(1) # bias_flux_norm ~ [B, 1, num_bands] # positive value between 0 and 1
            mult_norm = F.softplus(mult).unsqueeze(1) # mult_norm ~ [B, 1, num_bands] # positive values

            # apply the bias to the flux
            _xs, driving_reconstructed = self.apply_normalization(_xs, driving_reconstructed, mean, std, bias_norm, bias_flux_norm, mult_norm)
            
            # pad to add the extra kernel length again
            _xs_pad = torch.zeros(xs.shape[0], xs.shape[1]+padding_size, self.num_bands, dtype=dtype, device=device)
            _xs_pad[:,padding_size:] = _xs 

            mask = (xs[:,:,:self.out_dim] != 0.0).type_as(xs)

            # uncertainty estimation of the predicted light curve

            log_var = self.uncertainty_net(torch.cat((driving_reconstructed[:, padding_size:], xs, _xs, mask*(_xs-xs[:,:,:self.out_dim])/(xs[:,:,self.out_dim:]+self.epsilon)), dim=2))
            log_var = torch.nan_to_num(log_var, nan=0.01, posinf=3.0, neginf=-3.0)

            _xs = torch.cat((_xs, log_var[:, :, 1:]), dim=2) # _xs ~ [B, T, 2*num_bands]
            driving_reconstructed = torch.cat((driving_reconstructed[:,padding_size:], log_var[:, :, 0:1]), dim=2)
            del log_var

            # unnormlize the observed light curve
            xs[:,:,:self.out_dim] = mask*(xs[:,:,:self.out_dim]*std + mean)
            xs[:,:,self.out_dim:] = mask*(xs[:,:,self.out_dim:]*std)

            # unnormlize the predicted light curve
            _xs[:,:,:self.out_dim] = _xs[:,:,:self.out_dim]*std + mean
            _xs[:,:,self.out_dim:] = _xs[:,:,self.out_dim:]+2*torch.log(std) # log(a^2std^2) = log(a^2)+log(b**2) , b > 0

            # unnormalize the driving signal
            driving_reconstructed[:,:,0] = driving_reconstructed[:,:,0]*std[:,:,self.reference_band] + mean[:,:,self.reference_band]
            driving_reconstructed[:,:,1] = driving_reconstructed[:,:,1]+2*torch.log(std[:,:,self.reference_band])

            if save_iterations:
                # save the transfer function for each iteration. We normalize it to be a probability distribution.
                kernel_list.append(kernels.detach().cpu().numpy())
                # unnormalize and save the predicted light curve
                _xs_list.append((_xs).detach().cpu().numpy())
                # save the driving signal for each iteration
                driving_list.append(driving.detach().cpu().numpy())

            # Convert the array of L to a lower triangular matrix
            L_time_delay = self.fill_lower_triangular(L_time_delay)
            L_time_delay = torch.nan_to_num(L_time_delay, nan=0.01, posinf=3.0, neginf=-3.0)

            # if true_LC is not None, we compute the loss. Otherwise, we just 0.0.
            if true_LC is not None:

                # Minimize the negative Gaussian log-likelihood of the predicted light curve vs the true light curve
                # Only include observed bands in the loss function by using mean_mask

                log_pxs = torch.mean(mean_mask * (0.5*(((true_LC-_xs[:,:,:self.out_dim])**2)/torch.exp(_xs[:,:,self.out_dim:])+_xs[:,:,self.out_dim:]+np.log(2*np.pi)))) / torch.mean(mean_mask)

                log_pxs_driving = 0.5*torch.mean((driving_true[:, padding_size:].squeeze(-1)-driving_reconstructed[:,:,0])**2/(torch.exp(driving_reconstructed[:,:,1])+self.epsilon)+driving_reconstructed[:,:,1]+np.log(2*np.pi))

                # Ensure that the predicted mean is close to the context points
                log_pxs2 = 0.5*torch.sum(mask*(((true_LC-_xs[:,:,:self.out_dim])**2)/(xs[:,:,self.out_dim:]**2+self.log_pxs2_leeway**2)))/torch.sum(mask)

                time_delay_dist = MultivariateNormal(loc=mean_time_pred, scale_tril=L_time_delay)
                time_delay_loss = -time_delay_dist.log_prob(mean_time_true).mean()/self.num_bands
                
                # Initialize the log probabilities
                log_probs = []
                for Gauss_indx in range(self.num_Gaussian_parameterization):
                    param_dist = MultivariateNormal(loc=param_pred_reshape[:, :, Gauss_indx], scale_tril=param_pred_L_matrix[:, :, :, Gauss_indx])
                    
                    # Calculate the log probability of the current Gaussian component
                    log_prob = param_dist.log_prob(param_true)
                    
                    # Add the weighted log probability to the list
                    weighted_log_prob = log_prob + torch.log(gaussian_mixture_coefficients[:, Gauss_indx])
                    log_probs.append(weighted_log_prob)

                # Stack the log probabilities along the Gaussian component axis
                log_probs = torch.stack(log_probs, dim=1)

                # Apply the log-sum-exp trick to calculate the log-sum of exponentiated values
                log_sum_exp = torch.logsumexp(log_probs, dim=1)

                # Compute the NLL by taking the negative log-sum-exp value and scaling it
                param_loss = -log_sum_exp.mean() / self.n_params
                del log_probs, log_sum_exp, weighted_log_prob


                if print_metrics:
                    print(f"iteration: {iteration}:, log_pxs: {log_pxs.item():.4f}, log_pxs_driving: {log_pxs_driving.item():.4f}, log_pxs2: {log_pxs2.item():.4f}, param_loss: {param_loss.item():.4f}, time_delay_loss: {time_delay_loss.item():.4f}")

                loss_val = self.log_pxs_weight * (log_pxs + log_pxs_driving/self.num_bands) + self.log_pxs2_weight * log_pxs2 + log_ratio + self.linear_anneal_weight(self.param_anneal_epochs)*(self.param_loss_weight*param_loss + self.time_delay_loss_weight*time_delay_loss)
                loss_list.append(loss_val.item())
                loss += loss_val * (iteration+1) # Each iteration is weighted by the iteration number to give more importance to the last iterations
                
                del loss_val, log_pxs, log_pxs_driving, log_pxs2, log_ratio, param_loss, time_delay_loss, param_dist, time_delay_dist
                # run garbage collector to free up memory just in case
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # if true_LC is None, we just set the loss to 0.0, this is only used during inference
                loss = 0.0

        # Average across the number of iteratiosns
        #loss = loss/self.num_iterations
        loss = loss / np.sum(np.arange(self.num_iterations)+1)

        # Saved _xs and kernels into numpy arrays of shape [num_iterations, B, T, num_bands] and [num_iterations, B, kernel_size, num_bands] respectively
        if save_iterations:
            _xs_list = np.stack(_xs_list, axis=0)
            kernel_list = np.stack(kernel_list, axis=0)
            driving_list = np.stack(driving_list, axis=0)

            param_pred_mean_list = np.stack(param_pred_mean_list, axis=0)
            param_pred_L_list = np.stack(param_pred_L_list, axis=0)

            gaussian_mixture_coefficients_list = np.stack(gaussian_mixture_coefficients_list, axis=0) # shape [num_iterations, B, num_Gaussian_parameterization]

        # We put all the outputs in a dictionary because there are so many of them. Just to have things better organized.
        output_dict = dict()
        output_dict['loss'] = loss
        output_dict['kernels'] = kernels
        output_dict['_xs'] = _xs
        output_dict['driving'] = driving
        output_dict['driving_reconstructed'] = driving_reconstructed
        output_dict['z0'] = z0
        output_dict['param_pred_reshape'] = param_pred_reshape
        output_dict['param_pred_L_matrix'] = param_pred_L_matrix
        output_dict['gaussian_mixture_coefficients'] = gaussian_mixture_coefficients
        output_dict['mean_time_pred'] = mean_time_pred
        output_dict['L_time_delay'] = L_time_delay
        output_dict['_xs_list'] = _xs_list
        output_dict['kernel_list'] = kernel_list
        output_dict['driving_list'] = driving_list
        output_dict['param_pred_mean_list'] = param_pred_mean_list
        output_dict['param_pred_L_list'] = param_pred_L_list
        output_dict['gaussian_mixture_coefficients_list'] = gaussian_mixture_coefficients_list
        output_dict['mean_time_pred_list'] = mean_time_pred_list
        output_dict['loss_list'] = loss_list
        output_dict['mean_mask'] = mean_mask

        return output_dict

    @torch.no_grad()
    def predict(self, xs, save_iterations = True, redshift=None, batch_fraction_sim=1.0):
        """
        This function can be used to predict the light curve and transfer function for a given light curve.
        Uses the forward function to do this but without calculating the loss.
        No gradients are computed since this is only used during inference.

        xs: [B, T, num_bands], light curve
        save_iterations: bool, whether to save the transfer function for each iteration
        redshift: float, redshift of the lens. If None the redshift is not used. (must also not be used during training)
        batch_fraction_sim: float, fraction of the batch that is used by the transfer function sim at a time to save GPU memory.

        returns output_dict containing many outputs
        """
        return self.forward(xs, save_iterations=save_iterations, redshift=redshift, batch_fraction_sim=batch_fraction_sim)
    
    # Only use if logqp is True
    @torch.no_grad()
    def sample_kernel(self, param_pred, batch_fraction_sim=1.0):
        """
        This function can be used to sample from the kernel network with a latent mean and log std.
        No gradients are computed since this is only used during inference.

        param_pred: [B, n_params], the predicted parameters of the accretion disk
        batch_fraction_sim: float, fraction of the batch that is used by the transfer function sim at a time to save GPU memory.

        returns kernels: [B, kernel_size, num_bands], the sampled kernels
        """
        params_accretion = param_pred[:,:self.n_params_accretion]

        kernels = self.generate_tf(self.get_physical_params(params_accretion), batch_fraction_sim=batch_fraction_sim)
        return kernels
