"""
VP-RNN
-------
This file contains the VP-RNN and MOVP-RNN specifications. In particular, we implement:
(1) Emitter
    Parametrizes the conditional output distribution p(x_t | \lambda_t) in the generative model (Eq.11, Section 3.2.4)
(2) Encoder:
    Parametrizes encoder network p(\lambda_t | h^q_t) for posterior inference (Eq.11, Section 3.2.4)
(3) VPRNN:
    (Single output) Variational Poisson-RNN (Section 3.2.4)
(4) MOVPRR:
    Multi-Output Variational Poisson-RNN (Section 4.1)
"""

import torch
from torch import nn

# pyro imports
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam, ClippedAdam
import pyro.distributions as dist
import pyro.poutine as poutine
from src.misc.utils import Trace_ELBO_Wrapper

class Emitter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Emitter, self).__init__()
        # initialize linear transformations
        self.lin_input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, output_dim)
        
        # initialize non-linearities
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        h = self.relu(self.lin_input_to_hidden(x))
        h = self.dropout(h)
        h = self.relu(self.lin_hidden_to_hidden(h))
        h = self.dropout(h)
        loc = self.lin_hidden_to_loc(h)
        scale = self.softplus(self.lin_hidden_to_scale(h))
        return loc, scale

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        # initialize linear transformations
        self.lin_input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, output_dim)
        
        # initialize non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        h = self.relu(self.lin_input_to_hidden(x))
        h = self.relu(self.lin_hidden_to_hidden(h))
        loc = self.lin_hidden_to_loc(h)
        scale = self.softplus(self.lin_hidden_to_scale(h))
        return loc, scale

class VPRNN(nn.Module):
    def __init__(self, input_dim=32, output_dim=1, p_model_dim=128, p_model_layers=1, 
                q_model_dim=128, q_model_layers=1, use_cuda=False, verbose=False):
        super(VPRNN, self).__init__()
        # initialize modules
        self.emitter = Emitter(p_model_dim, 32, output_dim)
        self.encoder = Encoder(q_model_dim, 32, output_dim)
        self.p_model = nn.GRU(input_size=input_dim + output_dim, hidden_size=p_model_dim, 
                              num_layers=p_model_layers, batch_first=True, bidirectional=False)
        self.q_model = nn.GRU(input_size=input_dim + output_dim, hidden_size=q_model_dim, 
                              num_layers=q_model_layers, batch_first=True, bidirectional=False)
        
        # initialize learnable initial hidden states
        self.h_0 = nn.Parameter(torch.zeros(p_model_dim))
        self.q_h_0 = nn.Parameter(torch.zeros(q_model_dim))
        
        self.use_cuda = use_cuda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p_model_dim = p_model_dim
        self.q_model_dim = q_model_dim
        self.verbose = verbose
        
        if self.use_cuda:
            self.cuda()
    
    def model(self, X=None, y=None, forecast=False):
        # get input shapes
        X = X[1:]
        T_max, D = X.shape[0], X.shape[1]
        
        # register parameters
        pyro.module("model", self)
        
        b = 1
        
        # initialize p_model hidden state
        h_prev = self.h_0.expand(b, self.h_0.size(0)).view(1, b, -1).contiguous()
        
        # initialize tensors to store results
        lambdas = torch.zeros((b, T_max, self.output_dim))
        x_samples = torch.zeros((b, T_max, self.output_dim))
        
        # extract feature embedding
        X_embedded = X

        # propagate p_model over time
        p_model_input = torch.cat((X_embedded.view(b, T_max, self.input_dim),
                                   y[:-1].view(b, T_max, self.output_dim)), dim=2)
        hidden_1_T, _ = self.p_model(p_model_input, h_prev)
        hidden_1_T = hidden_1_T.view(T_max, self.p_model_dim)
        
        # get mean and st.dev of (log) rate
        log_lambda_loc, log_lambda_scale = self.emitter(hidden_1_T)
        assert log_lambda_loc.shape == (T_max, self.output_dim)
        
        with pyro.plate("data", T_max):
            # sample lambda ~ N(lambda|mu(x), sigma(x))
            log_lambda = pyro.sample("log_lambda", dist.Normal(log_lambda_loc, log_lambda_scale).to_event(1))
            lambdas[0] = torch.exp(log_lambda)
            
            # sample observations y ~ Poisson(exp(log_lambda))
            if forecast:
                obs = pyro.sample("obs", dist.Poisson(torch.exp(log_lambda)).to_event(1), obs=None)
            else:
                obs = pyro.sample("obs", dist.Poisson(torch.exp(log_lambda)).to_event(1), obs=y[1:, :])
        return lambdas, obs
        
    def guide(self, X=None, y=None, forecast=False):
        # get input shapes
        X = X[1:]
        T_max, D = X.shape[0], X.shape[1]
        
        # register parameters
        pyro.module("model", self)
        
        b = 1
        
        # initialize p_model hidden state
        q_h_prev = self.q_h_0.view(1, b, self.q_h_0.size(-1)).contiguous()
        
        
        # extract feature embedding
        X_embedded = X

        # propagate p_model over time
        q_model_input = torch.cat((X_embedded.view(b, T_max, self.input_dim),
                                   y[:-1].view(b, T_max, self.output_dim)), dim=2)
        q_hidden_1_T, _ = self.q_model(q_model_input, q_h_prev)
        q_hidden_1_T = q_hidden_1_T.view(T_max, self.q_model_dim)
        
        # get mean and st.dev of (log) rate
        log_lambda_loc, log_lambda_scale = self.encoder(q_hidden_1_T)
        assert log_lambda_loc.shape == (T_max, self.output_dim)
        
        with pyro.plate("data", T_max):
            # sample lambda ~ N(lambda|mu(x), sigma(x))
            q_dist = dist.Normal(log_lambda_loc, log_lambda_scale)
            log_lambda = pyro.sample("log_lambda", q_dist.to_event(1))
            
        return log_lambda_loc, log_lambda_scale
    
    def _get_log_likelihood(self, X, y):
        trace_elbo = Trace_ELBO_Wrapper(num_particles=1)
        for model_trace, _ in trace_elbo._get_traces(self.model, self.guide, [X, y, True], {}):
            ll = -model_trace.nodes["obs"]["log_prob_sum"]
        return ll
    
class MOVPRNN(nn.Module):
    def __init__(self, input_dim=32, output_dim=3, p_model_dim=128, p_model_layers=1, 
                q_model_dim=128, q_model_layers=1, use_cuda=False, verbose=False):
        super(MOVPRNN, self).__init__()
        # initialize modules
        self.emitter = Emitter(p_model_dim, 32, output_dim)
        self.encoder = Encoder(q_model_dim, 32, output_dim)
        self.p_model = nn.GRU(input_size=input_dim + output_dim, hidden_size=p_model_dim, 
                              num_layers=p_model_layers, batch_first=True, bidirectional=False)
        self.q_model = nn.GRU(input_size=input_dim + output_dim, hidden_size=q_model_dim, 
                              num_layers=q_model_layers, batch_first=True, bidirectional=False)
        
        # initialize learnable initial hidden states
        self.h_0 = nn.Parameter(torch.zeros(p_model_dim))
        self.q_h_0 = nn.Parameter(torch.zeros(q_model_dim))
        
        self.use_cuda = use_cuda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p_model_dim = p_model_dim
        self.q_model_dim = q_model_dim
        self.verbose = verbose
        
        if self.use_cuda:
            self.cuda()
    
    def model(self, X=None, y=None, forecast=False):
        # get input shapes
        X = X[1:]
        T_max, D = X.shape[0], X.shape[1]
        
        # register parameters
        pyro.module("model", self)
        
        b = 1
        
        # initialize p_model hidden state
        h_prev = self.h_0.expand(b, self.h_0.size(0)).view(1, b, -1).contiguous()
        
        # initialize tensors to store results
        lambdas = torch.zeros((b, T_max, self.output_dim))
        x_samples = torch.zeros((b, T_max, self.output_dim))
        
        # extract feature embedding
        X_embedded = X

        # propagate p_model over time
        p_model_input = torch.cat((X_embedded.view(b, T_max, self.input_dim),
                                   y[:-1].view(b, T_max, self.output_dim)), dim=2)
        hidden_1_T, _ = self.p_model(p_model_input, h_prev)
        hidden_1_T = hidden_1_T.view(T_max, self.p_model_dim)
        
        # get mean and st.dev of (log) rate
        log_lambda_loc, log_lambda_scale = self.emitter(hidden_1_T)
        assert log_lambda_loc.shape == (T_max, self.output_dim)
        
        with pyro.plate("data", T_max):
            # sample lambda ~ N(lambda|mu(x), sigma(x))
            log_lambda = pyro.sample("log_lambda", dist.Normal(log_lambda_loc, log_lambda_scale).to_event(1))
            lambdas[0] = torch.exp(log_lambda[:, :2])
            
            # sample observations y ~ Poisson(exp(log_lambda))
            if forecast:
                obs = pyro.sample("obs", dist.Poisson(torch.exp(log_lambda[:, :2])).to_event(1), obs=None)
            else:
                obs = pyro.sample("obs", dist.Poisson(torch.exp(log_lambda[:, :2])).to_event(1), obs=y[1:, :2])
        return lambdas, obs
        
    def guide(self, X=None, y=None, forecast=False):
        # get input shapes
        X = X[1:]
        T_max, D = X.shape[0], X.shape[1]
        
        # register parameters
        pyro.module("model", self)
        
        b = 1
        
        # initialize p_model hidden state
        q_h_prev = self.q_h_0.view(1, b, self.q_h_0.size(-1)).contiguous()
        
        
        # extract feature embedding
        X_embedded = X

        # propagate p_model over time
        q_model_input = torch.cat((X_embedded.view(b, T_max, self.input_dim),
                                   y[:-1].view(b, T_max, self.output_dim)), dim=2)
        q_hidden_1_T, _ = self.q_model(q_model_input, q_h_prev)
        q_hidden_1_T = q_hidden_1_T.view(T_max, self.q_model_dim)
        
        # get mean and st.dev of (log) rate
        log_lambda_loc, log_lambda_scale = self.encoder(q_hidden_1_T)
        assert log_lambda_loc.shape == (T_max, self.output_dim)
        
        with pyro.plate("data", T_max):
            # sample lambda ~ N(lambda|mu(x), sigma(x))
            q_dist = dist.Normal(log_lambda_loc, log_lambda_scale)
            log_lambda = pyro.sample("log_lambda", q_dist.to_event(1))
            
        return log_lambda_loc, log_lambda_scale
    
    def _get_log_likelihood(self, X, y):
        trace_elbo = Trace_ELBO_Wrapper(num_particles=1)
        for model_trace, _ in trace_elbo._get_traces(self.model, self.guide, [X, y, True], {}):
            ll = -model_trace.nodes["obs"]["log_prob_sum"]
        return ll