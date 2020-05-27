##########################################################
# pytorch-kaldi v.0.1
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from distutils.util import strtobool
import math
import json

# uncomment below if you want to use SRU
# and you need to install SRU: pip install sru[cuda].
# or you can install it from source code: https://github.com/taolei87/sru.
# import sru


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def act_fun(act_type):

    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!


class MLP(nn.Module):
    def __init__(self, options, inp_dim):
        super(MLP, self).__init__()

        self.input_dim = inp_dim
        self.dnn_lay = list(map(int, options["dnn_lay"].split(",")))
        self.dnn_drop = list(map(float, options["dnn_drop"].split(",")))
        self.dnn_use_batchnorm = list(map(strtobool, options["dnn_use_batchnorm"].split(",")))
        self.dnn_use_laynorm = list(map(strtobool, options["dnn_use_laynorm"].split(",")))
        self.dnn_use_laynorm_inp = strtobool(options["dnn_use_laynorm_inp"])
        self.dnn_use_batchnorm_inp = strtobool(options["dnn_use_batchnorm_inp"])
        self.dnn_act = options["dnn_act"].split(",")

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.dnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.dnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_dnn_lay = len(self.dnn_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_dnn_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.dnn_drop[i]))

            # activation
            self.act.append(act_fun(self.dnn_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.dnn_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.dnn_lay[i], momentum=0.05))

            if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.dnn_lay[i], current_input).uniform_(
                    -np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                    np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                )
            )
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]))

            current_input = self.dnn_lay[i]

        self.out_dim = current_input

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.dnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.dnn_use_batchnorm_inp):

            x = self.bn0((x))

        for i in range(self.N_dnn_lay):

            if self.dnn_use_laynorm[i] and not (self.dnn_use_batchnorm[i]):
                x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] and not (self.dnn_use_laynorm[i]):
                x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](x)))))

            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](self.wx[i](x)))

        return x


class LSTM_cudnn(nn.Module):
    def __init__(self, options, inp_dim):
        super(LSTM_cudnn, self).__init__()

        self.input_dim = inp_dim
        self.hidden_size = int(options["hidden_size"])
        self.num_layers = int(options["num_layers"])
        self.bias = bool(strtobool(options["bias"]))
        self.batch_first = bool(strtobool(options["batch_first"]))
        self.dropout = float(options["dropout"])
        self.bidirectional = bool(strtobool(options["bidirectional"]))

        self.lstm = nn.ModuleList(
            [
                nn.LSTM(
                    self.input_dim,
                    self.hidden_size,
                    self.num_layers,
                    bias=self.bias,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            ]
        )

        for name,param in self.lstm[0].named_parameters():
            if 'weight_hh' in name:
                if self.batch_first:
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):

        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
            c0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)

        if x.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        output, (hn, cn) = self.lstm[0](x, (h0, c0))

        return output


class GRU_cudnn(nn.Module):
    def __init__(self, options, inp_dim):
        super(GRU_cudnn, self).__init__()

        self.input_dim = inp_dim
        self.hidden_size = int(options["hidden_size"])
        self.num_layers = int(options["num_layers"])
        self.bias = bool(strtobool(options["bias"]))
        self.batch_first = bool(strtobool(options["batch_first"]))
        self.dropout = float(options["dropout"])
        self.bidirectional = bool(strtobool(options["bidirectional"]))

        self.gru = nn.ModuleList(
            [
                nn.GRU(
                    self.input_dim,
                    self.hidden_size,
                    self.num_layers,
                    bias=self.bias,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            ]
        )

        for name,param in self.gru[0].named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):

        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)

        if x.is_cuda:
            h0 = h0.cuda()

        output, hn = self.gru[0](x, h0)

        return output


class RNN_cudnn(nn.Module):
    def __init__(self, options, inp_dim):
        super(RNN_cudnn, self).__init__()

        self.input_dim = inp_dim
        self.hidden_size = int(options["hidden_size"])
        self.num_layers = int(options["num_layers"])
        self.nonlinearity = options["nonlinearity"]
        self.bias = bool(strtobool(options["bias"]))
        self.batch_first = bool(strtobool(options["batch_first"]))
        self.dropout = float(options["dropout"])
        self.bidirectional = bool(strtobool(options["bidirectional"]))

        self.rnn = nn.ModuleList(
            [
                nn.RNN(
                    self.input_dim,
                    self.hidden_size,
                    self.num_layers,
                    nonlinearity=self.nonlinearity,
                    bias=self.bias,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            ]
        )

        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):

        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)

        if x.is_cuda:
            h0 = h0.cuda()

        output, hn = self.rnn[0](x, h0)

        return output


class LSTM(nn.Module):
    def __init__(self, options, inp_dim):
        super(LSTM, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.lstm_lay = list(map(int, options["lstm_lay"].split(",")))
        self.lstm_drop = list(map(float, options["lstm_drop"].split(",")))
        self.lstm_use_batchnorm = list(map(strtobool, options["lstm_use_batchnorm"].split(",")))
        self.lstm_use_laynorm = list(map(strtobool, options["lstm_use_laynorm"].split(",")))
        self.lstm_use_laynorm_inp = strtobool(options["lstm_use_laynorm_inp"])
        self.lstm_use_batchnorm_inp = strtobool(options["lstm_use_batchnorm_inp"])
        self.lstm_act = options["lstm_act"].split(",")
        self.lstm_orthinit = strtobool(options["lstm_orthinit"])

        self.bidir = strtobool(options["lstm_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wfx = nn.ModuleList([])  # Forget
        self.ufh = nn.ModuleList([])  # Forget

        self.wix = nn.ModuleList([])  # Input
        self.uih = nn.ModuleList([])  # Input

        self.wox = nn.ModuleList([])  # Output
        self.uoh = nn.ModuleList([])  # Output

        self.wcx = nn.ModuleList([])  # Cell state
        self.uch = nn.ModuleList([])  # Cell state

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wfx = nn.ModuleList([])  # Batch Norm
        self.bn_wix = nn.ModuleList([])  # Batch Norm
        self.bn_wox = nn.ModuleList([])  # Batch Norm
        self.bn_wcx = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.lstm_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.lstm_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_lstm_lay = len(self.lstm_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_lstm_lay):

            # Activations
            self.act.append(act_fun(self.lstm_act[i]))

            add_bias = True

            if self.lstm_use_laynorm[i] or self.lstm_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wfx.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wix.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wox.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wcx.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))

            # Recurrent connections
            self.ufh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))
            self.uih.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))
            self.uoh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))
            self.uch.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))

            if self.lstm_orthinit:
                nn.init.orthogonal_(self.ufh[i].weight)
                nn.init.orthogonal_(self.uih[i].weight)
                nn.init.orthogonal_(self.uoh[i].weight)
                nn.init.orthogonal_(self.uch[i].weight)

            # batch norm initialization
            self.bn_wfx.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wix.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wox.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wcx.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.lstm_lay[i]))

            if self.bidir:
                current_input = 2 * self.lstm_lay[i]
            else:
                current_input = self.lstm_lay[i]

        self.out_dim = self.lstm_lay[i] + self.bidir * self.lstm_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.lstm_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.lstm_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_lstm_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.lstm_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.lstm_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.lstm_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.lstm_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wfx_out = self.wfx[i](x)
            wix_out = self.wix[i](x)
            wox_out = self.wox[i](x)
            wcx_out = self.wcx[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.lstm_use_batchnorm[i]:

                wfx_out_bn = self.bn_wfx[i](wfx_out.view(wfx_out.shape[0] * wfx_out.shape[1], wfx_out.shape[2]))
                wfx_out = wfx_out_bn.view(wfx_out.shape[0], wfx_out.shape[1], wfx_out.shape[2])

                wix_out_bn = self.bn_wix[i](wix_out.view(wix_out.shape[0] * wix_out.shape[1], wix_out.shape[2]))
                wix_out = wix_out_bn.view(wix_out.shape[0], wix_out.shape[1], wix_out.shape[2])

                wox_out_bn = self.bn_wox[i](wox_out.view(wox_out.shape[0] * wox_out.shape[1], wox_out.shape[2]))
                wox_out = wox_out_bn.view(wox_out.shape[0], wox_out.shape[1], wox_out.shape[2])

                wcx_out_bn = self.bn_wcx[i](wcx_out.view(wcx_out.shape[0] * wcx_out.shape[1], wcx_out.shape[2]))
                wcx_out = wcx_out_bn.view(wcx_out.shape[0], wcx_out.shape[1], wcx_out.shape[2])

            # Processing time steps
            hiddens = []
            ct = h_init
            ht = h_init

            for k in range(x.shape[0]):

                # LSTM equations
                ft = torch.sigmoid(wfx_out[k] + self.ufh[i](ht))
                it = torch.sigmoid(wix_out[k] + self.uih[i](ht))
                ot = torch.sigmoid(wox_out[k] + self.uoh[i](ht))
                ct = it * self.act[i](wcx_out[k] + self.uch[i](ht)) * drop_mask + ft * ct
                ht = ot * self.act[i](ct)

                if self.lstm_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return x


class GRU(nn.Module):
    def __init__(self, options, inp_dim):
        super(GRU, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.gru_lay = list(map(int, options["gru_lay"].split(",")))
        self.gru_drop = list(map(float, options["gru_drop"].split(",")))
        self.gru_use_batchnorm = list(map(strtobool, options["gru_use_batchnorm"].split(",")))
        self.gru_use_laynorm = list(map(strtobool, options["gru_use_laynorm"].split(",")))
        self.gru_use_laynorm_inp = strtobool(options["gru_use_laynorm_inp"])
        self.gru_use_batchnorm_inp = strtobool(options["gru_use_batchnorm_inp"])
        self.gru_orthinit = strtobool(options["gru_orthinit"])
        self.gru_act = options["gru_act"].split(",")
        self.bidir = strtobool(options["gru_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.wr = nn.ModuleList([])  # Reset Gate
        self.ur = nn.ModuleList([])  # Reset Gate

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm
        self.bn_wz = nn.ModuleList([])  # Batch Norm
        self.bn_wr = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.gru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.gru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_gru_lay = len(self.gru_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_gru_lay):

            # Activations
            self.act.append(act_fun(self.gru_act[i]))

            add_bias = True

            if self.gru_use_laynorm[i] or self.gru_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))
            self.wr.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))
            self.ur.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))

            if self.gru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
                nn.init.orthogonal_(self.ur[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))
            self.bn_wr.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.gru_lay[i]))

            if self.bidir:
                current_input = 2 * self.gru_lay[i]
            else:
                current_input = self.gru_lay[i]

        self.out_dim = self.gru_lay[i] + self.bidir * self.gru_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.gru_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.gru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_gru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.gru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.gru_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.gru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.gru_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            wr_out = self.wr[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.gru_use_batchnorm[i]:

                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])

                wr_out_bn = self.bn_wr[i](wr_out.view(wr_out.shape[0] * wr_out.shape[1], wr_out.shape[2]))
                wr_out = wr_out_bn.view(wr_out.shape[0], wr_out.shape[1], wr_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # gru equation
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                rt = torch.sigmoid(wr_out[k] + self.ur[i](ht))
                at = wh_out[k] + self.uh[i](rt * ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand

                if self.gru_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return x


class logMelFb(nn.Module):
    def __init__(self, options, inp_dim):
        super(logMelFb, self).__init__()
        import torchaudio

        self._sample_rate = int(options["logmelfb_nr_sample_rate"])
        self._nr_of_filters = int(options["logmelfb_nr_filt"])
        self._stft_window_size = int(options["logmelfb_stft_window_size"])
        self._stft_window_shift = int(options["logmelfb_stft_window_shift"])
        self._use_cuda = strtobool(options["use_cuda"])
        self.out_dim = self._nr_of_filters
        self._mspec = torchaudio.transforms.MelSpectrogram(
            sr=self._sample_rate,
            n_fft=self._stft_window_size,
            ws=self._stft_window_size,
            hop=self._stft_window_shift,
            n_mels=self._nr_of_filters,
        )

    def forward(self, x):
        def _safe_log(inp, epsilon=1e-20):
            eps = torch.FloatTensor([epsilon])
            if self._use_cuda:
                eps = eps.cuda()
            log_inp = torch.log10(torch.max(inp, eps.expand_as(inp)))
            return log_inp

        assert x.shape[-1] == 1, "Multi channel time signal processing not suppored yet"
        x_reshape_for_stft = torch.squeeze(x, -1).transpose(0, 1)
        if self._use_cuda:
            window = self._mspec.window(self._stft_window_size).cuda()
        else:
            window = self._mspec.window(self._stft_window_size)
        x_stft = torch.stft(
            x_reshape_for_stft, self._stft_window_size, hop_length=self._stft_window_shift, center=False, window=window
        )
        x_power_stft = x_stft.pow(2).sum(-1)
        x_power_stft_reshape_for_filterbank_mult = x_power_stft.transpose(1, 2)
        mel_spec = self._mspec.fm(x_power_stft_reshape_for_filterbank_mult).transpose(0, 1)
        log_mel_spec = _safe_log(mel_spec)
        out = log_mel_spec
        return out


class channel_averaging(nn.Module):
    def __init__(self, options, inp_dim):
        super(channel_averaging, self).__init__()
        self._use_cuda = strtobool(options["use_cuda"])
        channel_weights = [float(e) for e in options["chAvg_channelWeights"].split(",")]
        self._nr_of_channels = len(channel_weights)
        numpy_weights = np.asarray(channel_weights, dtype=np.float32) * 1.0 / np.sum(channel_weights)
        self._weights = torch.from_numpy(numpy_weights)
        if self._use_cuda:
            self._weights = self._weights.cuda()
        self.out_dim = 1

    def forward(self, x):
        assert self._nr_of_channels == x.shape[-1]
        out = torch.einsum("tbc,c->tb", x, self._weights).unsqueeze(-1)
        return out

class fusionRNN_jit(torch.jit.ScriptModule):
    def __init__(self, options, inp_dim):
        super(fusionRNN_jit, self).__init__()

        # Reading parameters
        input_size = inp_dim
        hidden_size = list(map(int, options["fusionRNN_lay"].split(",")))[0]
        dropout = list(map(float, options["fusionRNN_drop"].split(",")))[0]
        num_layers = len(list(map(int, options["fusionRNN_lay"].split(","))))
        batch_size = int(options["batches"])
        self.do_fusion = map(strtobool, options["fusionRNN_do_fusion"].split(","))
        self.act = str(options["fusionRNN_fusion_act"])
        self.reduce = str(options["fusionRNN_fusion_reduce"])
        self.fusion_layer_size = int(options["fusionRNN_fusion_layer_size"])
        self.to_do = options["to_do"]
        self.number_of_mic = int(options["fusionRNN_number_of_mic"])
        self.save_mic = self.number_of_mic
        bidirectional = True

        self.out_dim = 2 * hidden_size

        current_dim = int(input_size)

        self.model = torch.nn.ModuleList([])

        if self.to_do == "train":
            self.training = True
        else:
            self.training = False

        for i in range(num_layers):
            rnn_lay = liGRU_layer(
                current_dim,
                hidden_size,
                num_layers,
                batch_size,
                dropout=dropout,
                bidirectional=bidirectional,
                device="cuda",
                do_fusion=self.do_fusion,
                fusion_layer_size=self.fusion_layer_size,
                number_of_mic=self.number_of_mic,
                act=self.act,
                reduce=self.reduce
            )
            if i == 0:

                if self.do_fusion:
                    if bidirectional:
                        current_dim = (self.fusion_layer_size // self.save_mic) * 2
                    else:
                        current_dim = self.fusion_layer_size // self.save_mic
                    #We need to reset the number of mic for the next layers so it is divided by 1
                    self.number_of_mic = 1
                else:
                    if bidirectional:
                        current_dim = hidden_size * 2
                    else:
                        current_dim = hidden_size
                self.do_fusion = False # DO NOT APPLY FUSION ON THE NEXT LAYERS
            else:
                if bidirectional:
                    current_dim = hidden_size * 2
                else:
                    current_dim == hidden_size
            self.model.append(rnn_lay)


    @torch.jit.script_method
    def forward(self, x):
        # type: (Tensor) -> Tensor

        for ligru_lay in self.model:

            x = ligru_lay(x)

        return x


class liGRU_layer(torch.jit.ScriptModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        nonlinearity="relu",
        bidirectional=True,
        device="cuda",
        do_fusion=False,
        fusion_layer_size=64,
        number_of_mic=1,
        act="relu",
        reduce="mean",
    ):

        super(liGRU_layer, self).__init__()

        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device
        self.do_fusion = do_fusion
        self.fusion_layer_size = fusion_layer_size
        self.number_of_mic = number_of_mic
        self.act = act
        self.reduce = reduce

        if self.do_fusion:
            self.hidden_size = self.fusion_layer_size //  self.number_of_mic

        if self.do_fusion:
            self.wz = FusionLinearConv(
                self.input_size, self.hidden_size, bias=True, number_of_mic = self.number_of_mic, act=self.act, reduce=self.reduce
            ).to(device)

            self.wh = FusionLinearConv(
                self.input_size, self.hidden_size, bias=True, number_of_mic = self.number_of_mic, act=self.act, reduce=self.reduce
            ).to(device)
        else:
            self.wz = nn.Linear(
                self.input_size, self.hidden_size, bias=True
            ).to(device)

            self.wh = nn.Linear(
                self.input_size, self.hidden_size, bias=True
            ).to(device)

            self.wz.bias.data.fill_(0)
            torch.nn.init.xavier_normal_(self.wz.weight.data)
            self.wh.bias.data.fill_(0)
            torch.nn.init.xavier_normal_(self.wh.weight.data)

        self.u = nn.Linear(
            self.hidden_size, 2 * self.hidden_size, bias=False
        ).to(device)

        # Adding orthogonal initialization for recurrent connection
        nn.init.orthogonal_(self.u.weight)

        self.bn_wh = nn.BatchNorm1d(self.hidden_size, momentum=0.05).to(
            device
        )

        self.bn_wz = nn.BatchNorm1d(self.hidden_size, momentum=0.05).to(
            device
        )


        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False).to(device)
        self.drop_mask_te = torch.tensor([1.0], device=device).float()
        self.N_drop_masks = 100
        self.drop_mask_cnt = 0

        # Setting the activation function
        self.act = torch.nn.ReLU().to(device)

    @torch.jit.script_method
    def forward(self, x):
        # type: (Tensor) -> Tensor

        if self.bidirectional:
            x_flip = x.flip(0)
            x = torch.cat([x, x_flip], dim=1)

        # Feed-forward affine transformations (all steps in parallel)
        wz = self.wz(x)
        wh = self.wh(x)

        # Apply batch normalization
        wz_bn = self.bn_wz(wz.view(wz.shape[0] * wz.shape[1], wz.shape[2]))
        wh_bn = self.bn_wh(wh.view(wh.shape[0] * wh.shape[1], wh.shape[2]))

        wz = wz_bn.view(wz.shape[0], wz.shape[1], wz.shape[2])
        wh = wh_bn.view(wh.shape[0], wh.shape[1], wh.shape[2])

        # Processing time steps
        h = self.ligru_cell(wz, wh)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=1)
            h_b = h_b.flip(0)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    @torch.jit.script_method
    def ligru_cell(self, wz, wh):
        # type: (Tensor, Tensor) -> Tensor

        if self.bidirectional:
            h_init = torch.zeros(
                2 * self.batch_size,
                self.hidden_size,
                device="cuda",
            )
            drop_masks_i = self.drop(
                torch.ones(
                    self.N_drop_masks,
                    2 * self.batch_size,
                    self.hidden_size,
                    device="cuda",
                )
            ).data

        else:
            h_init = torch.zeros(
                self.batch_size,
                self.hidden_size,
                device="cuda",
            )
            drop_masks_i = self.drop(
                torch.ones(
                    self.N_drop_masks,
                    self.batch_size,
                    self.hidden_size,
                    device="cuda",
                )
            ).data

        hiddens = []
        ht = h_init

        if self.training:

            drop_mask = drop_masks_i[self.drop_mask_cnt]
            self.drop_mask_cnt = self.drop_mask_cnt + 1

            if self.drop_mask_cnt >= self.N_drop_masks:
                self.drop_mask_cnt = 0
                if self.bidirectional:
                    drop_masks_i = (
                        self.drop(
                            torch.ones(
                                self.N_drop_masks,
                                2 * self.batch_size,
                                self.hidden_size,
                            )
                        )
                        .to(self.device)
                        .data
                    )
                else:
                    drop_masks_i = (
                        self.drop(
                            torch.ones(
                                self.N_drop_masks,
                                self.batch_size,
                                self.hidden_size,
                            )
                        )
                        .to(self.device)
                        .data
                    )

        else:
            drop_mask = self.drop_mask_te

        for k in range(wh.shape[0]):

            uz, uh = self.u(ht).chunk(2, 1)

            at = wh[k] + uh
            zt = wz[k] + uz

            # ligru equation
            zt = torch.sigmoid(zt)
            hcand = self.act(at) * drop_mask
            ht = zt * ht + (1 - zt) * hcand
            hiddens.append(ht)

        # Stacking hidden states
        h = torch.stack(hiddens)
        return h

class liGRU(nn.Module):
    def __init__(self, options, inp_dim):
        super(liGRU, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.ligru_lay = list(map(int, options["ligru_lay"].split(",")))
        self.ligru_drop = list(map(float, options["ligru_drop"].split(",")))
        self.ligru_use_batchnorm = list(map(strtobool, options["ligru_use_batchnorm"].split(",")))
        self.ligru_use_laynorm = list(map(strtobool, options["ligru_use_laynorm"].split(",")))
        self.ligru_use_laynorm_inp = strtobool(options["ligru_use_laynorm_inp"])
        self.ligru_use_batchnorm_inp = strtobool(options["ligru_use_batchnorm_inp"])
        self.ligru_orthinit = strtobool(options["ligru_orthinit"])
        self.ligru_act = options["ligru_act"].split(",")
        self.bidir = strtobool(options["ligru_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm
        self.bn_wz = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.ligru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.ligru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_ligru_lay = len(self.ligru_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_ligru_lay):

            # Activations
            self.act.append(act_fun(self.ligru_act[i]))

            add_bias = True

            if self.ligru_use_laynorm[i] or self.ligru_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))

            if self.ligru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.ligru_lay[i]))

            if self.bidir:
                current_input = 2 * self.ligru_lay[i]
            else:
                current_input = self.ligru_lay[i]

        self.out_dim = self.ligru_lay[i] + self.bidir * self.ligru_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.ligru_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.ligru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_ligru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.ligru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.ligru_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(
                    torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.ligru_drop[i])
                )
            else:
                drop_mask = torch.FloatTensor([1 - self.ligru_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.ligru_use_batchnorm[i]:

                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # ligru equation
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                at = wh_out[k] + self.uh[i](ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand

                if self.ligru_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return x


class minimalGRU(nn.Module):
    def __init__(self, options, inp_dim):
        super(minimalGRU, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.minimalgru_lay = list(map(int, options["minimalgru_lay"].split(",")))
        self.minimalgru_drop = list(map(float, options["minimalgru_drop"].split(",")))
        self.minimalgru_use_batchnorm = list(map(strtobool, options["minimalgru_use_batchnorm"].split(",")))
        self.minimalgru_use_laynorm = list(map(strtobool, options["minimalgru_use_laynorm"].split(",")))
        self.minimalgru_use_laynorm_inp = strtobool(options["minimalgru_use_laynorm_inp"])
        self.minimalgru_use_batchnorm_inp = strtobool(options["minimalgru_use_batchnorm_inp"])
        self.minimalgru_orthinit = strtobool(options["minimalgru_orthinit"])
        self.minimalgru_act = options["minimalgru_act"].split(",")
        self.bidir = strtobool(options["minimalgru_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm
        self.bn_wz = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.minimalgru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.minimalgru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_minimalgru_lay = len(self.minimalgru_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_minimalgru_lay):

            # Activations
            self.act.append(act_fun(self.minimalgru_act[i]))

            add_bias = True

            if self.minimalgru_use_laynorm[i] or self.minimalgru_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.minimalgru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.minimalgru_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.minimalgru_lay[i], self.minimalgru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.minimalgru_lay[i], self.minimalgru_lay[i], bias=False))

            if self.minimalgru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.minimalgru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.minimalgru_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.minimalgru_lay[i]))

            if self.bidir:
                current_input = 2 * self.minimalgru_lay[i]
            else:
                current_input = self.minimalgru_lay[i]

        self.out_dim = self.minimalgru_lay[i] + self.bidir * self.minimalgru_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.minimalgru_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.minimalgru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_minimalgru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.minimalgru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.minimalgru_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(
                    torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.minimalgru_drop[i])
                )
            else:
                drop_mask = torch.FloatTensor([1 - self.minimalgru_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.minimalgru_use_batchnorm[i]:

                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # minimalgru equation
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                at = wh_out[k] + self.uh[i](zt * ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand

                if self.minimalgru_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return x


class RNN(nn.Module):
    def __init__(self, options, inp_dim):
        super(RNN, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.rnn_lay = list(map(int, options["rnn_lay"].split(",")))
        self.rnn_drop = list(map(float, options["rnn_drop"].split(",")))
        self.rnn_use_batchnorm = list(map(strtobool, options["rnn_use_batchnorm"].split(",")))
        self.rnn_use_laynorm = list(map(strtobool, options["rnn_use_laynorm"].split(",")))
        self.rnn_use_laynorm_inp = strtobool(options["rnn_use_laynorm_inp"])
        self.rnn_use_batchnorm_inp = strtobool(options["rnn_use_batchnorm_inp"])
        self.rnn_orthinit = strtobool(options["rnn_orthinit"])
        self.rnn_act = options["rnn_act"].split(",")
        self.bidir = strtobool(options["rnn_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.rnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.rnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_rnn_lay = len(self.rnn_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_rnn_lay):

            # Activations
            self.act.append(act_fun(self.rnn_act[i]))

            add_bias = True

            if self.rnn_use_laynorm[i] or self.rnn_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.rnn_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.rnn_lay[i], self.rnn_lay[i], bias=False))

            if self.rnn_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.rnn_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.rnn_lay[i]))

            if self.bidir:
                current_input = 2 * self.rnn_lay[i]
            else:
                current_input = self.rnn_lay[i]

        self.out_dim = self.rnn_lay[i] + self.bidir * self.rnn_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.rnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.rnn_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_rnn_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.rnn_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.rnn_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.rnn_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.rnn_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.rnn_use_batchnorm[i]:

                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # rnn equation
                at = wh_out[k] + self.uh[i](ht)
                ht = self.act[i](at) * drop_mask

                if self.rnn_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return x


class CNN(nn.Module):
    def __init__(self, options, inp_dim):
        super(CNN, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.cnn_N_filt = list(map(int, options["cnn_N_filt"].split(",")))

        self.cnn_len_filt = list(map(int, options["cnn_len_filt"].split(",")))
        self.cnn_max_pool_len = list(map(int, options["cnn_max_pool_len"].split(",")))

        self.cnn_act = options["cnn_act"].split(",")
        self.cnn_drop = list(map(float, options["cnn_drop"].split(",")))

        self.cnn_use_laynorm = list(map(strtobool, options["cnn_use_laynorm"].split(",")))
        self.cnn_use_batchnorm = list(map(strtobool, options["cnn_use_batchnorm"].split(",")))
        self.cnn_use_laynorm_inp = strtobool(options["cnn_use_laynorm_inp"])
        self.cnn_use_batchnorm_inp = strtobool(options["cnn_use_batchnorm_inp"])

        self.N_cnn_lay = len(self.cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):

            N_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm([N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])])
            )

            self.bn.append(
                nn.BatchNorm1d(
                    N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]), momentum=0.05
                )
            )

            if i == 0:
                self.conv.append(nn.Conv1d(1, N_filt, len_filt))

            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]))

            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])

        self.out_dim = current_input * N_filt

    def forward(self, x):

        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):

            if self.cnn_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

        x = x.view(batch, -1)

        return x


class SincNet(nn.Module):
    def __init__(self, options, inp_dim):
        super(SincNet, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.sinc_N_filt = list(map(int, options["sinc_N_filt"].split(",")))

        self.sinc_len_filt = list(map(int, options["sinc_len_filt"].split(",")))
        self.sinc_max_pool_len = list(map(int, options["sinc_max_pool_len"].split(",")))

        self.sinc_act = options["sinc_act"].split(",")
        self.sinc_drop = list(map(float, options["sinc_drop"].split(",")))

        self.sinc_use_laynorm = list(map(strtobool, options["sinc_use_laynorm"].split(",")))
        self.sinc_use_batchnorm = list(map(strtobool, options["sinc_use_batchnorm"].split(",")))
        self.sinc_use_laynorm_inp = strtobool(options["sinc_use_laynorm_inp"])
        self.sinc_use_batchnorm_inp = strtobool(options["sinc_use_batchnorm_inp"])

        self.N_sinc_lay = len(self.sinc_N_filt)

        self.sinc_sample_rate = int(options["sinc_sample_rate"])
        self.sinc_min_low_hz = int(options["sinc_min_low_hz"])
        self.sinc_min_band_hz = int(options["sinc_min_band_hz"])

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.sinc_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.sinc_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_sinc_lay):

            N_filt = int(self.sinc_N_filt[i])
            len_filt = int(self.sinc_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.sinc_drop[i]))

            # activation
            self.act.append(act_fun(self.sinc_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm([N_filt, int((current_input - self.sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i])])
            )

            self.bn.append(
                nn.BatchNorm1d(
                    N_filt, int((current_input - self.sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i]), momentum=0.05
                )
            )

            if i == 0:
                self.conv.append(
                    SincConv(
                        1,
                        N_filt,
                        len_filt,
                        sample_rate=self.sinc_sample_rate,
                        min_low_hz=self.sinc_min_low_hz,
                        min_band_hz=self.sinc_min_band_hz,
                    )
                )

            else:
                self.conv.append(nn.Conv1d(self.sinc_N_filt[i - 1], self.sinc_N_filt[i], self.sinc_len_filt[i]))

            current_input = int((current_input - self.sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i])

        self.out_dim = current_input * N_filt

    def forward(self, x):

        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.sinc_use_laynorm_inp):
            x = self.ln0(x)

        if bool(self.sinc_use_batchnorm_inp):
            x = self.bn0(x)

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_sinc_lay):

            if self.sinc_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i]))))

            if self.sinc_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i]))))

            if self.sinc_use_batchnorm[i] == False and self.sinc_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i])))

        x = x.view(batch, -1)

        return x


class SincConv(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
    ):

        super(SincConv, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel) / self.sample_rate

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2
        self.n_ = torch.arange(-n, n + 1).view(1, -1) / self.sample_rate

    def sinc(self, x):
        # Numerically stable definition
        x_left = x[:, 0 : int((x.shape[1] - 1) / 2)]
        y_left = torch.sin(x_left) / x_left
        y_right = torch.flip(y_left, dims=[1])

        sinc = torch.cat([y_left, torch.ones([x.shape[0], 1]).to(x.device), y_right], dim=1)

        return sinc

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz / self.sample_rate + torch.abs(self.low_hz_)
        high = low + self.min_band_hz / self.sample_rate + torch.abs(self.band_hz_)

        f_times_t = torch.matmul(low, self.n_)

        low_pass1 = 2 * low * self.sinc(2 * math.pi * f_times_t * self.sample_rate)

        f_times_t = torch.matmul(high, self.n_)
        low_pass2 = 2 * high * self.sinc(2 * math.pi * f_times_t * self.sample_rate)

        band_pass = low_pass2 - low_pass1
        max_, _ = torch.max(band_pass, dim=1, keepdim=True)
        band_pass = band_pass / max_

        self.filters = (band_pass * self.window_).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
    ):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :, getattr(torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda])().long(), :
    ]
    return x.view(xsize)


class SRU(nn.Module):
    def __init__(self, options, inp_dim):
        super(SRU, self).__init__()
        self.input_dim = inp_dim
        self.hidden_size = int(options["sru_hidden_size"])
        self.num_layers = int(options["sru_num_layers"])
        self.dropout = float(options["sru_dropout"])
        self.rnn_dropout = float(options["sru_rnn_dropout"])
        self.use_tanh = bool(strtobool(options["sru_use_tanh"]))
        self.use_relu = bool(strtobool(options["sru_use_relu"]))
        self.use_selu = bool(strtobool(options["sru_use_selu"]))
        self.weight_norm = bool(strtobool(options["sru_weight_norm"]))
        self.layer_norm = bool(strtobool(options["sru_layer_norm"]))
        self.bidirectional = bool(strtobool(options["sru_bidirectional"]))
        self.is_input_normalized = bool(strtobool(options["sru_is_input_normalized"]))
        self.has_skip_term = bool(strtobool(options["sru_has_skip_term"]))
        self.rescale = bool(strtobool(options["sru_rescale"]))
        self.highway_bias = float(options["sru_highway_bias"])
        self.n_proj = int(options["sru_n_proj"])
        self.sru = sru.SRU(
            self.input_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            rnn_dropout=self.rnn_dropout,
            bidirectional=self.bidirectional,
            n_proj=self.n_proj,
            use_tanh=self.use_tanh,
            use_selu=self.use_selu,
            use_relu=self.use_relu,
            weight_norm=self.weight_norm,
            layer_norm=self.layer_norm,
            has_skip_term=self.has_skip_term,
            is_input_normalized=self.is_input_normalized,
            highway_bias=self.highway_bias,
            rescale=self.rescale,
        )
        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size * 2)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)
        if x.is_cuda:
            h0 = h0.cuda()
        output, hn = self.sru(x, c0=h0)
        return output


class PASE(nn.Module):
    def __init__(self, options, inp_dim):
        super(PASE, self).__init__()

        # To use PASE within PyTorch-Kaldi, please clone the current PASE repository: https://github.com/santi-pdp/pase
        # Note that you have to clone the dev branch.
        # Take a look into the requirements (requirements.txt) and install in your environment what is missing. An important requirement is QRNN (https://github.com/salesforce/pytorch-qrnn).
        # Before starting working with PASE, it could make sense to a quick test  with QRNN independently (see “usage” section in the QRNN repository).
        # Remember to install pase. This way it can be used outside the pase folder directory.  To do it, go into the pase folder and type:
        # "python setup.py install"

        from pase.models.frontend import wf_builder

        self.input_dim = inp_dim
        self.pase_cfg = options["pase_cfg"]
        self.pase_model = options["pase_model"]

        self.pase = wf_builder(self.pase_cfg)

        self.pase.load_pretrained(self.pase_model, load_last=True, verbose=True)

        # Reading the out_dim from the config file:
        with open(self.pase_cfg) as json_file:
            config = json.load(json_file)

        self.out_dim = int(config["emb_dim"])

    def forward(self, x):

        x = x.unsqueeze(0).unsqueeze(0)
        output = self.pase(x)

        return output

class FusionLinearConv(nn.Module):
    r"""Applies a FusionLayer as described in:
        'FusionRNN: Shared Neural Parameters for
        Multi-Channel Distant Speech Recognition', Titouan P. et Al.

        Input channels are supposed to be concatenated along the last dimension
    """

    def __init__(self, in_features, out_features, number_of_mic=1, bias=True,seed=None,act="leaky",reduce="sum"):

        super(FusionLinearConv, self).__init__()
        self.in_features       = in_features // number_of_mic
        self.out_features      = out_features
        self.number_of_mic     = number_of_mic
        self.reduce            = reduce

        if act == "leaky_relu":
            self.act_function = nn.LeakyReLU()
        elif act == "prelu":
            self.act_function = nn.PReLU()
        elif act == "relu":
            self.act_function = nn.ReLU()
        else:
            self.act_function = nn.Tanh()

        self.conv = nn.Conv1d(1, self.out_features, kernel_size=self.in_features, stride=self.in_features, bias=True, padding=0)

        self.conv.bias.data.fill_(0)
        torch.nn.init.xavier_normal_(self.conv.weight.data)


    def forward(self, input):

        orig_shape = input.shape

        out = self.act_function(self.conv(input.view(orig_shape[0]*orig_shape[1], 1, -1)))

        if self.reduce == "mean":
            out = torch.mean(out, dim=-1)
        else:
            out = torch.sum(out, dim=-1)

        return out.view(orig_shape[0],orig_shape[1], -1)
