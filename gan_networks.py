import torch
import torch.nn as nn
from distutils.util import strtobool
from torch.nn.utils import spectral_norm
import math


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


def create_module_list(input_dim, output_dim, cfg):
    dnn_lay = list(map(int, cfg["dnn_lay"].split(",")))
    dnn_drop = list(map(float, cfg["dnn_drop"].split(",")))
    dnn_batchnorm = list(map(strtobool, cfg["dnn_batchnorm"].split(",")))
    dnn_act = cfg["dnn_act"].split(",")

    dnn_lay.append(output_dim)

    layers = nn.ModuleList([])
    N_dnn_lay = len(dnn_lay)

    current_input = input_dim
    add_bias = True

    for i in range(N_dnn_lay):
        # Linear operations
        layers.append(nn.Linear(current_input, dnn_lay[i], bias=add_bias))

        add_bias = False

        # batch norm
        if dnn_batchnorm[i]:
            layers.append(nn.BatchNorm1d(dnn_lay[i], momentum=0.05))

        # activation
        if dnn_act[i] != "linear":
            layers.append(act_fun(dnn_act[i]))

        # dropout
        if dnn_drop[i] > 0:
            layers.append(nn.Dropout(p=dnn_drop[i]))

        current_input = dnn_lay[i]

    return layers


class Block_Encode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=int((kernel_size - 1) / 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Block_Decode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=int(math.ceil((kernel_size - 2) / 2))),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Block_Encode_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=int((kernel_size - 1) / 2)),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Block_Decode_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=int(math.ceil((kernel_size - 2) / 2))),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Block_Encode_SN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()

        self.block = nn.Sequential(
            spectral_norm(nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=int((kernel_size - 1) / 2))),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Block_Decode_SN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()

        self.block = nn.Sequential(
            spectral_norm(nn.ConvTranspose1d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=int(math.ceil((kernel_size - 2) / 2)))),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator_BCE(nn.Module):
    def __init__(self, inp_dim, cfg):
        super(Discriminator_BCE, self).__init__()

        self.input_dim = inp_dim
        self.output_dim = 1
        leaky_alpha = 0.2

        self.block = nn.Sequential(

            nn.Conv1d(in_channels=1,
                      out_channels=16,
                      kernel_size=41,
                      padding=20,
                      stride=1),
            # nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=16,
                      out_channels=16,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            # nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=16,
                      out_channels=32,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),

            nn.Conv1d(in_channels=32,
                      out_channels=32,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),

            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),

            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),

            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),

            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            # nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),

        )

        self.out_block = nn.Sequential(
            nn.Linear(128, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block(x.view(-1, 1, self.input_dim))
        x = self.out_block(x.view(-1, 128))
        return x


def create_conv_module_list(input_dim, output_dim, type, cfg):
    conv_features = list(map(int, cfg["conv_features"].replace(" ", "").split(",")))
    conv_kernals = list(map(int, cfg["conv_kernals"].replace(" ", "").split(",")))
    conv_dropout = list(map(float, cfg["conv_dropout"].replace(" ", "").split(",")))
    conv_batchnorm = list(map(str, cfg["conv_batchnorm"].replace(" ", "").split(",")))
    conv_act = list(map(str, cfg["conv_act"].replace(" ", "").split(",")))

    layers = nn.ModuleList([])
    N_features_lay = len(conv_features)

    if type == "Cycle_Generator" or type == "Cycle_Discriminator":
        for i in range(N_features_lay - 1):

            # Convolutional layers
            layers.append(nn.Conv1d(in_channels=conv_features[i],
                                    out_channels=conv_features[i + 1],
                                    kernel_size=conv_kernals[i],
                                    padding=int((conv_kernals[i] - 1) / 2),
                                    stride=1))

            # Batch norm
            if conv_batchnorm[i] == "True":
                layers.append(nn.BatchNorm1d(conv_features[i + 1]))

            # Activation
            if conv_act[i] != "linear":
                layers.append(act_fun(conv_act[i]))

            # Dropout
            if conv_dropout[i] > 0:
                layers.append(nn.Dropout(p=conv_dropout[i]))

    if type.lower().__contains__("discriminator"):
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Sigmoid())

    return layers


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size


class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = ((self.kernel_size[0] - 1) // 2)


class Discriminator_wgan_spectral_norm(nn.Module):
    def __init__(self, inp_dim, cfg):
        super(Dis_WGAN_SN, self).__init__()

        self.input_dim = inp_dim
        self.output_dim = 1
        leaky_alpha = 0.2
        dropout = 0.3

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=16,
                      kernel_size=41,
                      padding=20,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=16,
                      out_channels=16,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=16,
                      out_channels=32,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=32,
                      out_channels=32,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=13,
                      padding=6,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
        )

        self.out_block = nn.Sequential(
            spectral_norm(nn.Linear(128, self.output_dim)),
        )

    def forward(self, x):
        x = self.block(x.view(-1, 1, self.input_dim))
        x = self.out_block(x.view(-1, 128))
        return x


class Generator_small(nn.Module):
    def __init__(self, inp_dim, out_dim, cfg):
        super(Generator_small, self).__init__()

        dropout = 0
        self.inp_dim = inp_dim
        leaky_alpha = 0.2
        kernel_size = 5

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=16,
                      kernel_size=kernel_size,
                      padding=int((kernel_size - 1) / 2),
                      stride=1),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=16,
                      out_channels=16,
                      kernel_size=kernel_size,
                      padding=int((kernel_size - 1) / 2),
                      stride=1),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=16,
                      out_channels=32,
                      kernel_size=kernel_size,
                      padding=int((kernel_size - 1) / 2),
                      stride=1),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=32,
                      out_channels=16,
                      kernel_size=kernel_size,
                      padding=int((kernel_size - 1) / 2),
                      stride=1),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=16,
                      out_channels=1,
                      kernel_size=kernel_size,
                      padding=int((kernel_size - 1) / 2),
                      stride=1),
        )

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = x.view(-1, 1, self.inp_dim)
        return self.block(x).view(-1, self.inp_dim)


class Generator_large(nn.Module):
    def __init__(self, inp_dim, out_dim, cfg):
        super(Generator_large, self).__init__()

        dropout = 0

        self.encode1 = Block_Encode(in_channels=1, out_channels=16, kernel_size=7, stride=2, dropout=0)
        self.encode2 = Block_Encode(16, 16, 7, 2, dropout)
        self.encode3 = Block_Encode(16, 32, 5, 2, dropout)
        self.encode4 = Block_Encode(32, 32, 5, 2, dropout)
        self.encode5 = Block_Encode(32, 64, 3, 2, dropout)

        self.decode1 = Block_Decode(64, 32, 4, 2, dropout)
        self.decode2 = Block_Decode(64, 32, 5, 2, dropout)
        self.decode3 = Block_Decode(64, 16, 6, 2, dropout)
        self.decode4 = Block_Decode(32, 16, 6, 2, dropout)
        self.decode5 = Block_Decode(32, 1, 6, 2, dropout)

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # [b x 1 x 440]
        e1 = self.encode1(x)  # [b x 16 x 220]
        e2 = self.encode2(e1)  # [b x 16 x 110]
        e3 = self.encode3(e2)  # [b x 32 x 55]
        e4 = self.encode4(e3)  # [b x 32 x 28]
        e5 = self.encode5(e4)  # [b x 64 x 14]

        d1 = self.decode1(e5)
        d2 = self.decode2(torch.cat((d1, e4), dim=1))
        d3 = self.decode3(torch.cat((d2, e3), dim=1))
        d4 = self.decode4(torch.cat((d3, e2), dim=1))
        d5 = self.decode5(torch.cat((d4, e1), dim=1))

        return torch.squeeze(d5, dim=1)


class Discriminator_spectral_norm(nn.Module):
    def __init__(self, inp_dim, cfg):
        super(Discriminator_spectral_norm, self).__init__()

        self.input_dim = inp_dim
        self.output_dim = 1
        leaky_alpha = 0.2
        dropout = 0.25

        kernel_size = 5

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=16,
                      kernel_size=kernel_size,
                      padding=int((kernel_size-1)/2),
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=16,
                      out_channels=16,
                      kernel_size=kernel_size,
                      padding=int((kernel_size-1)/2),
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=16,
                      out_channels=32,
                      kernel_size=kernel_size,
                      padding=int((kernel_size-1)/2),
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=32,
                      out_channels=32,
                      kernel_size=kernel_size,
                      padding=int((kernel_size-1)/2),
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),

            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=kernel_size,
                      padding=int((kernel_size-1)/2),
                      stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(leaky_alpha, inplace=True),

        )

        self.out_block = nn.Sequential(
            spectral_norm(nn.Linear(832, self.output_dim)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block(x.view(-1, 1, self.input_dim))
        x = self.out_block(x.view(-1, 832))
        return x
