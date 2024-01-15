from __future__ import annotations

import torch
from torch import nn
import torchinfo

from .dataset import constants as const

##const.NUM_CHANNELS = 1 ########################################################

class TEMPONet(nn.Module):

    '''
    Used in
    F. Conti et al., AI-Powered Collision Avoidance Safety System for
    Industrial Woodworking Machinery
    In: Artificial Intelligence for Digitising Industry - Applications
    2021
    DOI: https://doi.org/10.1201/9781003337232
    URL:
    https://www.taylorfrancis.com/chapters/oa-edit/10.1201/9781003337232-17

    Inspired by
    Zanghieri et al., Robust Real-Time Embedded EMG Recognition Framework Using
    Temporal Convolutional Networks on a Multicore IoT Processor
    In: IEEE Transactions on Biomedical Circuits and Systems
    2020
    DOI and URL: https://doi.org/10.1109/TBCAS.2019.2959160

    Also refer to
    Zanghieri et al., sEMG-based Regression of Hand Kinematics with Temporal
    Convolutional Networks on a Low-Power Edge Microcontroller
    In: 2021 IEEE International Conference on Omni-Layer Intelligent Systems
    (COINS)
    2021
    DOI and URL: https://doi.org/10.1109/COINS51742.2021.9524188
    '''

    def __init__(self):
        super(TEMPONet, self).__init__()

        # CONVOLUTIONAL BLOCK 0

        self.b0_tcn0 = nn.Conv1d(
            const.NUM_CHANNELS, 4, 3, padding=1, bias=False)
        self.b0_tcn0_bn = nn.BatchNorm1d(4)
        self.b0_tcn0_relu = nn.ReLU()
        self.b0_tcn1 = nn.Conv1d(
            4, 4, 3, padding=1, bias=False)
        self.b0_tcn1_bn = nn.BatchNorm1d(4)
        self.b0_tcn1_relu = nn.ReLU()
        self.b0_conv = nn.Conv1d(  # stride = 1
            4, 4, 5, stride=1, padding=2, bias=False)
        self.b0_conv_bn = nn.BatchNorm1d(4)
        self.b0_conv_relu = nn.ReLU()
        self.b0_conv_pool = nn.MaxPool1d(2)

        # CONVOLUTIONAL BLOCK 1

        self.b1_tcn0 = nn.Conv1d(
            4, 4, 3, padding=1, bias=False)
        self.b1_tcn0_bn = nn.BatchNorm1d(4)
        self.b1_tcn0_relu = nn.ReLU()
        self.b1_tcn1 = nn.Conv1d(
            4, 4, 3, padding=1, bias=False)
        self.b1_tcn1_bn = nn.BatchNorm1d(4)
        self.b1_tcn1_relu = nn.ReLU()
        self.b1_conv = nn.Conv1d(  # stride = 2
            4, 4, 5, stride=2, padding=2, bias=False)
        self.b1_conv_bn = nn.BatchNorm1d(4)
        self.b1_conv_relu = nn.ReLU()
        self.b1_conv_pool = nn.MaxPool1d(2)

        # CONVOLUTIONAL BLOCK 2

        self.b2_tcn0 = nn.Conv1d(
            4, 2, 3, padding=1, bias=False)
        self.b2_tcn0_bn = nn.BatchNorm1d(2)
        self.b2_tcn0_relu = nn.ReLU()
        self.b2_tcn1 = nn.Conv1d(
            2, 2, 3, padding=1, bias=False)
        self.b2_tcn1_bn = nn.BatchNorm1d(2)
        self.b2_tcn1_relu = nn.ReLU()
        self.b2_conv = nn.Conv1d(  # stride = 4
            2, 2, 5, stride=4, padding=2, bias=False)
        self.b2_conv_bn = nn.BatchNorm1d(2)
        self.b2_conv_relu = nn.ReLU()
        self.b2_conv_pool = nn.MaxPool1d(2)

        # DENSE BLOCK

        self.fc0 = nn.Linear(64, 16, bias=False)
        self.fc0_bn = nn.BatchNorm1d(16)
        self.fc0_relu = nn.ReLU()

        self.fc1 = nn.Linear(16, 16, bias=False)
        self.fc1_bn = nn.BatchNorm1d(16)
        self.fc1_relu = nn.ReLU()

        self.fc2 = nn.Linear(16, 1, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # linear --> BN --> activation --> pool

        x = self.b0_tcn0_relu(self.b0_tcn0_bn(self.b0_tcn0(x)))
        x = self.b0_tcn1_relu(self.b0_tcn1_bn(self.b0_tcn1(x)))
        x = self.b0_conv_relu(self.b0_conv_bn(self.b0_conv(x)))
        x = self.b0_conv_pool(x)

        x = self.b1_tcn0_relu(self.b1_tcn0_bn(self.b1_tcn0(x)))
        x = self.b1_tcn1_relu(self.b1_tcn1_bn(self.b1_tcn1(x)))
        x = self.b1_conv_relu(self.b1_conv_bn(self.b1_conv(x)))
        x = self.b1_conv_pool(x)

        x = self.b2_tcn0_relu(self.b2_tcn0_bn(self.b2_tcn0(x)))
        x = self.b2_tcn1_relu(self.b2_tcn1_bn(self.b2_tcn1(x)))
        x = self.b2_conv_relu(self.b2_conv_bn(self.b2_conv(x)))
        x = self.b2_conv_pool(x)

        x = x.flatten(1)

        x = self.fc0_relu(self.fc0_bn(self.fc0(x)))
        x = self.fc1_relu(self.fc1_bn(self.fc1(x)))
        y = self.fc2(x)

        return y


class TinierNet(nn.Module):

    def __init__(self):
        super(TinierNet, self).__init__()

        self.b0_tcn = nn.Conv1d(
            const.NUM_CHANNELS, 4, kernel_size=3, padding=1,
            stride=2, bias=False)
        self.b0_bn = nn.BatchNorm1d(4)
        self.b0_relu = nn.ReLU()

        self.b1_tcn = nn.Conv1d(
            4, 4, 3, padding=1, stride=2, bias=False)
        self.b1_bn = nn.BatchNorm1d(4)
        self.b1_relu = nn.ReLU()

        self.b2_tcn = nn.Conv1d(
            4, 2, 3, padding=1, stride=2, bias=False)
        self.b2_bn = nn.BatchNorm1d(2)
        self.b2_relu = nn.ReLU()

        self.b3_tcn = nn.Conv1d(
            2, 2, 3, padding=1, stride=2, bias=False)
        self.b3_bn = nn.BatchNorm1d(2)
        self.b3_relu = nn.ReLU()

        self.b4_tcn = nn.Conv1d(
            2, 1, 3, padding=1, stride=2, bias=False)
        self.b4_bn = nn.BatchNorm1d(1)
        self.b4_relu = nn.ReLU()

        self.b5_tcn = nn.Conv1d(
            1, 1, 3, padding=1, stride=2, bias=False)
        self.b5_bn = nn.BatchNorm1d(1)
        self.b5_relu = nn.ReLU()

        self.fc0 = nn.Linear(32, 8, bias=False)
        self.fc0_bn = nn.BatchNorm1d(8)
        self.fc0_relu = nn.ReLU()

        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc1_bn = nn.BatchNorm1d(8)
        self.fc1_relu = nn.ReLU()

        self.fc2 = nn.Linear(8, 1, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:

        x = self.b0_relu(self.b0_bn(self.b0_tcn(x)))
        x = self.b1_relu(self.b1_bn(self.b1_tcn(x)))
        x = self.b2_relu(self.b2_bn(self.b2_tcn(x)))
        x = self.b3_relu(self.b3_bn(self.b3_tcn(x)))
        x = self.b4_relu(self.b4_bn(self.b4_tcn(x)))
        x = self.b5_relu(self.b5_bn(self.b5_tcn(x)))

        x = x.flatten(1)

        x = self.fc0_relu(self.fc0_bn(self.fc0(x)))
        x = self.fc1_relu(self.fc1_bn(self.fc1(x)))
        y = self.fc2(x)

        return y


def summarize(
    model: nn.Module,
    verbose: 0 | 1 | 2 = 0,
) -> torchinfo.ModelStatistics:

    # set all parameters for torch.summary

    input_size = (const.NUM_CHANNELS, const.NUM_SAMPLES)
    batch_dim = 0  # index of the batch dimension
    col_names = [
        'input_size',
        'output_size',
        'num_params',
        'params_percent',
        'kernel_size',
        'mult_adds',
        'trainable',
    ]
    device = 'cpu'
    mode = 'eval'
    row_settings = [
        'ascii_only',
        'depth',
        'var_names',
    ]

    # call the summary function

    model_stats = torchinfo.summary(
        model=model,
        input_size=input_size,
        batch_dim=batch_dim,
        col_names=col_names,
        device=device,
        mode=mode,
        row_settings=row_settings,
        verbose=verbose,
    )

    return model_stats


def main() -> None:

    # Display the summary of TEMPONet and TinierNet

    verbose = 1

    temponet = TEMPONet()
    temponet.eval()
    temponet_model_stats = summarize(temponet, verbose=verbose)

    tiniernet = TinierNet()
    tiniernet.eval()
    tiniernet_model_stats = summarize(tiniernet, verbose=verbose)


if __name__ == '__main__':
    main()
