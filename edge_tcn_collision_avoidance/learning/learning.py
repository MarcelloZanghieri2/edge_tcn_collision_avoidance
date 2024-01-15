from __future__ import annotations
import enum
import os
import random
import time

# non-torch imports
import numpy as np
import sklearn.utils as sklutils
import sklearn.metrics as sklmetrics
# torch imports
import torch
from torch import nn
import torch.utils.data

from ..dataset import constants as const
from .settings import DEVICE
from . import goodness as good


NUM_EPOCHS = 16  # epochs in floating point
MINIBATCH_SIZE_TRAIN = 64  # minibatch size for training
MINIBATCH_SIZE_INFER = 8192  # minibatch size for inference


class UltrasoundDataset():

    def __init__(
        self,
        x: np.ndarray[np.uint8],
        y: np.ndarray[np.uint8] | None = None,
    ):

        assert x.shape[1:] == (const.NUM_CHANNELS, const.NUM_SAMPLES)
        num_windows = x.shape[0]
        if y is not None:
            assert y.shape == (num_windows,)
            ymin, ymax = y.min(), y.max()
            assert ymin == 0 and ymax == const.NUM_CLASSES - 1

        self.x = x
        self.y = y
        self.num_windows = num_windows

    def __len__(self) -> int:
        return self.num_windows

    def __data_generation(
        self, idx: int,
    ) -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]:
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __getitem__(
        self, idx: int,
    ) -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]:
        return self.__data_generation(idx)


def collate_x_only(
    minibatch: list[np.ndarray[np.uint8]]
) -> torch.Tensor[torch.float32]:

    # concatenating in NumPy first should be faster
    x = np.array(minibatch, dtype=np.float32)
    del minibatch
    x = torch.tensor(x, dtype=torch.float32, device='cpu') / 255.0

    return x


def collate_xy_pairs(
    minibatch: list[tuple[np.ndarray[np.uint8], np.ndarray[bool]]]
) -> tuple[torch.Tensor[torch.float32], torch.Tensor[torch.float32]]:

    # concatenating in NumPy first should be faster
    x = np.array([xy[0] for xy in minibatch], dtype=np.float32)
    y = np.array([xy[1] for xy in minibatch], dtype=np.float32)
    del minibatch

    x = torch.tensor(x, dtype=torch.float32, device='cpu') / 255.0
    y = torch.tensor(y, dtype=torch.float32, device='cpu')

    return x, y


@enum.unique
class Mode(enum.Enum):
    TRAINING = 'TRAINING'
    INFERENCE = 'INFERENCE'


def dataset2dataloader(
    dataset: UltrasoundDataset,
    mode: Mode,
) -> torch.utils.data.DataLoader:

    assert isinstance(mode, Mode)

    if mode == Mode.TRAINING:
        batch_size = MINIBATCH_SIZE_TRAIN
        drop_last = True
        shuffle = True
        sampler = None
    elif mode == Mode.INFERENCE:
        batch_size = MINIBATCH_SIZE_INFER
        drop_last = False
        shuffle = False
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        raise ValueError

    collate_fn = collate_x_only if dataset.y is None else collate_xy_pairs

    dataloader = torch.utils.data.DataLoader(
        dataset,  # just arg, not a kwarg
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    return dataloader


def do_inference(
    xinfer: np.ndarray[np.uint32],
    model: torch.nn.Module,
    output_scale: float = 1.0,
) -> tuple:

    model.eval()
    model.to(DEVICE)

    dataset_infer = UltrasoundDataset(x=xinfer)
    dataloader_infer = dataset2dataloader(dataset_infer, mode=Mode.INFERENCE)

    yout = np.zeros((0,), dtype=np.float32)

    for x_b in dataloader_infer:
        x_b = x_b.to(DEVICE)
        with torch.no_grad():
            yout_b = model(x_b)
        yout_b = yout_b.detach()
        yout_b = yout_b.cpu()
        yout_b = yout_b.numpy()
        yout_b = yout_b.squeeze(1)
        yout = np.concatenate((yout, yout_b), axis=0)

    yout *= output_scale

    return yout


def do_training(
    xtrain: np.ndarray[np.uint32],
    ytrain: np.ndarray[bool],
    xvalid: np.ndarray[np.uint32],
    yvalid: np.ndarray[bool],
    model: torch.nn.Module,
    criterion: torch.nn.Module | None = None,  # None for default (ugly)
    optimizer: torch.optim.Optimizer | None = None,  # None for default (ugly)
    num_epochs: int = NUM_EPOCHS,
) -> tuple:

    model.to(DEVICE)
    model.train()

    dataset_train = UltrasoundDataset(x=xtrain, y=ytrain)
    dataloader_train = dataset2dataloader(dataset_train, mode=Mode.TRAINING)

    if criterion is None:
        pos_weight = 1.0 / ytrain.mean() - 1.0
        pos_weight_tensor = torch.tensor(
            [pos_weight], dtype=torch.float32, device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    if optimizer is None:
        params = model.parameters()
        lr = 0.0001
        weight_decay = 0.0
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    history = {
        'epoch': {},
    }

    print('\nEPOCH\t\tL_tr\tSens_tr\tSpec_tr\tAb_tr\tAUROC_tr\t\tL_va\tSens_va\
          \tSpec_va\tAb_va\tAUROC_va\t\tTime (s)\n')

    for idx_epoch in range(num_epochs):

        t0_s = time.time()

        for x_b, y_b in dataloader_train:
            x_b = x_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            optimizer.zero_grad()
            yout_b = model(x_b)
            yout_b = yout_b.squeeze(1)
            loss_b = criterion(yout_b, y_b)
            loss_b.backward()
            optimizer.step()

        yout_train = do_inference(xtrain, model)
        ysoft_train = 1.0 / (1.0 + np.exp(- yout_train))  # sigmoid
        metrics_train_epoch = good.compute_detection_metrics(
            ytrain, ysoft_train, pos_weight)

        yout_valid = do_inference(xvalid, model)
        ysoft_valid = 1.0 / (1.0 + np.exp(- yout_valid))  # sigmoid
        metrics_valid_epoch = good.compute_detection_metrics(
            yvalid, ysoft_valid, pos_weight)

        t1_s = time.time()
        deltat_s = t1_s - t0_s

        print("%d/%d\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t\t\t%.3f\t%.3f\t%.3f\t\
              %.3f\t%.3f\t\t\t%.1f" % (

            idx_epoch + 1,
            num_epochs,

            metrics_train_epoch['balanced_bce'],
            metrics_train_epoch['sensitivity_maxba'],
            metrics_train_epoch['specificity_maxba'],
            metrics_train_epoch['balanced_accuracy_maxba'],
            metrics_train_epoch['auroc'],

            metrics_valid_epoch['balanced_bce'],
            metrics_valid_epoch['sensitivity_maxba'],
            metrics_valid_epoch['specificity_maxba'],
            metrics_valid_epoch['balanced_accuracy_maxba'],
            metrics_valid_epoch['auroc'],

            deltat_s,
        ))

    history['epoch'][idx_epoch] = {
        'training': metrics_train_epoch,
        'validation': metrics_valid_epoch,
    }

    return model, history, ysoft_train, ysoft_valid


def main() -> None:
    pass


if __name__ == '__main__':
    main()
