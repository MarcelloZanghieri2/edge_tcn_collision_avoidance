"""
    Author(s):
    Marcello Zanghieri <marcello.zanghieri2@unibo.it>
    
    Copyright (C) 2024 University of Bologna and ETH Zurich
    
    Licensed under the GNU Lesser General Public License (LGPL), Version 2.1
    (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        https://www.gnu.org/licenses/lgpl-2.1.txt
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from __future__ import annotations
import enum

import numpy as np

from .dataset import loading as ld


@enum.unique
class Experiment(enum.IntEnum):
    EXPERIMENT_0 = 0
    EXPERIMENT_1 = 1
    EXPERIMENT_2 = 2
    EXPERIMENT_3 = 3


# @enum.unique
# class SplitOfNoisyData(str, enum.Enum):
#    CONTIGUOUS = 'CONTIGUOUS'
#    STRATIFIED = 'STRATIFIED'  # homogenenous


# SPLIT_OF_NOISY_DATA = SplitOfNoisyData.STRATIFIED
# SPLIT_OF_NOISY_DATA = SplitOfNoisyData.CONTIGUOUS

# assert isinstance(SPLIT_OF_NOISY_DATA, SplitOfNoisyData)

# if SPLIT_OF_NOISY_DATA == SplitOfNoisyData.STRATIFIED:
#    pass
    # SLICE_0_SESS_5 = slice(0, -1, 3)
    # SLICE_SESS_5_EXP_2 = slice(1, -1, 3)
    # SLICE_SESS_5_EXP_3 = slice(2, -1, 3)
    # SLICE_SESS_6_EXP_1 = slice(0, -1, 3)
    # SLICE_SESS_6_EXP_2 = slice(1, -1, 3)
    # SLICE_SESS_6_EXP_3 = slice(2, -1, 3)
    # SLICE_SESS_7_EXP_1 = slice(0, -1, 3)
    # SLICE_SESS_7_EXP_2 = slice(1, -1, 3)
    # SLICE_SESS_7_EXP_3 = slice(2, -1, 3)
# elif SPLIT_OF_NOISY_DATA == SplitOfNoisyData.CONTIGUOUS:
#    pass
# else:
#    raise ValueError


def compose_train_and_valid_sets(
    experiment: Experiment,
) -> tuple[
    tuple[np.ndarray[np.uint8], np.ndarray[bool]],
    tuple[np.ndarray[np.uint8], np.ndarray[bool]],
]:  # (x_train, y_train), (x_valid, y_valid)

    assert isinstance(experiment, Experiment)

    # load all data (even if not optimal)

    x0, y0 = ld.load_collection(idx_collection=0)
    x1, y1 = ld.load_collection(idx_collection=1)
    x2, y2 = ld.load_collection(idx_collection=2)
    x3, y3 = ld.load_collection(idx_collection=3)
    x4, y4 = ld.load_collection(idx_collection=4)
    x5, y5 = ld.load_collection(idx_collection=5)
    x6, y6 = ld.load_collection(idx_collection=6)
    x7, y7 = ld.load_collection(idx_collection=7)

    # split based on the experiment

    if experiment == Experiment.EXPERIMENT_0:

        # EXPERIMENT 0
        # Noiseless scenario: train on [I, III, V], test on [II, IV].

        xtrain = np.concatenate((x0, x2, x4), axis=0)
        ytrain = np.concatenate((y0, y2, y4), axis=0)

        xvalid = np.concatenate((x1, x3), axis=0)
        yvalid = np.concatenate((y1, y3), axis=0)

    elif experiment == Experiment.EXPERIMENT_1:

        # EXPERIMENT 1
        # See no noisy data at training, validate on last third.

        xtrain = np.concatenate((x0, x1, x2, x3, x4), axis=0)
        ytrain = np.concatenate((y0, y1, y2, y3, y4), axis=0)

        # xvalid = np.concatenate((x5[400:], x7[250:]), axis=0)
        # yvalid = np.concatenate((y5[400:], y7[250:]), axis=0)
        xvalid = np.concatenate((x5[2::3], x7[2::3]), axis=0)
        yvalid = np.concatenate((y5[2::3], y7[2::3]), axis=0)

    elif experiment == Experiment.EXPERIMENT_2:

        # EXPERIMENT 2
        # See first third of noisy data at training, validate on last third.

        # xtrain = np.concatenate(
        #     (x0, x1, x2, x3, x4, x5[:200], x6[:500]), axis=0)
        # ytrain = np.concatenate(
        #     (y0, y1, y2, y3, y4, y5[:200], y6[:500]), axis=0)
        xtrain = np.concatenate(
            (x0, x1, x2, x3, x4, x5[0::3], x6[0::3]), axis=0)
        ytrain = np.concatenate(
            (y0, y1, y2, y3, y4, y5[0::3], y6[0::3]), axis=0)

        # xvalid = np.concatenate((x5[400:], x7[250:]), axis=0)
        # yvalid = np.concatenate((y5[400:], y7[250:]), axis=0)
        xvalid = np.concatenate((x5[2::3], x7[2::3]), axis=0)
        yvalid = np.concatenate((y5[2::3], y7[2::3]), axis=0)

    elif experiment == Experiment.EXPERIMENT_3:

        # EXPERIMENT 3
        # See 2 thirds of noisy data at training; validate on last third.

        # xtrain = np.concatenate(
        #     (x0, x1, x2, x3, x4, x5[:400], x6, x7[:250]), axis=0)
        # ytrain = np.concatenate(
        #     (y0, y1, y2, y3, y4, y5[:400], y6, y7[:250]), axis=0)
        xtrain = np.concatenate(
            (x0, x1, x2, x3, x4, x5[0::3], x5[1::3], x6[0::3], x6[1::3]),
            axis=0,
        )
        ytrain = np.concatenate(
            (y0, y1, y2, y3, y4, y5[0::3], y5[1::3], y6[0::3], y6[1::3]),
            axis=0,
        )

        # xvalid = np.concatenate((x5[400:], x7[250:]), axis=0)
        # yvalid = np.concatenate((y5[400:], y7[250:]), axis=0)
        xvalid = np.concatenate((x5[2::3], x7[2::3]), axis=0)
        yvalid = np.concatenate((y5[2::3], y7[2::3]), axis=0)

    else:
        raise ValueError

    return (xtrain, ytrain), (xvalid, yvalid)


def main() -> None:
    pass


if __name__ == '__main__':
    main()
