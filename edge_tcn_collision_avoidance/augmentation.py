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
import time

import numpy as np

from .dataset import constants as const


AUGM_FACTOR = 64  # data augmentation factor

AUGM_SEED = 1

RESCALE = True
SHIFT = True

# max rescaling amplitude wrt 1:
# e.g., 0.2 means factors in [0.9, 1.1)
RESCALE_AMPLITUDE_RANGE = 0.1

# max shift range, in seconds
# e.g., 0.001 means shifts in [-0.5, +0.5] ms
SHIFT_DT_RANGE_S = 0.0005


def set_numpy_random_seed_for_augm(seed: int = AUGM_SEED) -> None:
    np.random.seed(seed)
    return


def gen_random_rescale_factors(
    num_new: np.uint32,
    rescale_amplitude_range: float,
) -> np.ndarray[np.float32]:

    r_ampl = np.random.random(num_new) - 0.5
    r_sign = np.random.random(num_new) > 0.5

    rescale_factor_ampls = 1 + rescale_amplitude_range * r_ampl  # amplitudes
    rescale_factor_signs = -1 + 2 * r_sign  # signs

    rescale_factors = rescale_factor_signs * rescale_factor_ampls  # factors
    rescale_factors = rescale_factors.astype(np.float32)

    return rescale_factors


def gen_random_shift_times(
    num_new: np.uint32,
    shift_dt_range_s: float
) -> np.ndarray[np.float32]:

    r = np.random.rand(num_new) - 0.5
    dt_s = r * shift_dt_range_s  # random shifts, in seconds

    return dt_s


def rescale_ultrasound_array_broadcast(
    x: np.ndarray[np.uint8],
    rescale_factors: np.ndarray[np.float32],
) -> np.ndarray[np.uint8]:

    """
    This function uses NumPy broadcast for "vectorization" in NumPy sense.
      x has format (num_windows, num_channels, num_samples)
      rescale_factors have format (num_windows,)
    """

    rescale_factors = rescale_factors.reshape((-1, 1, 1))  # broadcast

    x = x.astype(np.float32)
    x = x - 128.0
    x = x * rescale_factors  # vectorized
    x = x.round()
    x = np.clip(x, -128.0, +127.0)
    x = x + 128.0
    x = x.astype(np.uint8)

    return x


def shift_ultrasound_array_windowwise(
    x: np.ndarray[np.uint8],
    dt_s: np.ndarray[np.float32],
    fs_hz: float = const.FS_HZ,
) -> np.ndarray:

    """
    This function works on 1 example at a time.
      x has format (num_windows, num_channels, num_samples)
      dt_s have format (num_windows,)
    """

    num_windows, num_channels, _ = x.shape
    dt_samples = (fs_hz * dt_s).round().astype(np.int32)

    # 1 example at a time
    for idx_window in range(num_windows):

        padding_value = np.uint8(128)  # the center of {0, ..., 255}
        num_pad_samples = abs(dt_samples[idx_window])
        padding_size = (num_channels, num_pad_samples)
        padding_array = padding_value * np.ones(padding_size, dtype=np.uint8)

        if dt_samples[idx_window] > 0:
            idx_stop = - dt_samples[idx_window]
            x_kept = x[idx_window, :, :idx_stop]
            x[idx_window] = np.concatenate((padding_array, x_kept), axis=1)

        elif dt_samples[idx_window] < 0:
            idx_start = - dt_samples[idx_window]
            x_kept = x[idx_window, :, idx_start:]
            x[idx_window] = np.concatenate((x_kept, padding_array), axis=1)

        else:  # dt got rounded to 0 samples = 0.0 seconds
            pass  # do nothing: x[idx_window] stays the same

    return x


def augment_us_windows(
    x_orig: np.ndarray[np.uint8],
    y_orig: np.ndarray[np.uint8],
    augm_factor: int = AUGM_FACTOR,
    rescale: bool = RESCALE,
    rescale_amplitude_range: float = RESCALE_AMPLITUDE_RANGE,
    shift: bool = SHIFT,
    shift_dt_range_s: float = SHIFT_DT_RANGE_S,
    verbose: bool = True,
) -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]:

    """
    x has format (num_windows, num_channels, num_samples)
    y has format (num_windows,)
    """

    assert augm_factor > 0

    num_win_orig = len(y_orig)  # original windows
    num_win_augm = augm_factor * num_win_orig  # augmented windows

    # Allocation by replicas

    t_start_rep_s = time.time()

    x_augm = np.repeat(x_orig, augm_factor, axis=0)
    y_augm = np.repeat(y_orig, augm_factor, axis=0)

    t_end_rep_s = time.time()
    delta_t_rep_s = t_end_rep_s - t_start_rep_s

    # Rescalings

    t_start_rescale_s = time.time()

    if rescale:
        rescale_factors = gen_random_rescale_factors(
            num_win_augm, rescale_amplitude_range)
        x_augm = rescale_ultrasound_array_broadcast(x_augm, rescale_factors)

    t_end_rescale_s = time.time()
    delta_t_rescale_s = t_end_rescale_s - t_start_rescale_s

    # Shifts

    t_start_shift_s = time.time()

    if shift:
        dt_s = gen_random_shift_times(num_win_augm, shift_dt_range_s)
        x_augm = shift_ultrasound_array_windowwise(x_augm, dt_s)

    t_end_shift_s = time.time()
    delta_t_shift_s = t_end_shift_s - t_start_shift_s
    delta_t_total_s = delta_t_rep_s + delta_t_rescale_s + delta_t_shift_s

    # Optionally print a report
    if verbose:
        print(
            f"\n\n"
            f"--------------------------------\n"
            f"AUGMENTATION REPORT\n"
            f"\n"
            f"SETTINGS\n"
            f"Original examples:\t{num_win_orig}\n"
            f"Augmentation factor:\t{augm_factor}\n"
            f"Rescale?\t\t{str(rescale)}\n"
            f"Shift?\t\t\t{str(shift)}\n"
            f"\n"
            f"OUTCOME\n"
            f"New examples:\t\t{num_win_augm}\n"
            f"Total output examples:\t{num_win_augm}\n"
            f"Time for replicas (s):\t{delta_t_rep_s:.1f}\n"
            f"Time for rescale (s):\t{delta_t_rescale_s:.1f}\n"
            f"Time for shift (s):\t{delta_t_shift_s:.1f}\n"
            f"Time total (s):\t\t{delta_t_total_s:.1f}\n"
            f"--------------------------------\n"
            f"\n"
        )

    return x_augm, y_augm


def main() -> None:
    pass


if __name__ == '__main__':
    main()
