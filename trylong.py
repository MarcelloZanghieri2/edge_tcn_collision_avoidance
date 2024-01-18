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

# %%
from __future__ import annotations
from pathlib import Path
import pickle

from ultras import splitting as split
from ultras import augmentation as augm
from ultras import models as mdl
from ultras.learning import settings
from ultras.learning import learning as learn
from ultras.learning import quantization as quant

# %%
# Set settings for reproducibility
augm.set_numpy_random_seed_for_augm()
settings.set_reproducibility()
settings.set_visible_device()

# %%
# Experiment Settings

CURRENT_EXPERIMENT = split.Experiment.EXPERIMENT_1

RESULTS_FILENAME = 'results_experiment_%d.pkl' % CURRENT_EXPERIMENT.value
MODEL_FILENAME = 'model_experiment_%d.onnx' % CURRENT_EXPERIMENT.value
RESULTS_DIR_PATH = './results/'
RESULTS_FILE_FULLPATH = RESULTS_DIR_PATH + RESULTS_FILENAME
MODEL_FILE_FULLPATH = RESULTS_DIR_PATH + MODEL_FILENAME

NUM_REPETITIONS = 64
USE_SAFEGUARD = True

# %%
# Composition of training and validation set
# "orig" stands for non-augmented
(xtrain_orig, ytrain_orig), (xvalid, yvalid) = \
    split.compose_train_and_valid_sets(experiment=CURRENT_EXPERIMENT)

# %%
# Structure of dictionary for storing results

results = {
    'EXPERIMENT': CURRENT_EXPERIMENT,
    'NUM_REPETITIONS': NUM_REPETITIONS,
    'ytrain_orig': ytrain_orig,  # not augmented
    'yvalid': yvalid,
    'repetition': {},  # all results
}

# %%
# For each repetition:
# - training and validation in floating-point
# - quantization:
#   - Post-Training Qtantization (PTQ)
#   - (maybe) Quantization-Aware Tuning (QAT) for readjusting
#   - validation

for idx_rep in range(NUM_REPETITIONS):

    print(
        f"\n"
        f"\n------------------------------------------------------------------"
        f"\nREPETITION {idx_rep + 1}/{NUM_REPETITIONS}"
        f"\n------------------------------------------------------------------"
        f"\n"
    )

    # ----------------------------------------------------------------------- #

    # Augmentation

    xtrain_augm, ytrain_augm = augm.augment_us_windows(
        xtrain_orig, ytrain_orig, augm_factor=16, rescale=True, verbose=True)

    # ----------------------------------------------------------------------- #

    # Training & Validation

    # try:
    #    tiniernet
    # except:
    #    tiniernet = mdl.TinierNet()

    tiniernet = mdl.TinierNet()

    tiniernet, history_fp, ysoft_train_fp, ysoft_valid_fp = learn.do_training(
        xtrain_augm, ytrain_augm, xvalid, yvalid, tiniernet, num_epochs=2)

    # ----------------------------------------------------------------------- #

    # Quantization

    input_scale = 255.0

    model_tq, output_scale, history_q,\
        metrics_train_q, metrics_valid_q, \
        ysoft_train_q, ysoft_valid_q = quant.quantlib_flow(
            xtrain_augm,
            ytrain_augm,
            xvalid,
            yvalid,
            model=tiniernet,
            do_qat=True,
            num_epochs_qat=16,
            input_scale=input_scale,
            export=True,
            onnx_filename=MODEL_FILE_FULLPATH,
        )

    del xtrain_augm, ytrain_augm

    # ----------------------------------------------------------------------- #

    # Store and save the results after each repetition

    # store floating-point results
    results['repetition'][idx_rep] = {
        'history_fp': history_fp,
        'ysoft_train_fp': ysoft_train_fp,
        'ysoft_valid_fp': ysoft_valid_fp,

        'history_q': history_q,
        'ysoft_train_q': ysoft_train_q,
        'ysoft_valid_q': ysoft_valid_q,
        'output_scale': output_scale,
        'metrics_train_q': metrics_train_q,
        'metrics_valid_q': metrics_valid_q,
    }

    # save results
    results_outer_dict = {'results': results}
    Path(RESULTS_DIR_PATH).mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE_FULLPATH, 'wb') as f:
        pickle.dump(results_outer_dict, f)

    # ----------------------------------------------------------------------- #


# %%
