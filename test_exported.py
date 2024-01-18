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

import numpy as np
import torch
from torch import nn

from ultras import splitting as split
from ultras import augmentation as augm
from ultras.learning import learning as learn
from ultras import models as mdl

from ultras.learning import quantization as quant

# %%
#import random;
#random.seed(1)

#np.random.seed(1)
augm.set_numpy_random_seed_for_augm()

#torch.manual_seed(1)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.cuda.manual_seed(1)
from ultras.learning import settings
settings.set_reproducibility()
settings.set_visible_device()


# %%
# Experiment Settings

CURRENT_EXPERIMENT = split.Experiment.EXPERIMENT_3

RESULTS_FILENAME = 'results_experiment_%d.pkl' % CURRENT_EXPERIMENT.value
MODEL_FILENAME = 'model_experiment_%d.onnx' % CURRENT_EXPERIMENT.value
RESULTS_DIR_PATH = './results/'
RESULTS_FILE_FULLPATH = RESULTS_DIR_PATH + RESULTS_FILENAME
MODEL_FILE_FULLPATH = RESULTS_DIR_PATH + MODEL_FILENAME

NUM_REPETITIONS = 1
USE_SAFEGUARD = True

# %%
# Composition of training and validation set
(xtrain, ytrain), (xvalid, yvalid) = \
    split.compose_train_and_valid_sets(experiment=CURRENT_EXPERIMENT)

# %%
# Augmentation
xtrain, ytrain = \
    augm.augment_us_windows(xtrain, ytrain, augm_factor=8, verbose=True)

# %%
# Formatting to dataset object
trainset = learn.UltrasoundDataset(x=xtrain, y=ytrain)
validset = learn.UltrasoundDataset(x=xvalid, y=yvalid)

num_train = len(trainset)
num_valid = len(validset)

# %%
# RESULTS STRUCTURE

results = {
    'CURRENT_EXPERIMENT': CURRENT_EXPERIMENT,
    'ytr': ytrain,
    'yva': yvalid,
    'NUM_REPETITIONS': NUM_REPETITIONS,
    
    'history_fp' : [None for _ in range(NUM_REPETITIONS)],
    'ylogittr_fp': np.zeros((NUM_REPETITIONS, num_train), dtype=np.float32),
    'ylogitva_fp': np.zeros((NUM_REPETITIONS, num_valid), dtype=np.float32),

    'history_fq' : [None for _ in range(NUM_REPETITIONS)],
    'ylogittr_id': np.zeros((NUM_REPETITIONS, num_train), dtype=np.float32),
    'ylogitva_id': np.zeros((NUM_REPETITIONS, num_valid), dtype=np.float32),
    'eps_out'    : np.empty((NUM_REPETITIONS,), dtype=np.float32),
}

# %%
# For each repetition: training, quantization and validations

for idx_rep in range(NUM_REPETITIONS):
    
    print(f"\n\n\nREPETITION {idx_rep + 1}/{NUM_REPETITIONS}")

    # ----------------------------------------------------------------------- #

    # Training & Validation
    tiniernet = mdl.TinierNet()
    tiniernet.to(learn.DEVICE)

    # ----------------------------------------------------------------------- #

    #classes_array = np.arange(const.NUM_CLASSES)
    #class_weights = sklutils.class_weight.compute_class_weight(
    #    class_weight='balanced', classes=classes_array, y=ytrain)
    #class_weights.astype(np.float32)
    #class_weights /= class_weights.sum()
    #class_weights = torch.tensor(
    #    class_weights, dtype=torch.float32, requires_grad=False, device='cpu')
    #pos_weight = class_weights[1]

    pos_weight = 1.0 / ytrain.mean() - 1
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=learn.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight)
    #criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    
    print(f"pos_weight: {pos_weight.item():.3f}")

    # ----------------------------------------------------------------------- #

    optimizer = torch.optim.Adam(tiniernet.parameters(), lr=0.0001)
    
    tiniernet, history_fp = learn.do_training(
        trainset, validset, tiniernet, criterion, optimizer, num_epochs=16)
    

    # just to debug it
    input_scale = 0.999999
    model_fq_rounded = quant.quantlib_flow(
        trainset, validset, tiniernet, input_scale, criterion,
        export=True, onnx_filename=MODEL_FILE_FULLPATH)

    
    # Apply the safeguard against bad trainings: if bad, redo the current
    # repetiton from scratch
    # Note: experiment 1 is does not work anyway, due to noiseless training set
    # and noisy validation set.
    if USE_SAFEGUARD and CURRENT_EXPERIMENT != split.Experiment.EXPERIMENT_1: 
        amount_of_overfitting = \
           history_fp['aurocva'][learn.NUM_EPOCHS - 1] \
            - history_fp['aurocva'][learn.NUM_EPOCHS - 1]
        if amount_of_overfitting > 0.25:
            print('\n\nBAD TRAINING: THIS REPETITION WILL BE REDONE FROM SCRATCH.\n\n')
            continue # no update to loop index: redo this repetition

    # Store and save the results after each repetition

    # store floating-point results
    results['history_fp' ][idx_rep] = history_fp
    #results['ylogittr_fp'][idx_rep] = youttr_fp
    #results['ylogitva_fp'][idx_rep] = youtva_fp
    
    
    # save results
    # results_dict = {'results': results}
    # Path(RESULTS_DIR_PATH).mkdir(parents=True, exist_ok=True)
    # with open(RESULTS_FILE_FULLPATH, 'wb') as f:
    #     pickle.dump(results_dict, f)    

# %%
import torch
import onnx
from onnx2torch import convert

onnx_model = onnx.load(MODEL_FILE_FULLPATH)
torch_model = convert(onnx_model)
# short for torch_model = convert(MODEL_FILE_FULLPATH)

import onnxruntime as ort

# Create example data
x = torch.ones((1, 9, 2048))

out_torch = torch_model(x)

ort_sess = ort.InferenceSession(MODEL_FILE_FULLPATH)

input_name_str = onnx_model.graph.input[0].name
input_feed = {input_name_str, x.numpy()}

outputs_ort = ort_sess.run(
    output_names=None,
    input_feed=input_feed,
    run_options=None,
)


# Check the Onnx output against PyTorch
print(torch.max(torch.abs(outputs_ort - out_torch.detach().numpy())))
print(np.allclose(outputs_ort, out_torch.detach().numpy(), atol=1.e-7))

# %%



