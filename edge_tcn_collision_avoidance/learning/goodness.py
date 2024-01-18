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

import numpy as np
from sklearn import metrics as m


# clipping extremes for taking the log
EPS_CLIP_SOFT = 1.0e-7


# TODO: ALSO FOR F1 SCORE

def compute_sens_spec_max_bal_accuracy(
    ytrue: np.ndarray[bool],
    ysoft: np.ndarray[np.float32],
) -> dict:

    # compute the values

    fpr_swept, tpr_swept, thresholds_swept = m.roc_curve(ytrue, ysoft)
    num_thresh = len(thresholds_swept)

    sensitivity_swept = tpr_swept
    specificity_swept = 1.0 - fpr_swept

    balanced_accuracy_swept = (sensitivity_swept + specificity_swept) / 2.0
    idx_maxba = balanced_accuracy_swept.argmax()
    threshold_maxba = thresholds_swept[idx_maxba]
    sensitivity_maxba = sensitivity_swept[idx_maxba]
    specificity_maxba = specificity_swept[idx_maxba]
    balanced_accuracy_maxba = balanced_accuracy_swept[idx_maxba]

    # store the values into a dictionary to be returned

    metrics_maxba = {
        'threshold_maxba': threshold_maxba,
        'sensitivity_maxba': sensitivity_maxba,
        'specificity_maxba': specificity_maxba,
        'balanced_accuracy_maxba': balanced_accuracy_maxba,
    }

    return metrics_maxba


def compute_detection_metrics(
    ytrue: np.ndarray[bool],
    ysoft: np.ndarray[np.float32],
    pos_weight: float = 1.0,
    eps_clip_soft: float = EPS_CLIP_SOFT
) -> dict:

    # compute metrics' values

    yhard = ysoft > 0.5

    # balanced BCE
    # print('\n\n')
    # print(ysoft.min(), 1.0 - ysoft.max())
    YSOFT_MIN = eps_clip_soft
    YSOFT_MAX = 1.0 - eps_clip_soft
    ysoft_clipped = np.clip(ysoft, YSOFT_MIN, YSOFT_MAX)
    # print(ysoft_clipped.min(), 1.0 - ysoft_clipped.max())
    # print('\n\n')
    bce_pos = - np.mean(ytrue * np.log(ysoft_clipped))
    bce_neg = - np.mean((1.0 - ytrue) * np.log(1.0 - ysoft_clipped))
    del ysoft_clipped
    balanced_bce = pos_weight * bce_pos + 1.0 * bce_neg

    # AUROC
    auroc = m.roc_auc_score(ytrue, ysoft)  # uses soft

    # metrics maximizing balanced_accuracy
    metrics_maxba = compute_sens_spec_max_bal_accuracy(ytrue, ysoft)

    del ysoft

    # sensitivity
    sensitivity = m.recall_score(ytrue, yhard) if ytrue.any() else None

    # specificity
    specificity = m.recall_score(~ ytrue, ~ yhard) if not ytrue.all() else None

    # balanced accuracy
    balanced_accuracy = m.balanced_accuracy_score(ytrue, yhard)

    # recall
    recall = sensitivity

    # precision
    precision = m.precision_score(ytrue, yhard) if yhard.any() else None

    # F1 score
    # recall and precision must be defined, and at least one of them non-zero
    if (recall is not None and precision is not None) and \
            (recall > 0.0 or precision > 0.0):
        f1_score = m.f1_score(ytrue, yhard)
    else:
        f1_score = None

    # store into a dictionary

    detection_metrics = {
        'pos_weight': pos_weight,
        'eps_clip_soft': eps_clip_soft,
        'balanced_bce': balanced_bce,
        'auroc': auroc,

        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,

        'threshold_maxba': metrics_maxba['threshold_maxba'],
        'sensitivity_maxba': metrics_maxba['sensitivity_maxba'],
        'specificity_maxba': metrics_maxba['specificity_maxba'],
        'balanced_accuracy_maxba': metrics_maxba['balanced_accuracy_maxba'],

        'recall': recall,
        'precision': precision,
        'f1_score': f1_score,
    }

    return detection_metrics


def do_stats_from_repetitions(
    detection_metrics_list: list[dict],
) -> dict:

    keys_list = [
        'balanced_bce',
        'auroc',

        'sensitivity',
        'specificity',
        'balanced_accuracy',

        'recall',
        'precision',
        'f1_score',
    ]

    metrics_stats = {}

    for key in keys_list:
        metric_array = np.array(
            [dm[key] for dm in detection_metrics_list], dtype=np.float32)
        metric_avg = metric_array.mean()
        metric_std = metric_array.std()
        metrics_stats[key]['avg'] = metric_avg
        metrics_stats[key]['std'] = metric_std

    return metrics_stats


def main() -> None:
    pass


if __name__ == '__main__':
    main()
