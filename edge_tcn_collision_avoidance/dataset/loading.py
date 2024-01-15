from __future__ import annotations
import time

import numpy as np

from . import constants as const


def ids2filepath(
    idx_collection: int,
    run_label: int,
    idx_win: int,
) -> str:

    filename = f"test{run_label}_sample{idx_win}.csv"

    filepath = const.DIR_DATA_DOWNLOADED \
        + const.DIR_COLLECTION_LIST[idx_collection] + filename

    return filepath


def load_window(
    idx_collection: int,
    run_label: int,
    idx_win: int,
) -> np.ndarray[np.uint8]:

    filepath = ids2filepath(idx_collection, run_label, idx_win)
    x = np.loadtxt(filepath, delimiter=',')
    # x has already format (num_channels, num_samples) = (9, 2048)
    x = x.astype(np.uint8)

    # run '25', sample 8 (zero-based numbering) has 13 channels due to a
    # negligible lack of a check in the firmware; corrected starting from
    # the next collection
    if idx_collection == 4 and run_label == 25 and idx_win == 8:
        x = x[:const.NUM_CHANNELS]  # discard the supernumerary channels

    return x


def load_collection(
    idx_collection: int,
    verbose: bool = False,
) -> tuple[np.ndarray[np.uint8], np.ndarray[bool]]:

    struct_collection = const.dataset_structure['collection'][idx_collection]

    num_runs = len(struct_collection['run_label'])

    t_start_s = time.time()

    # loop over runs
    y_runs_list = [None for _ in range(num_runs)]
    x_runs_list = [None for _ in range(num_runs)]
    run_labels_list = list(struct_collection['run_label'].keys())
    for idx_run in range(num_runs):

        run_label = run_labels_list[idx_run]

        num_windows = struct_collection['run_label'][run_label]['num_windows']

        y_run_scalar = struct_collection['run_label'][run_label]['y']
        y_runs_list[idx_run] = np.full(num_windows, y_run_scalar, dtype=bool)

        x_run = np.zeros(
            (num_windows, const.NUM_CHANNELS, const.NUM_SAMPLES),
            dtype=np.uint8,
        )

        # loop over windows
        for idx_win in range(num_windows):
            x_run[idx_win] = load_window(idx_collection, run_label, idx_win)

        x_runs_list[idx_run] = x_run

        # optionally display information about the loaded run
        if verbose:
            if idx_run == 0:
                print(f"\n")
            print(
                f"Loaded {num_windows} windows "
                f"of Run {idx_run + 1 : 2d}/{num_runs}\t"
                f"(run label: '{run_label}'; class: {str(y_run_scalar)})\t"
                f"of Collection {idx_collection}"
            )
            if idx_run == num_runs - 1:
                print(f"\n")

    t_finish_s = time.time()
    delta_t_s = t_finish_s - t_start_s

    # optionally display info about the loaded collection
    if verbose:
        print(
            f"Finished loading all {num_runs} Runs "
            f"of Collection {idx_collection} "
            f"(collections are 0 to {const.NUM_COLLECTIONS - 1}). "
            f"Time taken: {delta_t_s:.2f}"
            f"\n"
        )

    y_collection_array = np.concatenate(y_runs_list, axis=0)
    x_collection_array = np.concatenate(x_runs_list, axis=0)

    return x_collection_array, y_collection_array


def main() -> None:
    pass


if __name__ == '__main__':
    main()
