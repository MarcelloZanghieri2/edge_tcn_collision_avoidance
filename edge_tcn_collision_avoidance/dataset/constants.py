
"""
This module defines the anatomy of the dataset.
"""

NUM_CLASSES = 2  # classes: 'obstacle' or 'nothing'
NUM_CHANNELS = 9  # channels: data of 2022 are nine-channel

FS_HZ = 1e5  # sampling rate 100kHz
TIMEWINDOW_S = 0.02048  # time window 20.48ms
NUM_SAMPLES = round(FS_HZ * TIMEWINDOW_S)  # samples per time window
NUM_COLLECTIONS = 8

DIR_DATA_DOWNLOADED = '/scratch/zanghieri/ai4di_scm/data/downloaded/'
DIR_COLLECTION_LIST = [
    'collection_0/',
    'collection_1/',
    'collection_2/',
    'collection_3/',
    'collection_4/',
    'collection_5/',
    'collection_6/',
    'collection_7/',
]
assert len(DIR_COLLECTION_LIST) == NUM_COLLECTIONS


struct_collection_0 = {
    'run_label': {
        # all positive except runs '4' and '9'
        # run '9' is the only one with 15 windows instead of 30
        # run '9' and run '10' look swapped in the documentation pdf scheme
        1: {'num_windows': 30, 'y': True},
        2: {'num_windows': 30, 'y': True},
        3: {'num_windows': 30, 'y': True},
        4: {'num_windows': 30, 'y': False},
        5: {'num_windows': 30, 'y': True},
        6: {'num_windows': 30, 'y': True},
        7: {'num_windows': 30, 'y': True},
        8: {'num_windows': 30, 'y': True},
        9: {'num_windows': 15, 'y': False},
        10: {'num_windows': 30, 'y': True},
    }
}


struct_collection_1 = {
    'run_label': {
        # all positive except runs '1', '7', and '13'
        1: {'num_windows': 30, 'y': False},
        2: {'num_windows': 30, 'y': True},
        3: {'num_windows': 30, 'y': True},
        4: {'num_windows': 30, 'y': True},
        5: {'num_windows': 30, 'y': True},
        6: {'num_windows': 30, 'y': True},
        7: {'num_windows': 30, 'y': False},
        8: {'num_windows': 30, 'y': True},
        9: {'num_windows': 30, 'y': True},
        10: {'num_windows': 30, 'y': True},
        11: {'num_windows': 30, 'y': True},
        12: {'num_windows': 30, 'y': True},
        13: {'num_windows': 30, 'y': False},
        14: {'num_windows': 30, 'y': True},
        15: {'num_windows': 30, 'y': True},
        16: {'num_windows': 30, 'y': True},
    }
}


struct_collection_2 = {
    'run_label': {
        # negative runs are '1' to '5' (included) and '26' to '30' (included)
        # all positive in between
        1: {'num_windows': 30, 'y': False},
        2: {'num_windows': 30, 'y': False},
        3: {'num_windows': 30, 'y': False},
        4: {'num_windows': 30, 'y': False},
        5: {'num_windows': 30, 'y': False},
        6: {'num_windows': 30, 'y': True},
        7: {'num_windows': 30, 'y': True},
        8: {'num_windows': 30, 'y': True},
        9: {'num_windows': 30, 'y': True},
        10: {'num_windows': 30, 'y': True},
        11: {'num_windows': 30, 'y': True},
        12: {'num_windows': 30, 'y': True},
        13: {'num_windows': 30, 'y': True},
        14: {'num_windows': 30, 'y': True},
        15: {'num_windows': 30, 'y': True},
        16: {'num_windows': 30, 'y': True},
        17: {'num_windows': 30, 'y': True},
        18: {'num_windows': 30, 'y': True},
        19: {'num_windows': 30, 'y': True},
        20: {'num_windows': 30, 'y': True},
        21: {'num_windows': 30, 'y': True},
        22: {'num_windows': 30, 'y': True},
        23: {'num_windows': 30, 'y': True},
        24: {'num_windows': 30, 'y': True},
        25: {'num_windows': 30, 'y': True},
        26: {'num_windows': 30, 'y': False},
        27: {'num_windows': 30, 'y': False},
        28: {'num_windows': 30, 'y': False},
        29: {'num_windows': 30, 'y': False},
        30: {'num_windows': 30, 'y': False},
    }
}


struct_collection_3 = {
    'run_label': {
        # all negative
        1: {'num_windows': 30, 'y': False},
        2: {'num_windows': 30, 'y': False},
        3: {'num_windows': 30, 'y': False},
        4: {'num_windows': 30, 'y': False},
        5: {'num_windows': 30, 'y': False},
        6: {'num_windows': 30, 'y': False},
        7: {'num_windows': 30, 'y': False},
        8: {'num_windows': 30, 'y': False},
        9: {'num_windows': 30, 'y': False},
        10: {'num_windows': 30, 'y': False},
        11: {'num_windows': 30, 'y': False},
        12: {'num_windows': 30, 'y': False},
        13: {'num_windows': 30, 'y': False},
        14: {'num_windows': 30, 'y': False},
        15: {'num_windows': 30, 'y': False},
        16: {'num_windows': 30, 'y': False},
        17: {'num_windows': 30, 'y': False},
        18: {'num_windows': 30, 'y': False},
        19: {'num_windows': 30, 'y': False},
        20: {'num_windows': 30, 'y': False},
    }
}


struct_collection_4 = {
    'run_label': {
        # all negative
        # run '25', sample 8 (zero-based numbering) has 13 channels due to a
        # negligible lack of a check in the firmware; corrected starting from
        # the next collection
        21: {'num_windows': 30, 'y': False},
        22: {'num_windows': 30, 'y': False},
        23: {'num_windows': 30, 'y': False},
        24: {'num_windows': 30, 'y': False},
        25: {'num_windows': 30, 'y': False},
        26: {'num_windows': 30, 'y': False},
        27: {'num_windows': 30, 'y': False},
        28: {'num_windows': 30, 'y': False},
        29: {'num_windows': 30, 'y': False},
        30: {'num_windows': 30, 'y': False},
        31: {'num_windows': 30, 'y': False},
        32: {'num_windows': 30, 'y': False},
        33: {'num_windows': 30, 'y': False},
        34: {'num_windows': 30, 'y': False},
        35: {'num_windows': 30, 'y': False},
        36: {'num_windows': 30, 'y': False},
        37: {'num_windows': 30, 'y': False},
        38: {'num_windows': 30, 'y': False},
        39: {'num_windows': 30, 'y': False},
        40: {'num_windows': 30, 'y': False},
        41: {'num_windows': 30, 'y': False},
        42: {'num_windows': 30, 'y': False},
        43: {'num_windows': 30, 'y': False},
        44: {'num_windows': 30, 'y': False},
    }
}


struct_collection_5 = {
    'run_label': {
        # all positive
        1: {'num_windows': 30, 'y': True},
        2: {'num_windows': 30, 'y': True},
        3: {'num_windows': 30, 'y': True},
        4: {'num_windows': 30, 'y': True},
        5: {'num_windows': 30, 'y': True},
        6: {'num_windows': 30, 'y': True},
        7: {'num_windows': 30, 'y': True},
        8: {'num_windows': 30, 'y': True},
        9: {'num_windows': 30, 'y': True},
        10: {'num_windows': 30, 'y': True},
        11: {'num_windows': 30, 'y': True},
        12: {'num_windows': 30, 'y': True},
        13: {'num_windows': 30, 'y': True},
        14: {'num_windows': 30, 'y': True},
        15: {'num_windows': 30, 'y': True},
        16: {'num_windows': 30, 'y': True},
        17: {'num_windows': 30, 'y': True},
        18: {'num_windows': 30, 'y': True},
        19: {'num_windows': 30, 'y': True},
        20: {'num_windows': 30, 'y': True},
    }
}


struct_collection_6 = {
    'run_label': {
        # all negative
        # run '5', samples 27-29 are all-zero
        1: {'num_windows': 30, 'y': False},
        2: {'num_windows': 30, 'y': False},
        3: {'num_windows': 30, 'y': False},
        4: {'num_windows': 30, 'y': False},
        5: {'num_windows': 30, 'y': False},
        6: {'num_windows': 30, 'y': False},
        7: {'num_windows': 30, 'y': False},
        8: {'num_windows': 30, 'y': False},
        9: {'num_windows': 30, 'y': False},
        10: {'num_windows': 30, 'y': False},
        11: {'num_windows': 30, 'y': False},
        12: {'num_windows': 30, 'y': False},
        13: {'num_windows': 30, 'y': False},
        14: {'num_windows': 30, 'y': False},
        15: {'num_windows': 30, 'y': False},
        16: {'num_windows': 30, 'y': False},
        17: {'num_windows': 30, 'y': False},
        18: {'num_windows': 30, 'y': False},
        19: {'num_windows': 30, 'y': False},
        20: {'num_windows': 30, 'y': False},
        21: {'num_windows': 30, 'y': False},
        22: {'num_windows': 30, 'y': False},
        23: {'num_windows': 30, 'y': False},
        24: {'num_windows': 30, 'y': False},
        25: {'num_windows': 30, 'y': False},
    }
}


struct_collection_7 = {
    'run_label': {
        # all negative
        26: {'num_windows': 30, 'y': False},
        27: {'num_windows': 30, 'y': False},
        28: {'num_windows': 30, 'y': False},
        29: {'num_windows': 30, 'y': False},
        30: {'num_windows': 30, 'y': False},
        31: {'num_windows': 30, 'y': False},
        32: {'num_windows': 30, 'y': False},
        33: {'num_windows': 30, 'y': False},
        34: {'num_windows': 30, 'y': False},
        35: {'num_windows': 30, 'y': False},
        36: {'num_windows': 30, 'y': False},
        37: {'num_windows': 30, 'y': False},
        38: {'num_windows': 30, 'y': False},
        39: {'num_windows': 30, 'y': False},
        40: {'num_windows': 30, 'y': False},
        41: {'num_windows': 30, 'y': False},
        42: {'num_windows': 30, 'y': False},
        43: {'num_windows': 30, 'y': False},
        44: {'num_windows': 30, 'y': False},
        45: {'num_windows': 30, 'y': False},
        46: {'num_windows': 30, 'y': False},
        47: {'num_windows': 30, 'y': False},
        48: {'num_windows': 30, 'y': False},
        49: {'num_windows': 30, 'y': False},
        50: {'num_windows': 30, 'y': False},

    }
}


dataset_structure = {
    'collection': {
        0: struct_collection_0,
        1: struct_collection_1,
        2: struct_collection_2,
        3: struct_collection_3,
        4: struct_collection_4,
        5: struct_collection_5,
        6: struct_collection_6,
        7: struct_collection_7,
    }
}


def main() -> None:
    pass


if __name__ == '__main__':
    main()
