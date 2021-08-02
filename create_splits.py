import argparse
import glob
import os
import random

import numpy as np
import shutil
import math

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function
    files = os.listdir(os.path.join(data_dir, 'processed'))
    print(len(files))
    total_files = len(files)
    #train_pct = 75
    val_pct = 15
    test_pct = 15
    #train_idx = math.floor(train_pct*total_files/100) + 1
    val_idx = math.floor(val_pct*total_files/100)
    test_idx = val_idx + math.floor(test_pct*total_files/100)
    print(total_files - test_idx, val_idx, test_idx)
    
    dest = os.path.join(data_dir, 'val')
    os.makedirs(dest, exist_ok=True)
    for f1 in files[0:val_idx]:
        shutil.move(os.path.join(data_dir, 'processed', f1), os.path.join(dest, f1))

    dest = os.path.join(data_dir, 'testing')
    os.makedirs(dest, exist_ok=True)
    for f1 in files[val_idx:test_idx]:
        shutil.move(os.path.join(data_dir, 'processed', f1), os.path.join(dest, f1))

    dest = os.path.join(data_dir, 'training')
    os.makedirs(dest, exist_ok=True)
    for f1 in files[test_idx:]:
        shutil.move(os.path.join(data_dir, 'processed', f1), os.path.join(dest, f1))
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)