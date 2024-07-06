"""
The script converts the trial pickle data to llava format.
Usage:
cd /mnt/disk_1/guanxing/LLaVA
conda activate py310
python scripts/convert_homerobot_to_llava.py

ref:
https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md

"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
import pickle

import cv2
import numpy as np
from tqdm import tqdm



def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # iterate over the data folder
    data_list = os.listdir(args.data_dir)
    save_samples = []
    cnt = 0

    # action_type = set()   # ContinuousFullBodyAction or DiscreteNavigationAction

    print(f"example: {data_list[:3]}")
    for trial in tqdm(data_list):
        # load pickle data
        demo_path = os.path.join(args.data_dir, trial, "obs_data.pkl")
        with open(demo_path, "rb") as f:
            demo = pickle.load(f)
        for sample in demo:
            obs = sample['obs_data']
            # info = sample['info_data']
            action = sample['action_float_data']
            step = sample['step']

            # action_type.add(sample['action_raw_data'])

            # save the rgb image now as jpg
            rgb = obs['rgb']
            image_path = demo_path.replace("obs_data.pkl", f"rgb_{step}.jpg")
            cv2.imwrite(image_path, rgb)

            save_sample = {}
            save_sample['id'] = cnt
            save_sample['image'] = image_path

            save_sample['conversations'] = [
                {'from': 'human', 'value': '<image>\nSpecify the action of manipulating the object.',},
                {'from': 'gpt', 'value': f'The action is {action}',}
            ]
            save_samples.append(save_sample)
            cnt += 1

        # print(action_type)
        pass

    output_path = os.path.join(output_dir, f"manip_data.json")

    # write to json
    with open(output_path, "w") as f:
        json.dump(save_samples, f, indent=4)

    print(f"Data saved to {output_path}")
    print(f"Total frames: {len(save_samples)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/disk_1/guanxing/home-robot/data/datasets/rl_agent",
        help="(Absolute) Path to data root dir (home-robot data dir)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="playground/data",
        help="Path to output dir (LLaVA data dir)",
    )
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument(
        "overrides",
        default=['VISUALIZE=1'],
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    main(args)
