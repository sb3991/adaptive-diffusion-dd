import os
import random
import argparse
import collections
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from synthesize.utils import *
from validation.utils import ImageFolder

################################################### For making the input to the Diffusion Model ###################################################
import os
import random
import shutil

def copy_random_images(source_path, output_path, mipc=300):
    os.makedirs(output_path, exist_ok=True)

    classes = os.listdir(source_path)

    # Copy images for each class
    for class_name in classes:
        class_input_path = os.path.join(source_path, class_name)
        class_output_path = os.path.join(output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        # Use the smaller value between the number of images in the class and mipc
        images = os.listdir(class_input_path)
        num_images_to_copy = min(mipc, len(images))  # Use the smaller value between the number of class images and mipc
        selected_images = random.sample(images, num_images_to_copy)  # If mipc is larger than the number of images, select that many

        for img in selected_images:
            src = os.path.join(class_input_path, img)
            dst = os.path.join(class_output_path, img)
            shutil.copy2(src, dst)

    print(f"All images have been successfully copied to {output_path}.")

def main(args):
    # print(args)
    # with torch.no_grad():
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
    else:
        shutil.rmtree(args.syn_data_path)
        os.makedirs(args.syn_data_path)
    copy_random_images(args.train_dir, args.syn_data_path, args.mipc)

if __name__ == "__main__":
    pass