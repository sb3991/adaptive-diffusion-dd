from torch.utils.data import ConcatDataset
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
from validation.utils import ImageFolder, RDED_ImageFolder

################################################### Original RDED Codes ###################################################
def remove_last_directory(path):
    parent_dir, _ = os.path.split(path)
    return parent_dir


def copy_random_images(source_path, output_path, mipc=300):
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Get all class folders from the source path
    classes = os.listdir(source_path)

    # Copy images for each class
    for class_name in classes:
        class_input_path = os.path.join(source_path, class_name)
        class_output_path = os.path.join(output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        # Use the smaller value between the number of images in the class and mipc
        images = os.listdir(class_input_path)
        num_images_to_copy = min(mipc, len(images))  # Use the smaller value between the number of class images and mipc
        selected_images = random.sample(images, num_images_to_copy)  # Select as many as mipc if it's larger than the number of images

        for img in selected_images:
            src = os.path.join(class_input_path, img)
            dst = os.path.join(class_output_path, img)
            shutil.copy2(src, dst)

    print(f"All images have been successfully copied to {output_path}.")


def merge_datasets(paths_1, paths_2, output_path, num_samples=300):
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Get all class folders from both paths
    all_classes = set()
    for path in paths_1 + paths_2:
        all_classes.update(os.listdir(path))

    # Copy images for each class
    for class_name in all_classes:
        class_output_path = os.path.join(output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        # Randomly select images from paths_1
        for path in paths_1:
            class_path = os.path.join(path, class_name)
            if os.path.exists(class_path):
                images = os.listdir(class_path)
                selected_images = random.sample(images, min(num_samples, len(images)))  # Randomly select
                for img in selected_images:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(class_output_path, f"1_{img}")
                    shutil.copy2(src, dst)

        # Randomly select images from paths_2
        for path in paths_2:
            class_path = os.path.join(path, class_name)
            if os.path.exists(class_path):
                images = os.listdir(class_path)
                selected_images = random.sample(images, min(num_samples, len(images)))  # Randomly select
                for img in selected_images:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(class_output_path, f"2_{img}")
                    shutil.copy2(src, dst)

    print(f"All images have been successfully merged into {output_path}.")


def init_images(args, model=None): 
    args.train_dir = f"/home/sb/link/DD_DIF/Data/{args.subset}/train/"
    args.val_dir = f"/home/sb/link/DD_DIF/Data/{args.subset}/val/"
    additional_path = 'Diffusion_image'
    output_additional_path = 'Sampled'
    parent_dir = remove_last_directory(args.syn_data_path)
    Dif_Dir = os.path.join(parent_dir, additional_path)
    output_path = os.path.join(parent_dir, output_additional_path)
    additional_path_2 = 'syn_data'
    RDED_Dir = os.path.join(parent_dir, additional_path_2)
    paths_3 = [RDED_Dir] # syn data
    paths_1 = [Dif_Dir] # Diffusion image
    paths_4 = [output_path]

    trainset = RDED_ImageFolder(
        classes=args.classes,
        ipc=(args.mipc), # The first two are the number of data in the folder
        paths=paths_1, # train_dir
        shuffle=True,
        transform=None,
    )
    trainset.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            MultiRandomCrop(
                num_crop=args.num_crop, size=args.input_size, factor=args.factor
            ),
            normalize,
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=(args.mipc),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    for c, (images, labels) in enumerate(tqdm(train_loader)):
        images = selector(
            args.ipc * args.factor**2,
            model,
            images,
            labels,
            args.input_size,
            m=args.num_crop,
        )
        images = mix_images(images, args.input_size, args.factor, args.ipc)
        save_images(args, denormalize(images), c)


def save_images(args, images, class_id):
    additional_path = 'RDED_after_Dif'

    parent_dir = remove_last_directory(args.syn_data_path)
    save_path = os.path.join(parent_dir, additional_path)

    for id in range(images.shape[0]):
        dir_path = "{}/{:05d}".format(save_path, class_id)
        place_to_store = dir_path + "/class{:05d}_id{:05d}.jpg".format(class_id, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def main(args):
    print(args)
    with torch.no_grad():
        model_teacher = load_model(
            model_name=args.arch_name,
            dataset=args.subset,
            pretrained=True,
            classes=args.classes,
        )
        model_teacher = nn.DataParallel(model_teacher).cuda()
        model_teacher.eval()
        model_teacher = model_teacher.cuda()
        for p in model_teacher.parameters():
            p.requires_grad = False

        init_images(args, model_teacher)


if __name__ == "__main__":
    pass