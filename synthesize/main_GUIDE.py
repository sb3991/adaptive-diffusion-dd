import math
import os
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import EulerDiscreteScheduler, UNet2DConditionModel, DDIMScheduler, DPMSolverMultistepScheduler
import tqdm
from synthesize.LocalStableDiffusion import StableDiffusionPipeline
from torch.utils.data import Dataset, Sampler
import random
from synthesize.utils import load_model, normalize

from bayes_opt import BayesianOptimization
from functools import partial

from open_clip import tokenizer
import open_clip
# Initialize diffusion model
# Define the objective function

class CustomImagePromptDataset(Dataset):
    def __init__(self, image_root, label_prompt_file, additional_prompt_dir, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.classes = sorted(os.listdir(image_root))
        self.image_paths = []
        self.prompts = []
        self.negative_prompts = []
        self.class_names = []
        
        additional_path = 'prompts'
        parent_dir_ = remove_last_directory(self.image_root)
        additional_prompt_dir = os.path.join(parent_dir_, additional_path)
        # Load main label prompts
        with open(label_prompt_file, 'r') as f:
            label_prompts = f.readlines()
        label_prompts = [prompt.strip() for prompt in label_prompts]

        # Load additional prompts
        additional_prompts = {}
        for cls in self.classes:
            additional_prompt_file = os.path.join(additional_prompt_dir, f"{cls.zfill(5)}_captions.txt")
            if os.path.exists(additional_prompt_file):
                with open(additional_prompt_file, 'r') as f:
                    additional_prompts[cls] = f.readlines()
                additional_prompts[cls] = [prompt.strip() for prompt in additional_prompts[cls] if prompt.strip()]
            else:
                print(f"Warning: Additional prompt file not found for class {cls}")
                additional_prompts[cls] = []

        for cls in self.classes:
            cls_image_dir = os.path.join(image_root, cls)
            class_idx = int(cls)
            main_prompt = label_prompts[class_idx] if class_idx < len(label_prompts) else ""
            negative_prompt = " ".join([label_prompts[i] for i in range(len(label_prompts)) if i != class_idx])
            
            if os.path.isdir(cls_image_dir):
                images = [os.path.join(cls_image_dir, img) for img in os.listdir(cls_image_dir) if img.endswith(('.png', '.jpg', '.jpeg', '.JPEG'))]
                for img in images:
                    additional_prompt = random.choice(additional_prompts[cls]) if additional_prompts[cls] else ""
                    combined_prompt = f"{main_prompt} {additional_prompt}".strip()
                    self.image_paths.append(img)
                    self.prompts.append(combined_prompt)
                    self.negative_prompts.append(negative_prompt)
                    self.class_names.append(cls)
                    # print(f"프롬프트: {combined_prompt}")

                print(f"클래스 '{cls}': {len(images)}개의 이미지 로드됨")
            else:
                print(f"클래스 '{cls}': 이미지 디렉토리가 존재하지 않음.")

        print(f"총 이미지 수: {len(self.image_paths)}")
        print(f"총 프롬프트 수: {len(self.prompts)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        prompt = self.prompts[idx]
        negative_prompt = self.negative_prompts[idx]
        class_name = self.class_names[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, prompt, negative_prompt, class_name
    

def remove_last_directory(path):
    parent_dir, _ = os.path.split(path)
    return parent_dir

def get_max_image_size(image_root):
    max_width, max_height = 0, 0
    for root, _, files in os.walk(image_root):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    width, height = img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
    return max_width, max_height

def objective_function(pipe,prompts, images, class_names, classifier_scale, negative_prompts, early_stage ,late_stage,dino=None,teacher_model=None, args=None ,strength=0.0, guidance_scale=0.0,**kwargs):
    transform = transforms.Compose([transforms.Resize(args.input_size)])
    images = transform(images)
    soft_mix_label_0 = teacher_model(images)
    soft_mix_label = F.softmax(soft_mix_label_0 / args.temperature, dim=1)
    baseline = soft_mix_label[:,int(class_names[0])].mean()
    gen_imgs = pipe(prompt=prompts,
                    image=images,
                    strength=strength,
                    class_labels=class_names,
                    guidance_scale=guidance_scale,
                    gradient_scale=classifier_scale,
                    num_inference_steps=10,
                    negative_prompt=negative_prompts,
                    early_stage_end=early_stage,
                    late_stage_start=late_stage,
                    coeff=args.lam,
                    output_type=None,
                    dino=dino,
                    ).images
    gen_imgs = gen_imgs.transpose(0,3,1,2)
    gen_imgs = torch.from_numpy(gen_imgs)
    soft_mix_label_0 = teacher_model(gen_imgs.cuda())
    soft_mix_label = F.softmax(soft_mix_label_0 / args.temperature, dim=1)
    pred_conf = soft_mix_label[:,int(class_names[0])].mean()
    ret = pred_conf - baseline
    return ret.cpu().item()
    

def generate_images(data_loader, pipe, output_dir, image_size, images_per_class, classifier_scale, prompt_strength, inference_steps, early_stage, late_stage ,dino=None,teacher_model=None,args=None, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    class_indices = {class_name: 0 for class_name in set([class_name for _, _, _, class_name in data_loader.dataset])}
    print("prompt_strength", prompt_strength, "classifier_scale", classifier_scale)
    while any(class_indices[class_name] < images_per_class for class_name in class_indices):
        for batch_idx, (images, prompts, negative_prompts, class_names) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Generating images"):
            images = images.to("cuda")
            prompts = list(prompts)
            negative_prompts = ['cartoon, anime, painting'] * len(prompts)
            #negative_prompts = [''] * len(prompts)
            with torch.no_grad():
        
                obj_func = partial(objective_function,pipe=pipe,prompts=prompts, images=images, 
                                   class_names=class_names, classifier_scale=classifier_scale, 
                                   negative_prompts=negative_prompts, early_stage=early_stage,
                                   late_stage=late_stage ,dino=dino, teacher_model=teacher_model, args=args)
                pbounds = {
                    'strength': (args.strength_lb, args.strength_ub),      # Example range for guidance scale
                    'guidance_scale': (args.cfg_lb, args.cfg_ub)   # Example range for strength
                }
                # Initialize Bayesian Optimizer
                optimizer = BayesianOptimization(
                    f=obj_func,  # Objective function
                    pbounds=pbounds,       # Parameter bounds
                    random_state=42        # For reproducibility
                )

                # Run optimization
                optimizer.maximize(
                    init_points=args.init_points,         # Number of random initial samples
                    n_iter=args.n_iter,             # Number of optimization iterations
                )

                # Print the best parameters
                print("Best parameters:", optimizer.max)
                strength, guidance_scale = optimizer.max['params']['strength'], optimizer.max['params']['guidance_scale']          
                
                #strength, guidance_scale = 0.7, 8.0
                
                generated_images = pipe(
                    prompt=prompts,
                    image=images,
                    strength=strength, 
                    class_labels=class_names,
                    guidance_scale=guidance_scale,
                    gradient_scale=classifier_scale,
                    num_inference_steps=50,
                    negative_prompt=negative_prompts,
                    early_stage_end=early_stage,
                    late_stage_start=late_stage,
                    coeff=args.lam
                ).images
                resized_images =generated_images #[img.resize(image_size, Image.LANCZOS) for img in generated_images] #
            for i, (img, class_name, prompt, negative_prompt) in enumerate(zip(resized_images, class_names, prompts, negative_prompts)):
                if class_indices[class_name] < images_per_class:
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)

                    img_index = class_indices[class_name]
                    img.save(os.path.join(class_dir, f"Class_{class_name}_{img_index}.png"))

                    class_indices[class_name] += 1

                if all(class_indices[class_name] >= images_per_class for class_name in class_indices):
                    break


class SameClassBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.class_to_indices = defaultdict(list)
        for idx, class_name in enumerate(dataset.class_names):
            self.class_to_indices[class_name].append(idx)
            
    def __iter__(self):
        indices = []
        for class_indices in self.class_to_indices.values():
            np.random.shuffle(class_indices)
            for i in range(0, len(class_indices), self.batch_size):
                batch = class_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    indices.append(batch)
        
        np.random.shuffle(indices)
        flat_indices = []
        for batch in indices:
            flat_indices.extend(batch)
            
        return iter(flat_indices)
    
    def __len__(self):
        if self.drop_last:
            return sum(len(indices) // self.batch_size * self.batch_size 
                      for indices in self.class_to_indices.values())
        return sum(len(indices) for indices in self.class_to_indices.values())

def custom_collate_fn(batch):
    images, prompts, negative_prompts, class_names = zip(*batch)
    images = torch.stack(images) if torch.is_tensor(images[0]) else images
    return images, prompts, negative_prompts, class_names

def main(args):
    image_root = args.syn_data_path
    if 'cifar100' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/Data/label-prompt/cifar100_labels.txt'
        print("Load cifar100_labels.txt")
    elif 'cifar10' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/Data/label-prompt/CIFAR-10_labels.txt'
        print("Load CIFAR-10_labels.txt")
    elif 'tinyimagenet' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/Data/label-prompt/tinyimagenet.txt'
        print("Load tiny_imagenet_labels.txt")
    elif 'imagenet-100' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/Data/label-prompt/imagenet-100.txt'
        print("Load imagenet100_labels.txt")
    elif 'imagenet-woof' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/Data/label-prompt/Imagewoof.txt'
        print("Load imagenet100_labels.txt")
    elif 'imagenet-1k' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/Data/label-prompt/imagenet-1k.txt'
        print("Load imagenet100_labels.txt")
    elif 'imagenet-nette' in image_root:
        prompt_root = '/home/sb/link/DD_DIF/Data/label-prompt/imagenet-nette.txt'
        print("Load imagenet-nette_labels.txt") 
    elif 'imagenet-sketch' in image_root or 'imagenet-r' in image_root:
        prompt_root =f'/home/sb/link/DD_DIF/Data/label-prompt/{args.subset}.txt'
        print("Load imagenet-sketch_labels.txt")
    else:
        prompt_root = '/default/path/to/labels.txt'
    additional_prompt_dir = os.path.join(image_root, 'prompts')
    additional_path = 'Diffusion_image'
    parent_dir = remove_last_directory(args.syn_data_path)
    output_dir = os.path.join(parent_dir, additional_path)
    batch_size = 4
    images_per_class = args.mipc #args.mipc
    inference_step=args.num_inference_steps
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = CustomImagePromptDataset(image_root, prompt_root, additional_prompt_dir, transform=transform)
    sampler = SameClassBatchSampler(dataset, batch_size=batch_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, sampler=sampler)
    
    print(f"데이터셋 크기: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("데이터셋이 비어 있습니다. 이미지와 프롬프트 파일이 올바르게 로드되는지 확인하세요.")

    model_id = "runwayml/stable-diffusion-v1-5"
    
    max_width, max_height = get_max_image_size(image_root)
    image_size = (max(max_width, max_height), max(max_width, max_height))
    print("Image size:", image_size)
    print(f'args subset {args.subset}')

    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").half()
    dino = None
    
    teacher_model = load_model(
        model_name=args.arch_name,
        dataset=args.subset,
        pretrained=True,
        classes=args.classes,
        )
    for p in teacher_model.parameters():
        p.requires_grad = False

    teacher_model.eval()

    teacher_model = nn.DataParallel(teacher_model).cuda()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        unet=unet,
        scheduler=scheduler,
        saftey_checker=None,
        torch_dtype=torch.float16
    ).to("cuda")

    teacher_model = teacher_model.cuda()
    teacher_model.eval()      
    pipe.set_progress_bar_config(disable=True)
    prompt_strength=args.prompt_strength
    classifier_scale=args.classifier_scale
    early_stage=args.early_stage
    late_stage=args.late_stage

    generate_images(data_loader, pipe, output_dir, image_size, images_per_class, classifier_scale, prompt_strength, inference_step, early_stage, late_stage,teacher_model=teacher_model ,args=args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_data_path", type=str, required=True, help="Path to the root directory of images")
    parser.add_argument("--mipc", type=int, default=500, required=True, help="Number of images to generate per class")

    args = parser.parse_args()
    main(args)