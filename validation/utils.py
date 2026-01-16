import torch
from torch.utils.data import Dataset
import numpy as np
import os
import torch.distributed
import torchvision
from torchvision.transforms import functional as t_F
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms as transforms


# keep top k largest values, and smooth others
def keep_top_k(p, k, n_classes=1000):  # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes - k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(
        topk_smooth_p.sum().item(), p.shape[0]
    ), f"{topk_smooth_p.sum().item()} not close to {p.shape[0]}"
    return topk_smooth_p


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find("weight") >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(
        group_no_weight_decay
    )
    groups = [
        dict(params=group_weight_decay),
        dict(params=group_no_weight_decay, weight_decay=0.0),
    ]
    return groups



class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.targets = []
        
        # 클래스 이름을 디렉토리 이름으로 가정하고, 각 디렉토리 내의 파일을 리스트업
        classes = sorted(os.listdir(root))
        for class_index, class_name in enumerate(classes):
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    if os.path.isfile(image_path):
                        self.image_paths.append(image_path)
                        self.targets.append(class_index)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = self.targets[index]
        return image, target



class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, classes, ipc, mem=False, shuffle=False, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.mem = mem
        self.image_paths = []
        self.targets = []
        self.samples = []
        
        # 클래스 이름이 문자열인지 숫자인지 구분
        for class_name in classes:
            class_name=str(class_name)
            if class_name.isdigit():  # 클래스 이름이 숫자일 경우
                dir_path = os.path.join(self.root, class_name.zfill(5))  # 기존 방식 유지
            else:  # 클래스 이름이 문자열일 경우
                dir_path = os.path.join(self.root, class_name)  # 폴더 이름 그대로 사용
            
            file_ls = os.listdir(dir_path)
            if shuffle:
                random.shuffle(file_ls)
            for i in range(ipc):
                file_path = os.path.join(dir_path, file_ls[i])
                self.image_paths.append(file_path)
                # 클래스 인덱스 설정
                if class_name.isdigit():
                    self.targets.append(int(class_name))  # 숫자일 경우 그대로 사용
                else:
                    self.targets.append(classes.index(class_name))  # 문자열일 경우 인덱스로 변환
                if self.mem:
                    self.samples.append(self.loader(file_path))

    def __getitem__(self, index):
        if self.mem:
            sample = self.samples[index]
        else:
            sample = self.loader(self.image_paths[index])
        sample = self.transform(sample)
        return sample, self.targets[index]

    def __len__(self):
        return len(self.targets)

# class ImageFolder(torchvision.datasets.ImageFolder):
#     def __init__(self, classes, ipc, mem=False, shuffle=False, **kwargs):
#         super(ImageFolder, self).__init__(**kwargs)
#         self.mem = mem
#         self.image_paths = []
#         self.targets = []
#         self.samples = []
#         for c in range(len(classes)):
#             dir_path = self.root + "/" + str(classes[c]).zfill(5)
#             # print(dir_path)
#             file_ls = os.listdir(dir_path)
#             if shuffle:
#                 random.shuffle(file_ls)
#             # print(len(file_ls))
#             for i in range(ipc):
#                 self.image_paths.append(dir_path + "/" + file_ls[i])
#                 self.targets.append(c)
#                 if self.mem:
#                     self.samples.append(self.loader(dir_path + "/" + file_ls[i]))

#     def __getitem__(self, index):
#         if self.mem:
#             sample = self.samples[index]
#         else:
#             sample = self.loader(self.image_paths[index])
#         sample = self.transform(sample)
#         return sample, self.targets[index]

#     def __len__(self):
#         return len(self.targets)

  

class RDED_ImageFolder(torchvision.datasets.VisionDataset):
    def __init__(self, classes, ipc, paths, mem=False, shuffle=True, transform=None):
        super().__init__(root=None, transform=transform)  # VisionDataset을 상속받는 방식으로 수정
        self.mem = mem
        self.image_paths = []
        self.targets = []
        self.samples = []
        self.class_counts = {c: 0 for c in range(len(classes))}
        self.loader = torchvision.datasets.folder.default_loader

        def add_images_from_folder(folder_path, class_idx):
            file_ls = os.listdir(folder_path)
            file_ls = [f for f in file_ls if os.path.isfile(os.path.join(folder_path, f))]  # 파일만 골라냄
            if shuffle:
                random.shuffle(file_ls)
            count = 0
            for i in range(min(ipc, len(file_ls))):
                self.image_paths.append(os.path.join(folder_path, file_ls[i]))
                self.targets.append(class_idx)
                if self.mem:
                    self.samples.append(self.loader(os.path.join(folder_path, file_ls[i])))
                count += 1
            self.class_counts[class_idx] += count

        for path in paths:
            for c in range(len(classes)):
                dir_path = os.path.join(path, str(classes[c]).zfill(5))
                if os.path.exists(dir_path):
                    add_images_from_folder(dir_path, c)
                else:
                    print(f"Directory {dir_path} does not exist!")

        # 클래스별로 로드된 이미지 수 출력
        for class_idx, count in self.class_counts.items():
            print(f"Class {class_idx}: {count} images loaded")

    def __getitem__(self, index):
        if self.mem:
            sample = self.samples[index]
        else:
            sample = self.loader(self.image_paths[index])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.targets[index]

    def __len__(self):
        return len(self.targets)

# class RDED_ImageFolder(ImageFolder):
#     def __init__(self, classes, ipc, paths, mem=False, shuffle=False, transform=None):
#         self.mem = mem
#         self.image_paths = []
#         self.targets = []
#         self.samples = []
#         self.class_counts = {c: 0 for c in range(len(classes))}
#         self.loader = torchvision.datasets.folder.default_loader
#         self.transform = transform

#         def add_images_from_folder(folder_path, class_idx):
#             file_ls = os.listdir(folder_path)
#             if shuffle:
#                 random.shuffle(file_ls)
#             count = 0
#             for i in range(min(ipc, len(file_ls))):
#                 self.image_paths.append(os.path.join(folder_path, file_ls[i]))
#                 self.targets.append(class_idx)
#                 if self.mem:
#                     self.samples.append(self.loader(os.path.join(folder_path, file_ls[i])))
#                 count += 1
#             self.class_counts[class_idx] += count

#         for path in paths:
#             for c in range(len(classes)):
#                 dir_path = os.path.join(path, str(classes[c]).zfill(5))
#                 add_images_from_folder(dir_path, c)

#         # Print the number of images loaded for each class
#         for class_idx, count in self.class_counts.items():
#             print(f"Class {class_idx}: {count} images loaded")

#     def __getitem__(self, index):
#         if self.mem:
#             sample = self.samples[index]
#         else:
#             sample = self.loader(self.image_paths[index])
#         if self.transform is not None:
#             sample = self.transform(sample)
#         return sample, self.targets[index]

#     def __len__(self):
#         return len(self.targets)



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, args, rand_index=None, lam=None, bbox=None):
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(args.cutmix, args.cutmix)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None):
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(args.mixup, args.mixup)

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images, args, rand_index=None, lam=None, bbox=None):
    if args.mix_type == "mixup":
        return mixup(images, args, rand_index, lam)
    elif args.mix_type == "cutmix":
        return cutmix(images, args, rand_index, lam, bbox)
    else:
        return images, None, None, None


class ShufflePatches(torch.nn.Module):
    def shuffle_weight(self, img, factor):
        h, w = img.shape[1:]
        th, tw = h // factor, w // factor
        patches = []
        for i in range(factor):
            i = i * tw
            if i != factor - 1:
                patches.append(img[..., i : i + tw])
            else:
                patches.append(img[..., i:])
        random.shuffle(patches)
        img = torch.cat(patches, -1)
        return img

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        return img
