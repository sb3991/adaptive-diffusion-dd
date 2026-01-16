# An Adaptive Sampling Framework for Diffusion based Dataset Distillation

Official Repository for the AAAI 2026 Accepted Paper
“An Adaptive Sampling Framework for Diffusion-based Dataset Distillation with High Fidelity and Diversity”

This repository will contain the official implementation of our adaptive sampling–based dataset distillation framework, which leverages diffusion models to generate compact yet high-quality synthetic datasets.

The full paper will be available soon on the AAAI 2026 proceedings.

### Requirements

```
conda env create -f environment.yml python=3.11
conda activate my_iccv_project
pip install -r requirements.txt

```

### How to Run
To execute the main script, use the following command:

python ./main.py \
    --subset "cifar10" \
    --arch-name "resnet18_modified" \
    --num-crop 5 \
    --mipc 10 \
    --ipc 10 \
    --stud-name "conv3" \
    --lam 0.0 \
    --n_iter 5 \
    --init_points 15

🔹 Arguments Description
Argument	Description
--subset	Dataset to use (e.g., "cifar10")
--arch-name	Model architecture (e.g., "resnet18_modified")
--num-crop	Number of crops per image
--mipc	Maximum images per class
--ipc	Images per class
--stud-name	Student model name (e.g., "conv3")
--re-epochs	Number of re-training epochs
--lam	Regularization weight (e.g., 0.0)
--n_iter	Number of iterations for BO
--init_points	Initial number of points for BO

### Storage Format for Raw Datasets

All our raw datasets, including those like ImageNet-1K and CIFAR10, store their training and validation components in the following format to facilitate uniform reading using a standard dataset class method:

```
/path/to/dataset/
├── 00000/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
├── 00001/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
├── 00002/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
```
## 📂 Code Structure

## 📂 Project Structure

├── main.py                 # Main script for training and evaluation
├── argument.py             # Parses command-line arguments for main.py
├── data/
│   └── additional_trained_models/   # Pretrained teacher model checkpoints
├── Data/
│   ├── dataset             # Directory for storing datasets
│   └── labelprompt/        # Label prompt text files for datasets
├── synthesize/
│   ├── main_0.py           # Selects initial images randomly
│   ├── main_2.py           # Uses the teacher model to generate the final distilled dataset
│   ├── main_GUIDE.py       # Creates a synthetic dataset via diffusion from initial images
│   ├── models.py           # Model definitions for synthesis
│   ├── utils.py            # Utility functions for the synthesis process
│   └── LocalStableDiffusion.py  # Diffusion process incorporating repulsion
├── validation/ 
│   ├── main_3.py           # Functions for evaluation
│   └── utils.py            # Utility functions for evaluation
