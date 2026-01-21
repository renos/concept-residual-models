from __future__ import annotations

import pickle
import random
import torch

from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from typing import Callable, Literal
from torchvision import datasets
import numpy as np
import h5py
import os
import time
from skimage.io import imread
import torchxrayvision as xrv

from PIL import Image


class MIMIC_CXR_old(Dataset):
    """
    MIMIC CXR Dataset
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        subset = "cardiomegaly",
        transform: Callable | None = None,
        download: bool = False,
        *args,
        **kwargs,
    ):
        self.root = root

        self.split = split
        v_split = "valid" if split == "val" else split
        self.image_data_path = f"{root}/{subset}/dataset_g/dataset_g/{v_split}.h5"
        print(self.image_data_path)
        self.data = h5py.File(self.image_data_path, "r")[v_split]

        all_data = self.data[0]
        concepts = all_data[512*512*3+1:]
        self.transform = transform
        self.num_concepts = concepts.shape[-1]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        all_data = self.data[idx]
        img = all_data[:512*512*3].reshape(3, 512, 512)
        label = all_data[512*512*3]
        concepts = torch.tensor(all_data[512*512*3+1:].astype(float), dtype=torch.float32)
        #img = img.float()
        # Assuming arr_8bit is your NumPy array
        arr_8bit = (img / 256).astype(np.uint8)

        # Convert the NumPy array to a PIL image
        img = Image.fromarray(arr_8bit.transpose(1, 2, 0))
        if self.transform is not None:
            img = self.transform(img)
        
        return (img, concepts), int(label)



# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# from typing import Literal, Callable
# from pathlib import Path
# from tqdm import tqdm

# def convert_h5_to_numpy(h5_path: str, output_root: str, subset: str, split: str):
#     """
#     Converts H5 dataset to numpy files organized by split first, then subset
#     """
#     import h5py
    
#     output_root = Path(output_root)
#     split_dir = output_root / split
    
#     # Create directory structure
#     img_dir = split_dir / 'images'  # Images per split
#     metadata_dir = split_dir / 'metadata' / subset  # Split/subset metadata
    
#     # Create chunk directories
#     for i in range(10):
#         (img_dir / f"chunk_{i:02d}").mkdir(exist_ok=True, parents=True)
#         (metadata_dir / f"chunk_{i:02d}").mkdir(exist_ok=True, parents=True)
    
#     # Convert H5 to numpy files
#     with h5py.File(h5_path, 'r') as f:
#         split_name = list(f.keys())[0]
#         dataset = f[split_name]
        
#         print(f"Converting {len(dataset)} samples for {split}/{subset}...")
#         for idx in tqdm(range(len(dataset))):
#             chunk_idx = idx % 10
#             data = dataset[idx]
            
#             # Save image (per split)
#             img = data[:512*512*3].reshape(3, 512, 512)
#             img_path = img_dir / f"chunk_{chunk_idx:02d}" / f"{idx:08d}.npy"
#             if img_path.exists():
#                 # # Load existing image and verify it matches
#                 # existing_img = np.load(img_path)
#                 # if not np.array_equal(img, existing_img):
#                 #     raise ValueError(f"Image mismatch detected for {img_path}! This indicates data inconsistency.")
#                 pass
#             else:
#                 np.save(img_path, img)
            
#             # Save metadata (split and subset specific)
#             metadata = data[512*512*3:]  # includes both label and concepts
#             metadata_path = metadata_dir / f"chunk_{chunk_idx:02d}" / f"{idx:08d}.npy"
#             np.save(metadata_path, metadata)

# class MIMIC_CXR(Dataset):
#     """
#     MIMIC CXR Dataset organized by split first, then subset
#     """
#     def __init__(
#         self,
#         root: str,
#         split: Literal["train", "val", "test"] = "train",
#         subset: str = "cardiomegaly",
#         transform: Callable | None = None,
#         convert_if_needed: bool = True,
#         *args,
#         **kwargs,
#     ):
#         self.root = Path(root)
#         self.split = "valid" if split == "val" else split
#         self.subset = subset
#         self.transform = transform
        
#         # Setup paths
#         self.h5_path = self.root / subset / "dataset_g" / "dataset_g" / f"{self.split}.h5"
        
#         # Split is the top-level directory
#         self.split_dir = self.root / self.split
#         self.img_dir = self.split_dir / 'images'
#         self.metadata_dir = self.split_dir / 'metadata' / subset
        
#         # Convert if needed
#         if convert_if_needed and not self.metadata_dir.exists():
#             print(f"Converting {self.split} set for {subset}...")
#             convert_h5_to_numpy(str(self.h5_path), str(self.root), subset, self.split)
        
#         # Get file paths from metadata (which determines available samples)
#         self.file_indices = []
#         for i in range(10):
#             chunk_pattern = f"chunk_{i:02d}/*.npy"
#             chunk_files = sorted(self.metadata_dir.glob(chunk_pattern))
#             self.file_indices.extend([path.stem for path in chunk_files])
        
#         if not self.file_indices:
#             raise RuntimeError(f"No files found for {self.split}/{subset}")
        
#         # Get number of concepts from first file
#         first_metadata = np.load(next(self.metadata_dir.glob("chunk_00/*.npy")))
#         first_file = next(entry.path for entry in os.scandir(self.metadata_dir / "chunk_00") if entry.is_file())
#         first_metadata = np.load(first_file)
#         self.num_concepts = len(first_metadata) - 1 
    
#     def __len__(self):
#         return len(self.file_indices)
    
#     def __getitem__(self, idx):
#         # Get file paths for this index
#         #t1 = time.time()
#         file_id = self.file_indices[idx]
#         chunk_idx = f"{int(file_id) % 10:02d}"
        
#         # Load data
#         img = np.load(self.img_dir / f"chunk_{chunk_idx}" / f"{file_id}.npy")
#         metadata = np.load(self.metadata_dir / f"chunk_{chunk_idx}" / f"{file_id}.npy")
        
#         # Split metadata into label and concepts
#         label = metadata[0]
#         concepts = metadata[1:]
        
#         # Convert image to 8-bit and process
#         arr_8bit = (img / 256).astype(np.uint8)
#         img = Image.fromarray(arr_8bit.transpose(1, 2, 0))
        
#         if self.transform is not None:
#             img = self.transform(img)
#         # print("Loading")
#         # print(time.time()-t1)
#         return (img, torch.tensor(concepts, dtype=torch.float32)), int(label)

# def convert_dataset(root_path, subsets=None):
#     """
#     Utility function to convert all splits for specified subsets
#     """
#     if subsets is None:
#         subsets = ["cardiomegaly"]  # default subset
        
#     for subset in subsets:
#         for split in ["train", "valid", "test"]:
#             h5_path = Path(root_path) / subset / "dataset_g" / "dataset_g" / f"{split}.h5"
#             if h5_path.exists():
#                 print(f"\nConverting {split} split for {subset}...")
#                 convert_h5_to_numpy(str(h5_path), str(root_path), subset, split)


import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Literal, Callable
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
import os
import multiprocessing as mp
import h5py

def process_sample(args) -> None:
    """
    Process a single sample from the dataset
    """
    h5_path, img_dir, metadata_dir, idx, split_name = args
    
    # Get first digit for directory organization
    first_digit = str(idx)[0]
    
    # Read data for this index only
    with h5py.File(h5_path, 'r') as f:
        data = f[split_name][idx]
    
    # Process image
    img = data[:512*512*3].reshape(3, 512, 512)
    img_uint8 = (img / 256).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8.transpose(1, 2, 0))
    resize_transform = transforms.Resize(256)
    img_resized = resize_transform(img_pil)
    img_final = np.array(img_resized, dtype=np.uint8).transpose(2, 0, 1)
    
    # Save image with original index in digit-based directory
    img_path = os.path.join(img_dir, first_digit, f"{idx:08d}.npy")
    if not os.path.exists(img_path):
        np.save(img_path, img_final)
    
    # Save metadata with original index in digit-based directory
    metadata = data[512*512*3:]
    metadata_path = os.path.join(metadata_dir, first_digit, f"{idx:08d}.npy")
    np.save(metadata_path, metadata)

def convert_h5_to_numpy(h5_path: str, output_root: str, subset: str, split: str):
    """
    Converts H5 dataset to numpy files organized by split first, then subset
    Files are organized in directories based on their first index digit
    """
    # Replace the data path prefix with home directory
    output_root = str(output_root).replace('/data/Datasets', '/home/renos')
    output_root = Path(output_root)
    split_dir = output_root / split
    
    # Create directory structure
    img_dir = str(split_dir / 'images')
    metadata_dir = str(split_dir / 'metadata' / subset)
    
    # Create digit-based directories (0-9)
    for digit in range(10):
        os.makedirs(os.path.join(img_dir, str(digit)), exist_ok=True)
        os.makedirs(os.path.join(metadata_dir, str(digit)), exist_ok=True)
    
    # Get dataset size and split name
    with h5py.File(h5_path, 'r') as f:
        split_name = list(f.keys())[0]
        total_samples = len(f[split_name])
    
    print(f"Converting {total_samples} samples for {split}/{subset}...")
    
    # Prepare argument tuples without loading the data
    process_args = [(h5_path, img_dir, metadata_dir, idx, split_name) 
                   for idx in range(total_samples)]
    
    # Use multiprocessing to process samples
    num_cpus = 256
    with mp.Pool(num_cpus) as pool:
        list(tqdm(
            pool.imap(process_sample, process_args, chunksize=50),
            total=total_samples,
            desc="Processing samples"
        ))

class MIMIC_CXR(Dataset):
    """
    MIMIC CXR Dataset organized by split first, then subset
    Files are organized in directories based on their first index digit
    """
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        subset: str = "cardiomegaly",
        transform: Callable | None = None,
        convert_if_needed: bool = True,
        *args,
        **kwargs,
    ):
        self.root = Path(str(root).replace('/data/Datasets', '/home/renos'))
        self.split = "valid" if split == "val" else split
        self.subset = subset
        self.transform = transform
        
        # Setup paths
        self.h5_path = Path(str(root)) / subset / "dataset_g" / "dataset_g" / f"{self.split}.h5"
        
        # Split is the top-level directory
        self.split_dir = self.root / self.split
        self.img_dir = self.split_dir / 'images'
        self.metadata_dir = self.split_dir / 'metadata' / subset
        
        # Convert if needed
        if convert_if_needed and not self.metadata_dir.exists():
            print(f"Converting {self.split} set for {subset}...")
            convert_h5_to_numpy(str(self.h5_path), str(self.root), subset, self.split)
        
        # Get total number of samples from h5 file
        with h5py.File(self.h5_path, 'r') as f:
            split_name = list(f.keys())[0]
            self.total_samples = len(f[split_name])
        
        # Get number of concepts from first metadata file
        first_metadata = np.load(self.metadata_dir / "0" / f"{0:08d}.npy")
        self.num_concepts = len(first_metadata) - 1
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Get first digit for directory lookup
        first_digit = str(idx)[0]
        
        # Load data using the index from the appropriate directory
        img = np.load(self.img_dir / first_digit / f"{idx:08d}.npy")
        metadata = np.load(self.metadata_dir / first_digit / f"{idx:08d}.npy")
        
        # Split metadata into label and concepts
        label = metadata[0]
        concepts = metadata[1:]
        
        # Convert image for processing
        img = Image.fromarray(img.transpose(1, 2, 0))
        
        if self.transform is not None:
            img = self.transform(img)
            
        return (img, torch.tensor(concepts, dtype=torch.float32)), int(label)
    


import pandas as pd


class CheX_Dataset(Dataset):
    """CheXpert Dataset

    Citation:

    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and
    Expert Comparison. Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko,
    Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad
    Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong,
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson,
    Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng.
    https://arxiv.org/abs/1901.07031

    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/

    A small validation set is provided with the data as well, but is so tiny,
    it is not included here.
    """

    def __init__(self,
                 split="train",
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,

                 ):

        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.split = split
        if split == "train":
            self.df = pd.read_csv("/data/Datasets/mimic_cxr_processed/csvs/cardiomegaly/master_tot_15000.csv")
        else:
            self.df = pd.read_csv("/data/Datasets/Stanford_CheXpert_Dataset/CheXpert-v1.0-small/valid.csv")
            self.df = self.df.fillna(0)
            header = "Path" if split == "val" else "image_names"
            self.df = self.df.set_index(header)
            self.labels = [label.replace(" ", "_") for label in list(self.df.columns.values)[4:]]
            self.df.columns = list(self.df.columns.values)[0:4] + self.labels
            self.label_idx = self.labels.index("Cardiomegaly")
        self.parent_dir = "/data/Datasets/Stanford_CheXpert_Dataset/"
        self.transform = transform

        if split == "train":
            self.concepts = np.array(torch.load('/data/Datasets/mimic_cxr_processed/out/stanford_cxr/t/lr_0.1_epochs_90_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/sample_size_15000/cardiomegaly/train_sample_size_15000_sub_select_attributes.pt'))

        else:
            # Load the PyTorch file
            self.concepts = np.array(torch.load('/data/Datasets/mimic_cxr_processed/out/stanford_cxr/t/lr_0.1_epochs_90_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/sample_size_15000/cardiomegaly/test_sample_size_15000_sub_select_attributes.pt'))


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.split == "val":
            image = Image.open(os.path.join(self.parent_dir, self.df.index[idx]))
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)

            label = np.zeros(len(self.labels), dtype=int)
            for i in range(0, len(self.labels)):
                label[i] = self.df[self.labels[i].strip()].iloc[idx].astype('int')
                if label[i] == -1:
                    label[i] = self.uncertain
            return (image, torch.tensor(self.concepts[idx], dtype=torch.float32)), int(torch.FloatTensor(label)[self.label_idx])
        else:
            image_names = self.df.loc[idx, "image_names"]
            disease_label = self.df.loc[idx, "GT"]
            image = Image.open(os.path.join(self.parent_dir, image_names))
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)

            return (image, torch.tensor(self.concepts[idx], dtype=torch.float32)), int(disease_label)
