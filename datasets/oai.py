from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pickle
import shutil
import torch
import tqdm

from copy import deepcopy
from functools import cache
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Literal


class OAI(Dataset):
    """
    Osteoarthritis Initiative dataset.
    See https://nda.nih.gov/oai/ for more info.

    This dataset has 16,556 training images, 2,948 validation images,
    and 8,723 test images.

    The default setting has 4 target classes with 10 concepts.
    """

    C_cols = [
        "xrosfm",
        "xrscfm",
        "xrjsm",
        "xrostm",
        "xrsctm",
        "xrosfl",
        "xrscfl",
        "xrjsl",
        "xrostl",
        "xrsctl",
    ]

    y_cols = ["xrkl"]

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val", "test"] = "train",
        transform: Callable | None = None,
        processed_data_dir: Path | str = "/data/Datasets/oia_processed/",
        num_concepts: int = -1,
        use_binary_concepts: bool = False,
    ):
        """
        Parameters
        ----------
        root : Path or str
            Root directory of dataset
        split : one of {'train', 'val', 'test'}
            The dataset split to use
        transform : Callable, optional
            A function / transform that takes in a (C, H, W) image tensor and returns a
            transformed version (e.g. `torchvision.transforms.RandomCrop`)
        processed_data_dir : Path or str
            Directory where processed data is stored
            (see https://github.com/epierson9/pain-disparities).
            If the data is not found in the root directory,
            data from this directory will be re-processed and saved to the root
            directory.
        use_binary_concepts : bool
            Whether to binarize the concept values
        """
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.data_dir = self.root / self.__class__.__name__ / split
        self.data_dir.mkdir(parents=True, exist_ok=True)
        #self.data_dir = Path("/data/Datasets/oia_processed/") / split
        # Reprocess data if necessary
        self.image_data_path = self.data_dir / f"{split}.h5"
        if not self.image_data_path.exists():
            self.reprocess_dataset(self.data_dir, processed_data_dir, split)

        # Get concepts
        _, non_image_data, _ = self.load_non_image_data(split, self.data_dir)
        concepts = non_image_data[self.C_cols].values
        not_nan = ~np.isnan(concepts)
        weight_cols = [f'{col}_loss_class_wt' for col in self.C_cols]
        loss_class_wts = non_image_data[weight_cols].values

        # Bin each concept values into one of 4 classes for each concept
        if use_binary_concepts:
            concepts = np.concatenate([
                concepts < -1,
                (-1 <= concepts) & (concepts < 0),
                (0 <= concepts) & (concepts < 1),
                1 <= concepts,
            ], axis=-1)
            not_nan = not_nan.repeat(4, axis=-1)
            loss_class_wts = loss_class_wts.repeat(4, axis=-1)

        self.concepts = OAIConceptTensor(
            torch.as_tensor(concepts),
            not_nan=torch.as_tensor(not_nan),
            loss_class_wts=torch.as_tensor(loss_class_wts),
        ).float()

        # Get data & targets
        self.data = h5py.File(self.image_data_path, "r")[split]
        self.targets = non_image_data[self.y_cols].values
        self.targets = torch.as_tensor(self.targets).long().squeeze(1)
        self.num_concepts = num_concepts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = torch.from_numpy(img).float().unsqueeze(0)  # add channel dimension
        if self.transform is not None:
            img = self.transform(img)
        if self.num_concepts > 0:
            return (img, self.concepts[idx][: self.num_concepts]), self.targets[idx]
        return (img, self.concepts[idx]), self.targets[idx]

    @staticmethod
    def to_dicom_16_bit(img: np.ndarray[float]) -> np.ndarray[np.int16]:
        """
        Convert an image from z-score to DICOM 16-bit pixel values.

        Parameters
        ----------
        img : np.ndarray[float] of shape (3, H, W)
            OAI image in z-score pixel format
            (see https://github.com/epierson9/pain-disparities)

        Returns
        -------
        img : np.ndarray[int16] of shape (H, W)
            OAI image in DICOM 16-bit pixel format
        """
        mu, std, unit = 0.06786825477, 0.06260688486, 4.7422292390866816e-05
        img *= std
        img += mu
        img /= unit
        img -= 2**15  # uint16 -> int16
        img = img.round().astype(np.int16)

        # Verify that image is grayscale
        assert img.shape == (3, 1024, 1024)
        assert (img[0] == img[1]).all()
        assert (img[0] == img[2]).all()
        return img[0]

    @staticmethod
    def reprocess_dataset(
        save_dir: Path | str,
        processed_data_dir: Path | str,
        split: Literal["train", "val", "test"],
    ):
        """
        Reprocesses the OAI dataset into a HDF5 file.

        Parameters
        ----------
        save_dir : Path or str
            Directory to save the HDF5 file
        processed_data_dir : Path or str
            Directory where processed data is stored
            (see https://github.com/epierson9/pain-disparities)
        split : str
            Dataset split to use
        """
        save_dir = Path(save_dir).expanduser().resolve()
        data_dir = (
            Path(
                processed_data_dir,
                split,
                (
                    "show_both_knees_True_"
                    "downsample_factor_None_"
                    "normalization_method_our_statistics"
                ),
            )
            .expanduser()
            .resolve()
        )

        # Copy non-image data to save directory
        for filename in ("image_codes.pkl", "non_image_data.csv"):
            shutil.copyfile(data_dir / filename, save_dir / filename)
            shutil.copyfile(data_dir / filename, save_dir / filename)

        # Load image paths
        img_paths = data_dir.glob("*.npy")
        img_paths = sorted(img_paths, key=lambda p: int(p.stem.strip("image_")))

        # Process image data
        with h5py.File(save_dir / f"{split}.h5", "w") as file:
            # Cre3te dataset
            print(f"Initializing HDF5 {split} dataset ...")
            file.create_dataset(
                name=split,
                shape=(len(img_paths), 1024, 1024),
                dtype=np.int16,
                chunks=(1, 1024, 1024),
            )

            # Populate dataset
            print("Converting images to DICOM 16-bit format ...")
            for i in tqdm.tqdm(range(len(img_paths))):
                img = np.load(img_paths[i])
                img = OAI.to_dicom_16_bit(img)
                file[split][i] = img

    @classmethod
    @cache
    def load_non_image_data(
        cls,
        split: Literal["train", "val", "test"],
        data_dir: Path | str,
        C_cols: tuple[str] | None = None,
        y_cols: tuple[str] | None = None,
        zscore_C: bool = True,
        zscore_Y: bool = False,
        transform_statistics=None,
        merge_klg_01: bool = True,
        truncate_C_floats: bool = True,
        shuffle_Cs: bool = False,
        return_CY_only: bool = False,
        check: bool = True,
        verbose: bool = False,
    ):
        """
        Load the non-image data for the OAI dataset.

        The processing code here is adapted from
        https://github.com/yewsiang/ConceptBottleneck/blob/master/OAI/dataset.py.

        Parameters
        ----------
        split : one of {'train', 'val', 'test'}
            Dataset split to use
        data_dir : Path or str
            Directory containing 'image_codes.pkl' and 'non_image_data.csv'
        C_cols : tuple[str], optional
            Concept columns to use
        y_cols : tuple[str], optional
            Target columns to use
        zscore_C : bool
            Whether to standardize the concept columns
        zscore_Y : bool
            Whether to standardize the target columns
        transform_statistics : dict[str, dict[str, float]], optional
            Dictionary of transform statistics
        merge_klg_01 : bool
            Whether to merge features KLG=0 and KLG=1 (Kellgren-Lawrence grade)
        truncate_C_floats : bool
            Whether to round concept columns to the nearest integer
        shuffle_Cs : bool
            Whether to shuffle the concept data
        return_CY_only : bool
            Whether to return only the concept and target data
        check : bool
            Whether to check that the image codes match the non-image data
        verbose : bool
            Whether to print verbose output
        """
        C_cols = list(cls.C_cols if C_cols is None else C_cols)
        y_cols = list(cls.y_cols if y_cols is None else y_cols)

        # Use the train dataset to compute the transform statistics
        if transform_statistics is None and split != "train":
            _, _, transform_statistics = OAI.load_non_image_data(
                split="train",
                data_dir=data_dir,
                C_cols=tuple(C_cols),
                y_cols=tuple(y_cols),
                zscore_C=zscore_C,
                zscore_Y=zscore_Y,
                transform_statistics=None,
                merge_klg_01=merge_klg_01,
                truncate_C_floats=truncate_C_floats,
                shuffle_Cs=shuffle_Cs,
                return_CY_only=False,
                check=check,
                verbose=verbose,
            )

        non_image_data = pd.read_csv(
            data_dir / "non_image_data.csv",
            index_col=0,
            low_memory=False,
        )

        with open(data_dir / "image_codes.pkl", "rb") as file:
            image_codes = pickle.load(file)

        if check:
            assert len(non_image_data) == len(image_codes)
            for idx in range(len(non_image_data)):
                barcode = str(non_image_data.iloc[idx]["barcdbu"])
                if len(barcode) == 11:
                    barcode = "0" + barcode
                side = str(non_image_data.iloc[idx]["side"])
                code_in_df = barcode + "*" + side

                if image_codes[idx] != code_in_df:
                    raise Exception(
                        f"Barcode mismatch at index {idx},"
                        f"{image_codes[idx]} != {code_in_df}"
                    )
            if verbose:
                print(f"All {len(non_image_data)} barcodes line up.")

        # Clip xrattl from [0,3] to [0,2]. Basically only for the 2 examples
        # with Class = 3 which do not appear in train dataset
        if verbose:
            print("Truncating xrattl")
        non_image_data["xrattl"] = np.minimum(2, non_image_data["xrattl"])

        # Data processing for non-image data
        if merge_klg_01:
            if verbose:
                print("Merging KLG")
            # Merge KLG 0,1 + Convert KLG scale to [0,3]
            non_image_data["xrkl"] = np.maximum(0, non_image_data["xrkl"] - 1)

        # Truncate odd decimals
        if truncate_C_floats:
            if verbose:
                print("Truncating A floats")
            for variable in C_cols + y_cols:
                # Truncate decimals
                non_image_data[variable] = (
                    non_image_data[variable].values.astype(int).astype(float)
                )

        # Mix up the As for the training set to see if performance of KLG worsens
        if shuffle_Cs:
            if verbose:
                print("Shuffling As")
            for col in C_cols:
                N = len(non_image_data)
                permutation = np.random.permutation(N)
                non_image_data[col] = non_image_data[col].values[permutation]

        # Give weights for each class within each attribute,
        # so that it can be used to reweigh the loss
        for variable in C_cols:
            new_variable = variable + "_loss_class_wt"
            attribute = non_image_data[variable].values
            unique_classes = np.unique(attribute)
            N_total = len(attribute)
            N_classes = len(unique_classes)
            weights = np.zeros(len(attribute))
            for cls_val in unique_classes:
                belongs_to_cls = attribute == cls_val
                counts = np.sum(belongs_to_cls)
                # Since each class has 'counts',
                # the total weight allocated to each class = 1
                # weights[belongs_to_cls] = 1. / counts
                weights[belongs_to_cls] = (N_total - counts) / N_total
            non_image_data[new_variable] = weights

        # Z-scoring of the Ys
        new_transform_statistics = {}
        y_feats = None
        if zscore_Y:
            y_feats = deepcopy(non_image_data[y_cols].values)
            for i in range(len(y_cols)):
                not_nan = ~np.isnan(y_feats[:, i])
                if transform_statistics is None:
                    std = np.std(y_feats[not_nan, i], ddof=1)
                    mu = np.mean(y_feats[not_nan, i])
                    new_transform_statistics[y_cols[i]] = {"mu": mu, "std": std}
                else:
                    std = transform_statistics[y_cols[i]]["std"]
                    mu = transform_statistics[y_cols[i]]["mu"]
                if verbose:
                    print(
                        f"Z-scoring additional feature {y_cols[i]}",
                        f"with mean {mu} and std {std}",
                    )
                non_image_data[f"{y_cols[i]}_original"] = y_feats[:, i]
                non_image_data[y_cols[i]] = (y_feats[:, i] - mu) / std
                y_feats[:, i] = non_image_data[y_cols[i]]

        # Z-scoring of the attributes
        C_feats = None
        if zscore_C:
            C_feats = deepcopy(non_image_data[C_cols].values)
            for i in range(len(C_cols)):
                not_nan = ~np.isnan(C_feats[:, i])
                if transform_statistics is None:
                    std = np.std(C_feats[not_nan, i], ddof=1)
                    mu = np.mean(C_feats[not_nan, i])
                    new_transform_statistics[C_cols[i]] = {"mu": mu, "std": std}
                else:
                    std = transform_statistics[C_cols[i]]["std"]
                    mu = transform_statistics[C_cols[i]]["mu"]
                if verbose:
                    print(
                        f"Z-scoring additional feature {C_cols[i]}",
                        f"with mean {mu} and std {std}",
                    )
                non_image_data[f"{C_cols[i]}_original"] = C_feats[:, i]
                non_image_data[C_cols[i]] = (C_feats[:, i] - mu) / std
                C_feats[:, i] = non_image_data[C_cols[i]]

        if return_CY_only:
            if y_feats is None:
                y_feats = deepcopy(non_image_data[y_cols].values)
            if C_feats is None:
                C_feats = deepcopy(non_image_data[C_cols].values)
            return C_feats, y_feats

        return data_dir, non_image_data, new_transform_statistics


class OAIConceptTensor(torch.Tensor):
    """
    Tensor subclass with extra attributes.
    """

    EXTRA_ATTRIBUTES = ["not_nan", "loss_class_wts"]

    def __new__(cls, *args, **kwargs):
        obj_kwargs = {k: v for k, v in kwargs.items() if k not in cls.EXTRA_ATTRIBUTES}
        obj = super().__new__(cls, *args, **obj_kwargs)
        for attr_name in cls.EXTRA_ATTRIBUTES:
            attr_value = kwargs.get(attr_name, None)
            setattr(obj, attr_name, attr_value)

        return obj

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        def recursive_getattr(x, attr_name):
            if isinstance(x, tuple):
                return tuple(recursive_getattr(item, attr_name) for item in x)
            elif isinstance(x, list):
                return [recursive_getattr(item, attr_name) for item in x]
            elif isinstance(x, dict):
                return {k: recursive_getattr(v, attr_name) for k, v in x.items()}
            elif isinstance(x, set):
                return {recursive_getattr(item, attr_name) for item in x}
            return getattr(x, attr_name, x)

        out = super().__torch_function__(func, types, args=args, kwargs=kwargs)
        if isinstance(out, cls):
            kwargs = {k: v for k, v in kwargs.items() if k != "out"}
            for attr_name in cls.EXTRA_ATTRIBUTES:
                attr_value = super().__torch_function__(
                    func,
                    types,
                    args=recursive_getattr(args, attr_name),
                    kwargs=kwargs,
                )
                setattr(out, attr_name, attr_value)

        return out
