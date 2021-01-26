from PIL import Image

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from faster_rcnn.utils import convert_xyxy_to_xywh


def get_dataset(name):
    """Get dataset initializer from its format"""
    datasets = {
        "voc": VOCDataset
    }
    if name not in datasets:
        raise ValueError(f"Invalid dataset format. Expected one of "
                         f"{list(datasets.keys())}, got {name} instead.")
    return datasets[name]


class VOCDataset(Dataset):
    """PASCAL VOC dataset.

    Parameters
    ----------
    info_path : str or list of str
        Path(s) to csv file(s) containing dataset information.
    difficult : bool
        Whether to keep difficult examples. (default: False)
    """
    def __init__(self, info_path, difficult=False):
        super(VOCDataset, self).__init__()
        self.info_path = info_path

        # Read info
        if isinstance(info_path, str):
            self.df = pd.read_csv(info_path)
        else:
            self.df = pd.concat([pd.read_csv(path) for path in info_path])
            self.df = self.df.reset_index(drop=True)

        self.df["bboxes"] = self.df["bboxes"].apply(
            lambda x: list(map(int, x.split(","))))
        self.df["labels"] = self.df["labels"].apply(
            lambda x: list(map(int, x.split(","))))
        self.df["difficults"] = self.df["difficults"].apply(
            lambda x: list(map(int, x.split(","))) if x is not np.nan else x)

        # Filter out difficult examples
        if not difficult:
            self._remove_difficults()

    def _remove_difficults(self):
        sub_df = self.df[["bboxes", "labels", "difficults"]]

        def transform(row):
            bboxes, labels, difficults = row[
                ["bboxes", "labels", "difficults"]]
            if difficults is np.nan:
                return row

            assert len(bboxes) % 4 == 0
            num_boxes = len(bboxes) // 4
            assert num_boxes == len(labels) == len(difficults)

            new_bboxes, new_labels = [], []

            for i in range(num_boxes):
                if difficults[i] == 0:
                    bbox = bboxes[i * 4:(i + 1) * 4]
                    new_bboxes.extend(bbox)
                    label = labels[i]
                    new_labels.append(label)

            new_difficults = [0] * len(new_labels)
            new_row = pd.Series({
                "bboxes": new_bboxes, "labels": new_labels,
                "difficults": new_difficults
            })
            return new_row

        sub_df = sub_df.apply(transform, axis=1)
        self.df[["bboxes", "labels", "difficults"]] = sub_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_info = self.df.iloc[index]
        bboxes, labels, img_path = img_info[["bboxes", "labels", "img_path"]]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return img, bboxes, labels


def collate_fn(batch, transforms):
    """Collate function to be passed to the PyTorch dataloader.

    Parameters
    ----------
    batch : list
        Uncollated batch of size `batch_size`.
    device : str or torch.device
        Current working device.
    transforms : callable
        Transformations to be applied on input PIL images.
    """
    images, bboxess, labelss = zip(*batch)

    # Bounding boxes
    bboxess_ = []
    for bboxes in bboxess:
        assert len(bboxes) % 4 == 0
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes = bboxes.view(-1, 4)
        bboxess_.append(bboxes)
    bboxess = bboxess_

    # Labels
    labelss_ = []
    for labels, bboxes in zip(labelss, bboxess):
        assert len(labels) == len(bboxes)
        labels = torch.tensor(labels, dtype=torch.float32)
        labelss_.append(labels)
    labelss = labelss_

    # Transformations
    images_trans, bboxess_trans, labelss_trans = [], [], []
    for image, bboxes, labels in zip(images, bboxess, labelss):
        outp = transforms(image=image, bboxes=bboxes, class_labels=labels)
        images_trans.append(outp["image"])
        bboxess_trans.append(convert_xyxy_to_xywh(outp["bboxes"]))
        labelss_trans.append(outp["class_labels"])

    images_trans = torch.stack(images_trans, dim=0)
    return images_trans, bboxess_trans, labelss_trans
