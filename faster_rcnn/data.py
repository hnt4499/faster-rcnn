from PIL import Image

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .utils import from_config
from .registry import register


@register("dataset")
class VOCDataset(Dataset):
    """PASCAL VOC dataset.

    Parameters
    ----------
    info_path : str or list of str
        Path(s) to csv file(s) containing dataset information.
    difficult : bool
        Whether to keep difficult examples. (default: False)
    """
    @from_config(requires_all=True)
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
        img_width, img_height = img_info[["width", "height"]]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return img, bboxes, labels, img_width, img_height


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
    images, bboxess, labelss, img_widths, img_heights = zip(*batch)

    # Turn image widths and heights into pseudo bounding boxes to retrieve back
    # later
    img_widths = torch.tensor(img_widths, dtype=torch.float32)
    img_heights = torch.tensor(img_heights, dtype=torch.float32)
    x1_or_y1 = torch.zeros_like(img_widths, dtype=torch.float32)
    pseudo_img_sizes = torch.stack(
        [x1_or_y1, x1_or_y1, img_widths, img_heights], dim=-1)

    # Bounding boxes
    bboxess_ = []
    for bboxes, pseudo_img_size in zip(bboxess, pseudo_img_sizes):
        assert len(bboxes) % 4 == 0
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes = bboxes.view(-1, 4)
        # Concat with (pseudo) image size
        bboxes = torch.cat((bboxes, pseudo_img_size[None, :]), dim=0)
        bboxess_.append(bboxes)
    bboxess = bboxess_

    # Labels
    labelss_ = []
    for labels, bboxes in zip(labelss, bboxess):
        assert len(labels) == len(bboxes) - 1  # accounts for pseudo image size
        labels = torch.tensor(labels, dtype=torch.float32)
        labelss_.append(labels)
    labelss = labelss_

    # Transformations
    images_trans, bboxess_trans, labelss_trans = [], [], []
    image_boundaries = []
    for image, bboxes, labels in zip(images, bboxess, labelss):
        outp = transforms(image=image, bboxes=bboxes, class_labels=labels)
        images_trans.append(outp["image"])
        labelss_trans.append(outp["class_labels"])

        bboxes_trans = outp["bboxes"]
        bboxess_trans.append(bboxes_trans[:-1, :])
        image_boundaries.append(bboxes_trans[-1, :])

    images_trans = torch.stack(images_trans, dim=0)
    image_boundaries = torch.stack(image_boundaries, dim=0)
    return images_trans, bboxess_trans, labelss_trans, image_boundaries
