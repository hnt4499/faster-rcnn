import numpy as np
import torch
from torchvision import transforms
import albumentations as A

from .utils import flexible_wrapper


def get_transforms(input_size=600, transforms_mode="no"):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    trans = {
        "no_pad": Compose([
            ToTensor(),
            AlbumentationsWrapper(A.SmallestMaxSize(input_size)),
            Normalize(mean, std),
        ]),
        "simple": Compose([
            ResizeAndPad(input_size),
            ToTensor(),
            Normalize(mean, std),
        ]),
        # "light": Compose([
        #     RandomResizedCrop(train_size),
        #     RandomHorizontalFlip(),
        #     ColorJitter(0.3, 0.3, 0.3),
        #     ToTensor(),
        #     Normalize(mean, std),
        # ]),
    }

    if transforms_mode not in trans:
        raise ValueError(
            f"Invalid transformation mode. Expected one of "
            f"{list(trans.keys())}, got {transforms_mode} instead.")

    return trans[transforms_mode]


class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __getattr__(self, key):
        return getattr(self.transform, key)

    def __call__(self, *args, **kwargs):
        # Convert every Torch tensor to numpy array before feeding
        assert len(args) == 0
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                new_kwargs[key] = value.numpy()
            else:
                new_kwargs[key] = value

        # Feed
        outp = self.transform(**new_kwargs)

        # Convert every numpy array back to Torch tensor
        new_outp = {}
        for key, value in outp.items():
            if isinstance(value, np.ndarray):
                new_outp[key] = torch.from_numpy(value)
            else:
                new_outp[key] = value

        return new_outp


class Compose(transforms.Compose):
    def __call__(self, **kwargs):
        outp = kwargs
        for t in self.transforms:
            outp = t(**outp)
        return outp


class ResizeAndPad:
    """
    Resize input image to a square image and pad with black color to the
    shorter edge. Only used for input PIL image.
    """
    def __init__(self, size, interpolation=2):
        self.size = size
        self.interpolation = interpolation

    @flexible_wrapper()
    def __call__(self, image, bboxes=None):
        """Perform resizing.

        Parameters
        ----------
        image : PIL.Image
            PIL image.
        bboxes : torch.Tensor
            Tensor of shape (N, 4) in "xyxy" (COCO) format, where N is the
            number of boxes.
        """
        width, height = image.size
        if width > height:
            scale = self.size / width
        else:
            scale = self.size / height
        height = round(height * scale)
        width = round(width * scale)

        # Resize keeping aspect ratio
        image = transforms.functional.resize(
            image, (height, width), interpolation=self.interpolation)
        if bboxes is not None:
            bboxes = bboxes * scale

        # Center crop (trick)
        if height > width:
            image = transforms.functional.center_crop(image, self.size)
            if bboxes is not None:
                bboxes[:, 0] = bboxes[:, 0] + (height - width) / 2
                bboxes[:, 2] = bboxes[:, 2] + (height - width) / 2

        elif height < width:
            image = transforms.functional.center_crop(image, self.size)
            if bboxes is not None:
                bboxes[:, 1] = bboxes[:, 1] + (width - height) / 2
                bboxes[:, 3] = bboxes[:, 3] + (width - height) / 2

        outp = {"image": image, "bboxes": bboxes}
        return outp

    def __repr__(self):
        s = (f"ResizeAndPad(size={self.size}, "
             f"interpolation={self.interpolation})")
        return s


def make_class_flexible(cls, func_args=None, output_names=None,
                        rename_args={}):

    class FlexibleClass(cls):
        @flexible_wrapper(func_args=func_args, output_names=output_names,
                          rename_args=rename_args)
        def __call__(self, **kwargs):
            return super(FlexibleClass, self).__call__(**kwargs)

    FlexibleClass.__name__ = cls.__name__
    return FlexibleClass


ToTensor = make_class_flexible(
    transforms.ToTensor, func_args=["pic"], output_names=["pic"],
    rename_args={"pic": "image"})
Normalize = make_class_flexible(
    transforms.Normalize, func_args=["tensor"], output_names=["tensor"],
    rename_args={"tensor": "image"})
# RandomResizedCrop = make_class_flexible(transforms.RandomResizedCrop)
# RandomHorizontalFlip = make_class_flexible(transforms.RandomHorizontalFlip)
ColorJitter = make_class_flexible(transforms.ColorJitter)
