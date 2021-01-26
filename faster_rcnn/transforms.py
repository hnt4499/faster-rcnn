import inspect
from functools import partial

import numpy as np
import torch
from torchvision import transforms
import albumentations as A


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


class IDict(dict):
    def __getitem__(self, key):
        if key not in self:
            return key
        return super(IDict, self).__getitem__(key)


def flexible_wrapper(func, input_args=None, output_names=None, rename_args={}):
    """Wraps transformation calls (`__call__`) so that it can take any keyword
    arguments. Note that the function is restricted to pass only keyword
    arguments to the `__call__` function, but NOT positinal arguments. Also,
    if `output_names` is not specified, the output of each `__call__` function
    must be again a dict to update the existing `kwargs` (see below).

    Parameters
    ----------
    output_names : list of str or None
        If not specified, the output of `func` must be a dict.
        If specified, map each of `output_names` to each the output of `func`
        to form a new dict.
    rename_args : dict
        A dict containing arguments in source function and their respective
        name in target function. Used to rename args for consistency.
    """
    if input_args is not None:
        func_args = input_args
    else:
        func_args = inspect.getfullargspec(func)[0][1:]  # not including `self`

    to_new_args = IDict(rename_args)
    assert len(set(to_new_args.values())) == len(to_new_args)

    def wrapper(self, *args, **kwargs):
        if len(args) > 0:
            raise TypeError(
                "You must pass all arguments as keyword arguments.")
        # Get actual args
        actual_kwargs = {}
        for arg in func_args:
            if to_new_args[arg] in kwargs:
                actual_kwargs[arg] = kwargs[to_new_args[arg]]
                del kwargs[to_new_args[arg]]

        # Call function
        outp = func(self, **actual_kwargs)

        # Post process
        if output_names is not None:
            if len(output_names) == 1:
                outp = {output_names[0]: outp}
            else:
                assert len(output_names) == len(outp)
                outp = dict(zip(output_names, outp))

        # Update exising kwargs
        kwargs.update(outp)

        # Convert to new args
        new_outp = {}
        for k, v in kwargs.items():
            new_outp[to_new_args[k]] = v

        return new_outp

    return wrapper


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

    @flexible_wrapper
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


def make_class_flexible(cls, input_args=None, output_names=None,
                        rename_args={}):
    wrapper = partial(
        flexible_wrapper, input_args=input_args, output_names=output_names,
        rename_args=rename_args)

    class FlexibleClass(cls):
        @wrapper
        def __call__(self, **kwargs):
            return super(FlexibleClass, self).__call__(**kwargs)

    FlexibleClass.__name__ = cls.__name__
    return FlexibleClass


ToTensor = make_class_flexible(
    transforms.ToTensor, input_args=["pic"], output_names=["pic"],
    rename_args={"pic": "image"})
Normalize = make_class_flexible(
    transforms.Normalize, input_args=["tensor"], output_names=["tensor"],
    rename_args={"tensor": "image"})
# RandomResizedCrop = make_class_flexible(transforms.RandomResizedCrop)
# RandomHorizontalFlip = make_class_flexible(transforms.RandomHorizontalFlip)
ColorJitter = make_class_flexible(transforms.ColorJitter)
