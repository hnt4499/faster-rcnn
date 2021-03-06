import numpy as np
import torch
from torchvision.ops import nms

from .utils import from_config


def get_anchor_boxes(feature_map_size, anchor_areas, aspect_ratios):
    """
    Parameters
    ----------
    feature_map_size : tuple
        Tuple of (width, height) representing the size of the feature map.
    anchor_areas : list
        List of anchor areas.
    aspect_ratios : list
        List of anchor aspect ratios (aspect_ratio = bbox_width / bbox_height).

    Returns
    -------
    Tensor of shape (height * width * len(aspect_ratios) * len(anchor_sizes),
    4) where each anchor box is parameterized as
    (x_center, y_center, width, height).
    """
    width, height = feature_map_size

    x_coords = np.arange(width, dtype="float32")
    y_coords = np.arange(height, dtype="float32")
    anchor_boxes = np.meshgrid(x_coords, y_coords, anchor_areas, aspect_ratios)
    anchor_boxes = np.array(anchor_boxes).T.reshape(-1, 4)

    box_heights = np.sqrt(anchor_boxes[:, 2] / anchor_boxes[:, 3])
    box_widths = box_heights * anchor_boxes[:, 3]
    anchor_boxes = np.vstack(
        [anchor_boxes[:, 0], anchor_boxes[:, 1], box_widths, box_heights]
    ).T  # each box is parameterized as (x_center, y_center, width, height)
    anchor_boxes = np.transpose(
        anchor_boxes.reshape(
            len(aspect_ratios) * len(anchor_areas), width, height, 4
        ),
        (2, 1, 0, 3)
    )  # (H, W, AS * AR, 4)
    anchor_boxes = anchor_boxes.reshape(-1, 4)  # (H * W * AS * AR, 4)

    return torch.tensor(anchor_boxes, dtype=torch.float32)


def batching_wrapper(*arg_idxs):
    """Merge the boxes along the batch axis, let the `func` function do
    whatever it does, and split back to the original sizes

    Parameters
    ----------
    arg_idxs : list[int]
        List of argument indices to be wrapped.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            arg_idxs_args = [i for i in arg_idxs if i < len(args)]
            arg_idxs_kwargs = [i - len(args) for i in arg_idxs
                               if i >= len(args)]

            # Wrap positional arguments
            ele_len_all = None
            is_list_all = False
            new_args = []
            for i, arg in enumerate(args):
                if i in arg_idxs_args:
                    assert isinstance(arg, (list, tuple, torch.Tensor))
                    # If not Tensor, concatenate along the batch axis
                    is_list = isinstance(arg, (list, tuple))
                    # If there is at least one list, output is split again
                    is_list_all = is_list or is_list_all

                    if is_list:
                        ele_len = [len(ele) for ele in arg]
                        if ele_len_all is None:
                            ele_len_all = ele_len
                        else:
                            # Elements' length this must be same across inputs
                            assert ele_len_all == ele_len
                        arg = torch.cat(arg, dim=0)
                new_args.append(arg)

            # Wrap keyword arguments
            new_kwargs = {}
            for i, (key, value) in enumerate(kwargs.items()):
                if i in arg_idxs_kwargs:
                    assert isinstance(value, (list, tuple, torch.Tensor))
                    # If not Tensor, concatenate along the batch axis
                    is_list = isinstance(value, (list, tuple))
                    # If there is at least one list, output is split again
                    is_list_all = is_list or is_list_all

                    if is_list:
                        ele_len = [len(ele) for ele in value]
                        if ele_len_all is None:
                            ele_len_all = ele_len
                        else:
                            # Elements' length this must be same across inputs
                            assert ele_len_all == ele_len
                        value = torch.cat(value, dim=0)
                new_kwargs[key] = value

            # Forward
            output = func(*new_args, **new_kwargs)
            assert isinstance(output, torch.Tensor)

            # Un-wrap
            if is_list_all:
                assert ele_len_all is not None
                output = torch.split(output, ele_len_all, dim=0)

            return output
        return wrapper
    return decorator


@batching_wrapper(0, 1)
def convert_coords_to_offsets(boxes, anchor_boxes):
    """Convert (calculate) box coordinates to box offsets."""
    num_dims = boxes.ndim
    x, y, w, h = torch.split(boxes, 1, dim=-1)  # box
    x_a, y_a, w_a, h_a = torch.split(anchor_boxes, 1, dim=-1)  # anchors

    tx = (x - x_a) / w_a
    ty = (y - y_a) / h_a
    tw = torch.log(w / w_a)
    th = torch.log(h / h_a)

    boxes_offsets = torch.cat([tx, ty, tw, th], dim=num_dims - 1)

    return boxes_offsets


@batching_wrapper(0, 1)
def convert_offsets_to_coords(offsets, anchor_boxes):
    """Convert (calculate) box offsets to box coordinates."""
    num_dims = offsets.ndim
    tx, ty, tw, th = torch.split(offsets, 1, dim=-1)  # box
    x_a, y_a, w_a, h_a = torch.split(anchor_boxes, 1, dim=-1)  # anchors

    x = tx * w_a + x_a
    y = ty * h_a + y_a
    w = torch.exp(tw) * w_a
    h = torch.exp(th) * h_a

    boxes = torch.cat([x, y, w, h], dim=num_dims - 1)

    return boxes


@batching_wrapper(0)
def convert_xywh_to_xyxy(boxes):
    """
    Parameters
    ----------
    boxes : torch.Tensor
        Tensor of shape (..., 4).

    Returns
    torch.Tensor
        Tensor of shape (..., 4).
    """
    num_dims = boxes.ndim
    x, y, w, h = torch.split(boxes, 1, dim=-1)
    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2

    boxes_xyxy = torch.cat([xmin, ymin, xmax, ymax], dim=num_dims - 1)
    return boxes_xyxy


@batching_wrapper(0)
def convert_xyxy_to_xywh(boxes):
    """
    Parameters
    ----------
    boxes : torch.Tensor
        Tensor of shape (..., 4).

    Returns
    torch.Tensor
        Tensor of shape (..., 4).
    """
    num_dims = boxes.ndim
    xmin, ymin, xmax, ymax = torch.split(boxes, 1, dim=-1)
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin)
    h = (ymax - ymin)

    boxes_xywh = torch.cat([x, y, w, h], dim=num_dims - 1)
    return boxes_xywh


@batching_wrapper(0)
def box_area(boxes, mode):
    if mode not in ["xyxy", "xywh"]:
        raise ValueError("Invalid mode value. Expected one of "
                         "['xyxy', 'xywh'], got {} instead".format(mode))

    if mode == "xywh":
        boxes = convert_xywh_to_xyxy(boxes)

    xmin, ymin, xmax, ymax = torch.split(boxes, 1, dim=-1)  # box
    boxes_area = (xmax - xmin) * (ymax - ymin)

    return boxes_area.squeeze(-1)


def batched_nms(boxes, scores, iou_threshold):
    """
    "Real" batched NMS, allowing input tensors with batch dimension.

    Parameters
    ----------
    boxes : torch.Tensor
        List of tensors, each of shape (A, 4), where batch size B is the number
        of list elements. xywh formatted.
    scores : torch.Tensor
        List of tensors, each of shape (A,).
    iou_threshold : float
        IoU threshold for NMS.

    Returns
    -------
    keep_idxs : list[Tensor[int]]
        List of tensors of indices to keep for each image in the batch.
    """
    assert len(boxes) == len(scores)
    keep_idxs = []
    for boxes_per_image, scores_per_image in zip(boxes, scores):
        keep = nms(boxes_per_image, scores_per_image, iou_threshold)
        keep_idxs.append(keep)
    return keep_idxs


class Matcher:
    """
    Adapted from
        https://github.com/pytorch/vision/blob/f16322b596c7dc9e9d67d3b40907694f29e16357/torchvision/models/detection/_utils.py#L225
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    The matcher returns a tensor of size N containing the index of the
    ground-truth element m that matches to prediction n. If there is no match,
    a negative value is returned.
    """

    @from_config(main_args="evaluating->post_process->rpn->matcher",
                 requires_all=True)
    def __init__(self, high_threshold, low_threshold,
                 allow_low_quality_matches=False):
        """
        Parameters
        ----------
        high_threshold : float
            Quality values greater than or equal to this value are candidate
            matches.
        low_threshold : float
            A lower quality threshold used to stratify matches into three
            levels:
            1) matches >= high_threshold
            2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
            3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
        allow_low_quality_matches : bool
            If True, produce additional matches for predictions that have only
            low-quality match candidates. See set_low_quality_matches_ for more
            details.
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2

    def __call__(self, match_quality_matrix):
        """
        Parameters
        ----------
        match_quality_matrix : Tensor[float]
            An MxN tensor, containing the pairwise quality between M
            ground-truth elements and N predicted elements.

        Returns:
        matches : Tensor[int64]
            A tensor of size N where N[i] is a matched gt in [0, M - 1] or a
            negative value indicating that prediction i could not be matched.
        """
        assert match_quality_matrix.numel() != 0

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each
        # prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(
                matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches,
                                 match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality
        matches. Specifically, for each ground-truth find the set of
        predictions that have maximum overlap with it (including ties); for
        each prediction in that set, if it is unmatched, then match it to the
        ground-truth with which it has the highest quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including
        # ties
        gt_pred_pairs_of_highest_quality = torch.where(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]
