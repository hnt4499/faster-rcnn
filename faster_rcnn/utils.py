import numpy as np
import torch


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


def convert_coords_to_offsets(boxes, anchor_boxes):
    """Convert (calculate) box coordinates to box offsets."""
    is_list = isinstance(boxes, (list, tuple))
    if is_list:
        num_boxes = [len(box) for box in boxes]
        boxes = torch.cat(boxes, dim=0)

        if not isinstance(anchor_boxes, (list, tuple)):
            anchor_boxes = [
                anchor_boxes.clone() for _ in range(len(num_boxes))]
        anchor_boxes = torch.cat(anchor_boxes, dim=0)

    num_dims = boxes.ndim
    x, y, w, h = torch.split(boxes, 1, dim=-1)  # box
    x_a, y_a, w_a, h_a = torch.split(anchor_boxes, 1, dim=-1)  # anchors

    tx = (x - x_a) / w_a
    ty = (y - y_a) / h_a
    tw = torch.log(w / w_a)
    th = torch.log(h / h_a)

    boxes_offsets = torch.cat([tx, ty, tw, th], dim=num_dims - 1)

    if is_list:
        return torch.split(boxes_offsets, num_boxes, dim=0)
    return boxes_offsets


def convert_offsets_to_coords(offsets, anchor_boxes):
    """Convert (calculate) box offsets to box coordinates."""
    is_list = isinstance(offsets, (list, tuple))
    if is_list:
        num_boxes = [len(box) for box in offsets]
        offsets = torch.cat(offsets, dim=0)

        if not isinstance(anchor_boxes, (list, tuple)):
            anchor_boxes = [
                anchor_boxes.clone() for _ in range(len(num_boxes))]
        anchor_boxes = torch.cat(anchor_boxes, dim=0)

    num_dims = offsets.ndim
    tx, ty, tw, th = torch.split(offsets, 1, dim=-1)  # box
    x_a, y_a, w_a, h_a = torch.split(anchor_boxes, 1, dim=-1)  # anchors

    x = tx * w_a + x_a
    y = ty * h_a + y_a
    w = torch.exp(tw) * w_a
    h = torch.exp(th) * h_a

    boxes = torch.cat([x, y, w, h], dim=num_dims - 1)

    if is_list:
        return torch.split(boxes, num_boxes, dim=0)
    return boxes


def smooth_l1_loss(input, target, beta=1. / 9, size_average=False):
    """Smmoth L1 loss, as defined in the Fast R-CNN paper.
        Girshick, R. (2015). Fast R-CNN.
    """
    diff = torch.abs(input - target)
    mask = (diff < beta)
    loss = torch.where(mask, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    if size_average:
        return loss.mean()
    return loss.sum()


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


def index_argsort(x, index, dim=-1):
    """Multi-indexer over multiple dimensions. The `index` tensor could be, for
    example, result of the `argsort` function. The first n shape of `x` tensor
    must be equal to the shape of `index`, i.e.,
        x.shape[index.ndim] == index.shape

    Parameters
    ----------
    dim : int
        Dimension over which argsort was performed.

    """
    x = x.transpose(0, dim)
    index = index.transpose(0, dim)
    if index.ndim < x.ndim:
        diff = x.ndim - index.ndim
        s = index.shape
        s = list(s) + [1] * diff
        index = index.view(*s)

    idxs = []
    for i, dim_ in enumerate(x.shape[1:][::-1]):
        idx = torch.arange(dim_)
        s = idx.shape
        s = list(s) + [1] * i
        idx = idx.view(*s)
        idxs.append(idx)

    idxs = [index] + idxs[::-1]
    x = x[tuple(idxs)].transpose(0, dim)
    return x


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
