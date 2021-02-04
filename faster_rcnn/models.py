import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import box_iou

from .utils import (smooth_l1_loss, index_argsort, apply_mask, index_batch,
                    batching, from_config)
from .box_utils import (
    get_anchor_boxes, convert_xyxy_to_xywh, convert_xywh_to_xyxy,
    convert_coords_to_offsets, convert_offsets_to_coords,
    box_area, batched_nms, Matcher
)
from .registry import registry


def get_layers(model_name):
    """Get layers information."""
    layers_mapping = {
        "resnet": ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2",
                   "layer3", "layer4"],
        "inception_v3": [
            # Block 0: input to maxpool1
            "Conv2d_1a_3x3",
            "Conv2d_2a_3x3",
            "Conv2d_2b_3x3",
            ("MaxPool_1", nn.MaxPool2d(kernel_size=3, stride=2)),

            # Block 1: maxpool1 to maxpool2
            "Conv2d_3b_1x1",
            "Conv2d_4a_3x3",
            ("MaxPool_2", nn.MaxPool2d(kernel_size=3, stride=2)),

            # Block 2: maxpool2 to aux classifier
            "Mixed_5b",
            "Mixed_5c",
            "Mixed_5d",
            "Mixed_6a",
            "Mixed_6b",
            "Mixed_6c",
            "Mixed_6d",
            "Mixed_6e",

            # Block 3
            "Mixed_7a",
            "Mixed_7b",
            "Mixed_7c",
        ]
    }
    return layers_mapping[model_name]


def get_roi_stat(model_name):
    """Get the RoI pooling layer statistics, which is needed in the computation
    of RoI pooling (i.e., Spatial Pyramid Pooling, SPP) layer.
    Reference:
        He, K., Zhang, X., Ren, S., & Sun, J. (2014). Spatial Pyramid Pooling
        in Deep Convolutional Networks for Visual Recognition.
    (see section "Mapping a Window to Feature Maps" in the appendix).

    Note that since the value for Inception V3 is manually inferred, using it
    as backbone can give inferior results.

    Returns
    -------
    A callable that return pooled value (not rounded) of a given coordinate.
    """
    roi_stats = {
        "resnet": lambda x: x / 32,
        "inception_v3": lambda x: (x - 70) / 31,  # manually inferred
    }
    return roi_stats[model_name]


def get_orig_stat(model_name):
    """Same as `get_roi_stat`, but in reverse direction."""
    orig_stats = {
        "resnet": lambda x: x * 32,
        "inception_v3": lambda x: x * 31 + 70,  # manually inferred
    }
    return orig_stats[model_name]


class BackboneModel(nn.Module):
    """Backbone model of Faster R-CNN."""
    @from_config(main_args="model->backbone", requires_all=True)
    def __init__(self, model_name="resnet50", freeze_all=False):
        """Build the model.

        Parameters
        ----------
        model_name : str
            Name of the model. Should be a retrievable from
            `torchvision.models`.

        """
        super(BackboneModel, self).__init__()
        self.model_name = model_name
        self.freeze_all = freeze_all
        if "res" in model_name:
            self.model_type = "resnet"
        else:
            self.model_type = model_name

        # Get model
        model_initializer = getattr(models, model_name, None)
        if model_initializer is None:
            raise RuntimeError(f"Model named {model_name} not found.")
        model = model_initializer(pretrained=True)

        # Get layers
        layers = get_layers(self.model_type)
        for layer in layers:
            # Callable modules
            if isinstance(layer, tuple):
                name, module = layer
                setattr(self, name, module)
            else:
                module = getattr(model, layer)
                setattr(self, layer, module)

        # Freeze all
        if freeze_all:
            for params in self.parameters():
                params.requires_grad = False

        # Feed a dummy input to get the number of channels of the outputs.
        dummy_input = torch.zeros(1, 3, 600, 600, dtype=torch.float32)
        with torch.no_grad():
            dummy_output = self(dummy_input)
            self.out_channels = dummy_output.shape[1]

    def forward(self, inp):
        """Get feature maps.

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW.

        Returns
        -------
        torch.autograd.Variable
            The features extracted from the last conv layer.
        """
        output = inp
        for name, module in self.named_children():
            output = module(output)

        return output


class RPNModel(nn.Module):
    """Region Proposal Network (RPN) as proposed in:
        Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards
        Real-Time Object Detection with Region Proposal Networks.

    This shares the same backbone (`BackboneModel`) with Fast R-CNN model.

    Parameters
    ----------
    backbone_model : BackboneModel
        Initialized backbone model.
    anchor_areas : list
        List of anchor areas (before scaled).
    aspect_ratios : list
        List of anchor aspect ratios (aspect_ratio = bbox_width / bbox_height).
    kernel_size : int
        Size of the n x n conv layer right after the last conv layer of the
        backbone model. (default: 3, as in the paper)
    num_channels : int
        Number of channels of the n x n conv layer and two 1 x 1 sibling conv
        layers. Note that 1 x 1 conv layers are implemented as linear layers.
        (default: 512, as in the paper)
    sampler_name : str
        Positive/negative sampler name. (default: "random_sampler")
    positive_fraction : float
        The fraction of number of postive samples per image batch.
        (default: 0.5)
    batch_size_per_image : int
        Number of boxes (positive + negative) per image. (default: 256)
    reg_lambda : float
        Weight for regression loss, given that the weight for classifcation is
        1.0 (i.e., `final_loss = loss_cls + reg_lambda * loss_t`).
    normalize_offsets : bool
        Whether to normalize box offsets as in the original paper. Note that
        in `torchvision`, offsets are NOT normalized, so this is False by
        default.
    handle_cross_boundary_boxes : dict
        A dictionary with two keys: "during_training" and "during_testing".
        Handle cross-boundary boxes:
            During training, anchor boxes that crosses image boundary will be
            ignored.
            During inference, the predicted boxes will be clipped to the image
            boundary.
        Note that similar to `normalize_offsets`, this setting is mentioned in
        the Faster R-CNN paper, but is ignored in `torchvision`.
        (default: {"during_training": False, "during_testing": True})
    """
    @from_config(main_args="model", requires_all=True)
    def __init__(self, backbone_model, anchor_areas, aspect_ratios,
                 kernel_size=3, num_channels=512,
                 sampler_name="random_sampler",
                 positive_fraction=0.5, batch_size_per_image=256,
                 reg_lambda=1.0, normalize_offsets=False,
                 handle_cross_boundary_boxes={"during_training": False,
                                              "during_testing": True},
                 pre_nms_top_n=2000, post_nms_top_n=100,
                 nms_iou_threshold=0.7, score_threshold=0.1, min_scale=0.01):
        super(RPNModel, self).__init__()
        self.anchor_areas = anchor_areas
        self.aspect_ratios = aspect_ratios
        self.num_anchor_boxes = len(anchor_areas) * len(aspect_ratios)
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.sampler = registry["sampler"][sampler_name](self.config)
        self.reg_lambda = reg_lambda
        self.normalize_offsets = normalize_offsets
        self.handle_cross_boundary_boxes = handle_cross_boundary_boxes

        # Post-processing
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold
        self.min_scale = min_scale

        # Box matcher
        self.box_matcher = Matcher(self.config)

        # Get RoI transformation function
        self.scale_input_to_fm = get_roi_stat(backbone_model.model_type)
        self.scale_fm_to_input = get_orig_stat(backbone_model.model_type)

        # Box offsets' mean and std
        # Source: https://github.com/jwyang/faster-rcnn.pytorch/blob/f9d984d27b48a067b29792932bcb5321a39c1f09/lib/model/utils/config.py#L117
        self.offsets_mean = torch.tensor(
            [0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        self.offsets_std = torch.tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=torch.float32)

        # Additional layers
        self.conv_sliding = nn.Conv2d(
            in_channels=backbone_model.out_channels, out_channels=num_channels,
            kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.box_regression = nn.Conv2d(
            num_channels, self.num_anchor_boxes * 4, kernel_size=1, stride=1)
        self.cls_probs = nn.Conv2d(
            num_channels, self.num_anchor_boxes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

        for layer in [self.conv_sliding, self.box_regression, self.cls_probs]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    """
    Helper functions
    """

    def _normalize_box_offsets(self, boxes):
        """Normmalize box offsets (not box coordinates) using pre-defined mean
        and variance. This is to ensure the regression loss (which depends on
        box offset values) is balanced with the classification loss.

        Refer to the Fast R-CNN paper, multi-task loss for more details.
            Girshick, R. (2015). Fast R-CNN.
        """
        if not self.normalize_offsets:
            return boxes

        is_list = isinstance(boxes, (list, tuple))
        if is_list:
            num_boxes = [len(box) for box in boxes]
            boxes = torch.cat(boxes, dim=0)

        self.offsets_mean = self.offsets_mean.to(boxes.device)
        self.offsets_std = self.offsets_std.to(boxes.device)
        boxes = (boxes - self.offsets_mean) / self.offsets_std

        if is_list:
            return torch.split(boxes, num_boxes, dim=0)
        return boxes

    def _inv_normalize_box_offsets(self, boxes):
        """Same as `_normalize_box_offsets`, but is inversed."""
        if not self.normalize_offsets:
            return boxes

        is_list = isinstance(boxes, (list, tuple))
        if is_list:
            num_boxes = [len(box) for box in boxes]
            boxes = torch.cat(boxes, dim=0)

        self.offsets_mean = self.offsets_mean.to(boxes.device)
        self.offsets_std = self.offsets_std.to(boxes.device)
        boxes = boxes * self.offsets_std + self.offsets_mean

        if is_list:
            return torch.split(boxes, num_boxes, dim=0)
        return boxes

    def _get_anchor_mask(self, anchor_boxes, image_boundaries, training):
        """Get mask with True indicating anchor boxes to ignore and False
        indicating anchor boxes to keep.

        Returns
        -------
        all_masks : list[Tensor]
            A list of masks for all input images, where each mask is of shape
            (H * W * A,) indicating if the i-th anchor should be ignored or
            not.
        """
        # Filter all boxes whose height or width is not postive due to
        # approximation in RoI calculation.
        mask_all = (
            ((anchor_boxes[:, :, 2] - anchor_boxes[:, :, 0]) <= 0)
            | ((anchor_boxes[:, :, 3] - anchor_boxes[:, :, 1]) <= 0)
        )  # (B, H * W * A)

        # Filter all boxes that cross image boundary during training.
        if training and self.handle_cross_boundary_boxes["during_training"]:
            assert len(anchor_boxes) == len(image_boundaries) == len(mask_all)
            mask = (
                (anchor_boxes[:, :, :2] < image_boundaries[:, :, :2]).any(-1)
                | (anchor_boxes[:, :, 2:] > image_boundaries[:, :, 2:]).any(-1)
            )  # (B, H * W * A)
            mask = mask | mask_all  # (B, H * W * A)
        else:
            mask = mask_all

        return mask  # (B, H * W * A)

    @staticmethod
    def _get_batch_mask(anchor_mask):
        return anchor_mask.all(dim=-1)  # (B,)

    def _label_anchors(self, gt_boxes, anchor_boxes):
        """Label anchors as positive or negative or "ignored" given the
        groundtruth boxes

        Parameters
        ----------
        gt_boxes : list[Tensor[float]]
            List of groundtruth boxes for each image in the xyxy format.
        anchor_boxes : list[Tensor[float]]
            List of of anchor boxes for each image over the original input size
            in the xyxy format.

        Returns
        -------
        labels : list[Tensor[float]]
            List of labels for each image, where each tensor is of size (F,)
            denoting `1.0` if the corresponding anchor box is matched
            (positive), `0.0` if it is unmatched (negative), and `-1.0` if it
            is ignored.
        matched_gt_boxes : list[Tensor[float]]
            List of groundtruth matches for each image in the xyxy format.
        """
        labels, matched_gt_boxes = [], []

        for gt_boxes_i, anchor_boxes_i in zip(gt_boxes, anchor_boxes):
            iou = box_iou(gt_boxes_i, anchor_boxes_i)  # (x_i, F)
            matched_idxs = self.box_matcher(iou)  # (F,)
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes_i[
                matched_idxs.clamp(min=0)]  # (F,)

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = (
                matched_idxs == self.box_matcher.BELOW_LOW_THRESHOLD)
            labels_per_image[bg_indices] = 0.0

            # discard indices that are between thresholds
            inds_to_discard = (
                matched_idxs == self.box_matcher.BETWEEN_THRESHOLDS)
            labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)

        return labels, matched_gt_boxes

    def _clip_boxes_to_image_boundary(self, boxes, image_boundaries, mode):
        """Clip boxes to image boundary.

        Parameters
        ----------
        boxes : torch.Tensor
            Tensor of shape (B, ..., 4) or list of size `batch_size`, where
            each element is a tensor of shape (..., 4), where `...` could be
            any of number of dimensions.
        image_boundaries : torch.Tensor
            Tensor of shape (B, 4) representing the original image boundaries
            as bounding boxes in the transformed batch.
        mode : str
            Either "xyxy" or "xywh", denote the how the boxes are represented.
        """
        if not self.handle_cross_boundary_boxes["during_testing"]:
            return boxes

        if mode not in ["xyxy", "xywh"]:
            raise ValueError("Invalid mode value. Expected one of "
                             "['xyxy', 'xywh'], got {} instead".format(mode))

        is_list = isinstance(boxes, (list, tuple))
        if mode == "xywh":
            # Convert to xyxy
            boxes = convert_xywh_to_xyxy(boxes)
            image_boundaries = convert_xywh_to_xyxy(image_boundaries)
        assert len(boxes) == len(image_boundaries)
        # Clip
        new_boxes = []
        for boxes_per_image, image_boundary in zip(boxes, image_boundaries):
            num_dims = boxes_per_image.ndim
            xmin, ymin, xmax, ymax = torch.split(boxes_per_image, 1, dim=-1)
            img_xmin, img_ymin, img_xmax, img_ymax = image_boundary

            xmin = torch.clamp(xmin, min=img_xmin, max=img_xmax)
            ymin = torch.clamp(ymin, min=img_ymin, max=img_ymax)
            xmax = torch.clamp(xmax, min=img_xmin, max=img_xmax)
            ymax = torch.clamp(ymax, min=img_ymin, max=img_ymax)

            # Convert back
            new_boxes_per_image = torch.cat(
                [xmin, ymin, xmax, ymax], dim=num_dims - 1)
            if mode == "xywh":
                new_boxes_per_image = convert_xyxy_to_xywh(new_boxes_per_image)

            new_boxes.append(new_boxes_per_image)

        if not is_list:
            new_boxes = torch.stack(new_boxes, dim=0)
        return new_boxes

    def _remove_unsatisfactory_boxes(self, boxes, cls, image_boundaries):
        """Remove unsatisfactory boxes, including small boxes and low scoring
        boxes.

        Parameters
        ----------
        boxes : torch.Tensor
            Tensor of shape (B, ..., 4) or list of size `batch_size`, where
            each element is a tensor of shape (..., 4),, where `...` could be
            any of number of dimensions. xyxy formatted.
        cls : torch.Tensor
            Tensor of shape (B, ...) denoting the predicted scores for each
            box.
        image_boundaries : torch.Tensor
            Tensor of shape (B, 4) representing the original image boundaries
            as bounding boxes in the transformed batch.

        Returns
        -------
        mask : torch.Tensor
            A mask indicating whether each box should be filtered out or not.
            Shaped (B, ...), where `...` corresponds to that of `boxes`.
        """
        assert len(boxes) == len(image_boundaries)
        is_list = isinstance(boxes, (list, tuple))
        if not is_list:
            boxes = [boxes]
            cls = [cls]
            image_boundaries = [image_boundaries]

        new_boxes, new_cls = [], []
        for boxes_i, cls_i, image_boundaries_i in \
                zip(boxes, cls, image_boundaries):
            box_areas_i = box_area(boxes_i, mode="xyxy")  # (...)
            image_areas_i = box_area(image_boundaries_i, mode="xyxy")  # scalar

            # Need to resize during inference, where we are processing a BATCH
            # of image (i.e., is_list == False and
            # `boxes_i.shape[0] == batch_size`)
            if image_areas_i.ndim > 0:
                num_dim_to_add = box_areas_i.ndim - 1
                new_view = [image_areas_i.shape[0]] + [1] * num_dim_to_add
                image_areas_i = image_areas_i.view(*new_view).expand_as(
                    box_areas_i)

            mask_small_i = (
                (box_areas_i / image_areas_i) < self.min_scale)  # (...)
            mask_low_score_i = cls_i < self.score_threshold  # (...)
            mask = mask_small_i | mask_low_score_i

            if is_list:
                boxes_i = boxes_i[mask]
                cls_i = cls_i[mask]
            else:
                boxes_i = apply_mask(boxes_i, mask)
                cls_i = apply_mask(cls_i, mask)

            new_boxes.append(boxes_i)
            new_cls.append(cls_i)

        if not is_list:
            new_boxes = new_boxes[0]
            new_cls = new_cls[0]

        return new_boxes, new_cls

    """
    Main functions, which combine logics and multiple helper functions
    """

    def _forward(self, inp):
        """Feed forward"""
        # Get feature map
        feature_map = self.relu(self.conv_sliding(inp))  # (B, C, H, W)
        batch_size, _, fm_height, fm_width = feature_map.shape

        # `A` denotes the number of anchor boxes at each receptive point.
        pred_cls = self.cls_probs(feature_map)  # (B, A, H, W)
        pred_cls = self.sigmoid(pred_cls).permute(0, 2, 3, 1).reshape(
            batch_size, -1)  # (B, H * W * A)
        pred_t = self.box_regression(feature_map)  # (B, A * 4, H, W)
        pred_t = pred_t.permute(0, 2, 3, 1).reshape(
            batch_size, -1, 4)  # (B, H * W * A, 4)

        return feature_map, pred_cls, pred_t

    def _get_anchor_boxes(self, fm_width, fm_height, batch_size, device):
        # Get anchor boxes for this feature map
        anchor_boxes = get_anchor_boxes(
            (fm_width, fm_height),
            self.anchor_areas, self.aspect_ratios
        ).to(device)  # (H * W * A, 4)
        # Scale the anchor x, y coordinates back to the original image sizes
        anchor_boxes[:, :2] = self.scale_fm_to_input(anchor_boxes[:, :2])
        # Convert from xywh to xyxy
        anchor_boxes = convert_xywh_to_xyxy(anchor_boxes)
        anchor_boxes = anchor_boxes.repeat(
            batch_size, 1, 1)  # (B, H * W * A, 4)

        return anchor_boxes  # (B, H * W * A, 4)

    def _nms_pre_process(self, gt_boxes, image_boundaries, anchor_boxes,
                         pred_cls, pred_t):
        """This applies pre-processing steps to boxes before NMS is applied.
        During training:
            This function handles all anchor boxes that cross image boundaries
            by either removing them all or clip to image boundaries, depending
            on whether it is training or testing, and on the settings.
            This function also filters out all images whose respective anchor
            boxes are **all** filtered out in the previous step.

        During inference
            This function limit the number of anchor boxes per image.
        """
        if gt_boxes is not None:  # during training
            # Get a mask to filter out-of-image (cross_boundary) anchor boxes
            valid_anchor_mask = self._get_anchor_mask(
                anchor_boxes, image_boundaries, training=True
            )  # (B, H * W * A)
            # Get a mask to filter images whose anchor boxes are all invalid
            batch_mask = self._get_batch_mask(valid_anchor_mask)  # (B,)
            # Filter images whose anchor boxes are all invalid
            anchor_boxes, pred_t, pred_cls, valid_anchor_mask = [
                tensor[~batch_mask] for tensor in
                [anchor_boxes, pred_t, pred_cls, valid_anchor_mask]
            ]
            gt_boxes = [
                gt_box for gt_box, m in zip(gt_boxes, batch_mask) if not m]
            # Filter out-of-image anchor boxes; each of the resulting objects
            # is a list of size `batch_size'`, where `batch_size'` <=
            # `batch_size`, and where each i-th element if either of shape
            # (A_i,) or (A_i, 4) where A_i is the number of valid anchor boxes.
            anchor_boxes, pred_t, pred_cls = [
                apply_mask(tensor, valid_anchor_mask) for tensor in
                [anchor_boxes, pred_t, pred_cls]
            ]
        else:  # during inference
            # Get top_n before NMS
            _, idxs = torch.topk(
                pred_cls, k=min(self.pre_nms_top_n, pred_cls.shape[-1]),
                dim=-1, largest=True, sorted=True)  # (B, N_pre)
            anchor_boxes = index_argsort(
                anchor_boxes, idxs, dim=1)  # (B, N_pre, 4)
            pred_t = index_argsort(pred_t, idxs, dim=1)  # (B, N_pre, 4)
            pred_cls = index_argsort(pred_cls, idxs, dim=1)  # (B, N_pre)

            # Un-normalize predicted offsets
            pred_t = self._inv_normalize_box_offsets(
                pred_t)  # (B, N_pre, 4)

        return gt_boxes, anchor_boxes, pred_cls, pred_t

    def _get_predicted_boxes(self, image_boundaries, anchor_boxes, pred_cls,
                             pred_t):
        """
        Get predicted boxes from anchor boxes and predicted offsets.
        The tensor shapes annotated in below also apply to list of tensors
        (during training, where each image might have different number of
        anchor boxes).
        """
        # Convert offsets to coordinates
        pred_boxes = convert_offsets_to_coords(
            pred_t, convert_xyxy_to_xywh(anchor_boxes)
        )  # (B, N_pre, 4)
        pred_boxes = convert_xywh_to_xyxy(pred_boxes)
        # Clip to feature map boundary
        pred_boxes = self._clip_boxes_to_image_boundary(
            pred_boxes, image_boundaries, mode="xyxy")  # (B, N_pre, 4)

        # Remove small or low scoring boxes; pred_boxes and pred_cls:
        # lists of size `batch_size`, where each element corresponds
        # to each image in the batch.
        pred_boxes, pred_cls = self._remove_unsatisfactory_boxes(
            pred_boxes, pred_cls, image_boundaries)

        return pred_boxes, pred_cls

    def _compute_losses(self, gt_boxes, anchor_boxes, pred_cls, pred_t):
        """Compute loss given the filtered groundtruth boxes (`gt_boxes`),
        anchor boxes (`anchor_boxes`), predicted probabilities (`pred_cls`)
        and predicted box offsets (`pred_t`).

        The behavior is as follow:
            If `gt_boxes` is None, meaning it is inference step, immediately
            end this step and return an empty dict since we don't need loss
            during inference.
            Otherwise
                1. Each anchor will be assigned to either one or zero
                groundtruth box.
                2. Compute (and normalize) groundtruth box offsets.
                3. Sample positive/negative examples.
                4. Compute losses.
        """
        if gt_boxes is None:
            return {}

        # Map each groundtruth with its corresponding anchor box(es)
        # labels: list of size `batch_size`, where each element is of shape
        # (A_i,)
        # matched_gt_boxes: list of size `batch_size`, where each element
        # is of shape (A_i, 4)
        labels, matched_gt_boxes = self._label_anchors(
            gt_boxes, anchor_boxes)

        # Calculate gt boxes' offsets and normalize
        # gt_t: list of size `batch_size`, where each element is of shape
        # (A_i, 4)
        gt_t = convert_coords_to_offsets(
            batching(convert_xyxy_to_xywh, matched_gt_boxes),
            batching(convert_xyxy_to_xywh, anchor_boxes))
        gt_t = self._normalize_box_offsets(gt_t)

        # sampled_pos_inds and sampled_neg_inds: list of size `batch_size`,
        # where each element is of shape (A_i,)
        sampled_pos_inds, sampled_neg_inds = self.sampler(
            anchor_labels=labels, pred_objectness=pred_cls)

        # sampled_pos_inds: (S_pos,); sampled_neg_inds: (S_neg,), where
        # S_pos, S_neg are the total number positive and negative examples
        # in the entire batch, respectively.
        sampled_pos_inds = torch.where(
            torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(
            torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat(
            [sampled_pos_inds, sampled_neg_inds],
            dim=0)  # (S_pos + S_neg,)

        labels = torch.cat(labels, dim=0)  # (Σ A_i,)
        gt_t = torch.cat(gt_t, dim=0)  # (Σ A_i, 4)
        pred_cls = torch.cat(pred_cls, dim=0)  # (Σ A_i,)
        pred_t = torch.cat(pred_t, dim=0)  # (Σ A_i, 4)

        loss_t = smooth_l1_loss(
            pred_t[sampled_pos_inds],
            gt_t[sampled_pos_inds],
            beta=1 / 9,
            size_average=False
        ) / (sampled_inds.numel())
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_cls[sampled_inds], labels[sampled_inds])
        loss = loss_cls + self.reg_lambda * loss_t

        return {"loss_t": loss_t, "loss_cls": loss_cls, "loss": loss}

    def _nms_post_process(self, pred_boxes, pred_cls):
        """Limit the number of output predictions per image"""
        new_pred_boxes = []
        new_pred_cls = []
        for pred_boxes_per_image, pred_cls_per_image in \
                zip(pred_boxes, pred_cls):
            k = min(self.post_nms_top_n, pred_cls_per_image.shape[-1])
            new_pred_cls_per_image, idxs = torch.topk(
                pred_cls_per_image, k=k, dim=-1,
                largest=True, sorted=True)
            new_pred_boxes_per_image = pred_boxes_per_image[idxs]

            new_pred_boxes.append(new_pred_boxes_per_image)
            new_pred_cls.append(new_pred_cls_per_image)

        return new_pred_boxes, new_pred_cls

    """
    Master function.
    """

    def forward(self, inp, gt_boxes=None, labels=None, image_boundaries=None):
        """
        Parameters
        ----------
        inp : torch.Tensor
            Mini-batch of feature maps of shape (B, C, H, W) produced by the
            backbone model.
        gt_boxes : list of torch.Tensor
            If specified, it should be list of size `batch_size`, where i-th
                element of shape (x_i, 4) represents the bounding boxes'
                coordinates (xmin, ymin, xmax, ymax) of the i-th image, and x_i
                is the number of boxes of the i-th image.
            If not specified, the function will return a list of proposal boxes
            and its scores, as is during inference.
        labels
            Not important. It is specified just to ensure a consistent
            arguments across different models.
        image_boundaries : torch.Tensor
            Tensor of shape (batch_size, 4) representing original image
            boundaries in the transformed images as a bounding box. This is
            used to filter out cross-boundary anchors during training, and to
            clip predicted boxes during inference. If not specified, this code
            assumes the original image boundaries to be the transformed
            images'.
        """
        output = {}

        """Forward step"""
        feature_map, pred_cls, pred_t = self._forward(inp)
        batch_size, _, fm_height, fm_width = feature_map.shape

        anchor_boxes = self._get_anchor_boxes(
            fm_width, fm_height, batch_size, inp.device)  # (B, H * W * A, 4)

        # This will be used if `handle_cross_boundary_boxes` is True
        if image_boundaries is None:
            image_boundaries = torch.tensor(
                [0, 0, inp.shape[3], inp.shape[2]]
            ).repeat(inp.shape[0], 1).to(anchor_boxes)  # (B, 4)

        """Pre-process before NMS.
        Training:
            Each of the resulting objects is a list of size `batch_size'`,
            where `batch_size'` <= `batch_size`, and where each i-th
            element if either of shape (A_i,) or (A_i, 4) where A_i is the
            number of valid anchor boxes.
        Inference:
            Each of the resulting objects is a Tensor of shape (B, N_pre)
            or (B, N_pre, 4).
        """
        out = self._nms_pre_process(
            gt_boxes, image_boundaries, anchor_boxes, pred_cls, pred_t)
        gt_boxes, anchor_boxes, pred_cls, pred_t = out

        """Compute loss; this does nothing during inference"""
        losses = self._compute_losses(
            gt_boxes, anchor_boxes, pred_cls, pred_t)
        output.update(losses)

        """Get predicted boxes"""
        with torch.no_grad():
            """Get raw predicted boxes"""
            pred_boxes, pred_cls = self._get_predicted_boxes(
                image_boundaries, anchor_boxes, pred_cls, pred_t)

            """Perform NMS"""
            # keep_idxs: list of size `batch_size`, where each element is a
            # tensor of indices to keep for each image.
            keep_idxs = batched_nms(
                pred_boxes, pred_cls, self.nms_iou_threshold)
            # Index NMS results; pred_boxes and pred_cls are both list of
            # size `batch_size`, where each element corresponds to each image.
            pred_boxes = index_batch(pred_boxes, keep_idxs)
            pred_cls = index_batch(pred_cls, keep_idxs)

            """
            Post-process, which limits the number of output predictions per
            image.
            """
            pred_boxes, pred_cls = self._nms_post_process(pred_boxes, pred_cls)
            out = {"pred_boxes": pred_boxes, "pred_probs": pred_cls}
            output.update(out)

        return output


class FasterRCNN(nn.Module):
    def __init__(self, backbone, rpn_head):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head

    def forward(self, inp, **kwargs):
        feature_map = self.backbone(inp)
        return self.rpn_head(feature_map, **kwargs)
