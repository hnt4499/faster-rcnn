import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import box_iou

from .utils import (get_anchor_boxes, smooth_l1_loss, convert_xywh_to_xyxy,
                    clip_boxes_to_image_boundary, index_argsort,
                    convert_coords_to_offsets, convert_offsets_to_coords,
                    get_sampler, Matcher)


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
        # dummy_input = torch.zeros(1, 3, 600, 600, dtype=torch.float32)
        # with torch.no_grad():
        #     dummy_output = self(dummy_input)
        #     self.out_channels = dummy_output.shape[1]
        self.out_channels = 2048

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
    sampler : str
        Positive/negative sampler name. (default: "random_sampler")
    positive_fraction : float
        The fraction of number of postive samples per image batch.
        (default: 0.5)
    batch_size_per_image : int
        Number of boxes (positive + negative) per image. (default: 256)
    reg_lambda : float
        Weight for regression loss, given that the weight for classifcation is
        1.0 (i.e., `final_loss = loss_cls + reg_lambda * loss_t`).
    """
    def __init__(self, backbone_model, anchor_areas, aspect_ratios,
                 kernel_size=3, num_channels=512, sampler="random_sampler",
                 positive_fraction=0.5, batch_size_per_image=256,
                 reg_lambda=1.0):
        super(RPNModel, self).__init__()
        self.backbone_model = backbone_model
        self.anchor_areas = anchor_areas
        self.aspect_ratios = aspect_ratios
        self.num_anchor_boxes = len(anchor_areas) * len(aspect_ratios)
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.sampler = get_sampler(sampler)(
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction)
        self.reg_lambda = reg_lambda

        # Box matcher
        self.box_matcher = Matcher(
            high_threshold=0.7, low_threshold=0.3,
            allow_low_quality_matches=True)

        # Get RoI transformation function
        self.scale_input_to_fm = get_roi_stat(backbone_model.model_type)
        self.scale_fm_to_input = get_orig_stat(backbone_model.model_type)

        # Box offsets' mean and std
        # Source: https://github.com/jwyang/faster-rcnn.pytorch/blob/f9d984d27b48a067b29792932bcb5321a39c1f09/lib/model/utils/config.py#L117
        # self.offsets_mean = torch.tensor(
        #     [0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        # self.offsets_std = torch.tensor(
        #     [0.1, 0.1, 0.2, 0.2], dtype=torch.float32)
        self.offsets_mean = torch.tensor(
            [0.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=False)
        self.offsets_std = torch.tensor(
            [1, 1, 1, 1], dtype=torch.float32, requires_grad=False)

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

    def _normalize_box_offsets(self, boxes):
        """Normmalize box offsets (not box coordinates) using pre-defined mean
        and variance. This is to ensure the regression loss (which depends on
        box offset values) is balanced with the classification loss.

        Refer to the Fast R-CNN paper, multi-task loss for more details.
            Girshick, R. (2015). Fast R-CNN.
        """
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

    def forward(self, inp, gt_boxes=None, labels=None):
        """
        Parameters
        ----------
        inp : torch.Tensor
            Mini-batch of images.
        gt_boxes : list of torch.Tensor
            If specified, it should be list of size `batch_size`, where i-th
                element of shape (x_i, 4) represents the bounding boxes'
                coordinates (x_center, y_center, width, height) of the i-th
                image, and x_i is the total number of boxes of the i-th image.
            If not specified, the function will return a list of proposal boxes
            and its scores.
        """
        training = (gt_boxes is not None)

        # Get feature map
        feature_map = self.backbone_model(inp)  # (B, C, H, W)
        feature_map = self.relu(self.conv_sliding(feature_map))  # (B, C, H, W)

        batch_size, _, fm_height, fm_width = feature_map.shape

        # Feed forward; `A` denotes the number of anchor boxes at each
        # receptive point.
        preds_t = self.box_regression(feature_map)  # (B, A * 4, H, W)
        preds_t = preds_t.permute(0, 2, 3, 1).reshape(
            batch_size, -1, 4)  # (B, H * W * A, 4)
        preds_cls = self.cls_probs(feature_map)  # (B, A, H, W)
        preds_cls = self.sigmoid(preds_cls).permute(0, 2, 3, 1).reshape(
            batch_size, -1)  # (B, H * W * A)

        # Get anchor boxes for this feature map
        anchor_boxes = get_anchor_boxes(
            (fm_width, fm_height),
            self.anchor_areas, self.aspect_ratios
        ).to(inp.device)  # (H * W * A, 4)
        # Scale the anchor x, y coordinates back to the original image sizes
        anchor_boxes[:, 0] = self.scale_fm_to_input(anchor_boxes[:, 0])
        anchor_boxes[:, 1] = self.scale_fm_to_input(anchor_boxes[:, 1])

        if training:
            # Map each groundtruth with its corresponding anchor box(es)
            # labels: list of size `batch_size`, where each element is of shape
            # (H * W * A,)
            # matched_gt_boxes: list of size `batch_size`, where each element
            # is of shape (H * W * A, 4)
            labels, matched_gt_boxes = self._label_anchors(
                gt_boxes, anchor_boxes)

            # Calculate gt boxes' offsets and normalize
            # gt_t: list of size `batch_size`, where each element is of shape
            # (H * W * A, 4)
            gt_t = convert_coords_to_offsets(
                matched_gt_boxes, anchor_boxes)
            gt_t = self._normalize_box_offsets(gt_t)

            # sampled_pos_inds and sampled_neg_inds: list of size `batch_size`,
            # where each element is of shape (H * W * A,)
            sampled_pos_inds, sampled_neg_inds = self.sampler(labels)

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

            preds_cls = preds_cls.flatten()  # (B * H * W * A,)

            labels = torch.cat(labels, dim=0)  # (B * H * W * A,)
            gt_t = torch.cat(gt_t, dim=0)  # (B * H * W * A, 4)
            preds_t = preds_t.reshape(-1, 4)  # (B * H * W * A, 4)

            loss_regression = smooth_l1_loss(
                preds_t[sampled_pos_inds],
                gt_t[sampled_pos_inds],
                beta=1 / 9,
                size_average=False,
            ) / (sampled_inds.numel())

            loss_cls = F.binary_cross_entropy_with_logits(
                preds_cls[sampled_inds], labels[sampled_inds]
            )

            loss = loss_cls + self.reg_lambda * loss_regression

            output = {
                "loss_t": loss_regression, "loss_cls": loss_cls, "loss": loss
            }

            return output

        else:
            anchor_boxes = anchor_boxes.repeat(batch_size, 1, 1)  # (B, F, 4)

            # Sort predictions by their scores
            idxs = torch.argsort(preds_cls, dim=1, descending=True)  # (B, F)
            anchor_boxes = index_argsort(
                anchor_boxes, idxs, dim=1)  # (B, F, 4)
            preds_t = index_argsort(preds_t, idxs, dim=1)  # (B, F, 4)
            preds_cls = index_argsort(preds_cls, idxs, dim=1)  # (B, F)

            # Un-normalize predicted offsets
            preds_t = self._inv_normalize_box_offsets(preds_t)

            # Convert offsets to coordinates
            pred_boxes = convert_offsets_to_coords(
                preds_t, anchor_boxes)  # (B, F, 4)
            # Clip to feature map boundary
            pred_boxes = clip_boxes_to_image_boundary(
                pred_boxes, fm_width, fm_height, mode="xywh")
            # Scale back to original input size
            pred_boxes = self.scale_fm_to_input(pred_boxes)

            output = {
                "pred_boxes": pred_boxes, "pred_probs": preds_cls
            }

        return output

    def _get_anchor_mask(self, anchor_boxes, feature_map_size, training):
        """Get mask with True indicating anchor boxes to ignore and False
        indicating anchor boxes to keep."""
        # Filter all boxes whose height or width is not postive due to
        # approximation in RoI calculation.
        mask = ((anchor_boxes[:, 2] <= 0)
                | (anchor_boxes[:, 3] <= 0))

        # Filter all boxes that cross image boundary during training.
        if training:
            feature_map_width, feature_map_height = feature_map_size
            anchor_boxes_xyxy = convert_xywh_to_xyxy(anchor_boxes)
            mask = (mask
                    | (anchor_boxes_xyxy[:, 0] < 0)
                    | (anchor_boxes_xyxy[:, 1] < 0)
                    | (anchor_boxes_xyxy[:, 2] > feature_map_width)
                    | (anchor_boxes_xyxy[:, 3] > feature_map_height))

        return mask

    @staticmethod
    def _reindex(x, idxs):
        """Reindex tensor along the batch axis and return the concatenated
        output."""
        outp = []
        for x_i, idxs_i in zip(x, idxs):
            outp.append(x_i[idxs_i])
        outp = torch.cat(outp, dim=0)
        return outp

    def _label_anchors(self, gt_boxes, anchor_boxes):
        """Label anchors as positive or negative or "ignored" given the
        groundtruth boxes

        Parameters
        ----------
        gt_boxes : list[Tensor[float]]
            List of groundtruth boxes for each image in the xywh format.
        anchor_boxes : Tensor[float]
            Tensor of anchor boxes over the original input size in the xywh
            format.

        Returns
        -------
        labels : list[Tensor[float]]
            List of labels for each image, where each tensor is of size (F,)
            denoting `1.0` if the corresponding anchor box is matched
            (positive), `0.0` if it is unmatched (negative), and `-1.0` if it
            is ignored.
        matched_gt_boxes : list[Tensor[float]]
            List of groundtruth matches for each image in the xywh format.
        """
        labels, matched_gt_boxes = [], []

        # Convert (x_center, y_center, width, height) to
        # (xmin, ymin, xmax, ymax)
        anchor_boxes_xyxy = convert_xywh_to_xyxy(anchor_boxes)

        for gt_boxes_i in gt_boxes:
            gt_boxes_i_xyxy = convert_xywh_to_xyxy(gt_boxes_i)
            iou = box_iou(gt_boxes_i_xyxy, anchor_boxes_xyxy)  # (x_i, F)
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
