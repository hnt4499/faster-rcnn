import os
import math
import inspect

import torch
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from loguru import logger

from .utils import index_argsort
from .registry import register, registry


class BaseMetric:
    """Base class for all metrics. Note that all metrics expect boxes in xyxy
    format."""
    def __init__(self):
        self.last_value = None

    def call(self, gt_boxes, gt_labels, pred_boxes, pred_objectness,
             pred_classes, iou_matrices=None):
        """Calculate metric. All inherited classes must follow this parameter
        call.

        Parameters
        ----------
        gt_boxes : list[Tensor]
            List of groundtruth boxes of size `batch_size`. Each element is a
            tensor of shape (M, 4).
        gt_labels : list[Tensor]
            List of groundtruth labels of size `batch_size`. Each element is a
            tensor of shape (M,).
        pred_boxes : list[Tensor]
            List of predictions of size `batch_size`. Each element is a tensor
            of shape (N, 4).
        pred_objectness : list[Tensor]
            Objectness scores (e.g., output of RPN model) corresponding to
            each prediction in `pred_boxes` tensor. List of size `batch_size`,
            where each element is a tensor of shape (N,).
        pred_classes : list[Tensor]
            Predicted classes (e.g., output of Fast R-CNN model) corresponding
            to each prediction in `pred_boxes` tensor. List of size
            `batch_size`, where each element is a tensor of shape (N,).
        iou_matrices : list[Tensor] or None
            List of NxM matrices of precomputed IoU value.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        value = self.call(*args, **kwargs)
        self.last_value = value
        return value

    def get_str(self):
        """Used to log results"""
        raise NotImplementedError

    def write_to_tensorboard(self, writer):
        writer.add_scalar(
            self._metric_name, self.last_value,
            display_name=self._metric_description)


@register("metric")
class BoxRecall(BaseMetric):
    _metric_name = "box_recall"
    _metric_description = "Box Recall"
    """Calculate box recall for class-agnostic task.

    Parameters
    ----------
    iou_threshold : float
        Groundtruths with IoU overlap with any of predictions higher than
        this threshold will be considered detected.
    """
    def __init__(self, iou_threshold=0.5, config=None):
        super(BoxRecall, self).__init__()
        self.iou_threshold = iou_threshold

    def call(self, gt_boxes, pred_boxes, iou_matrices=None):
        """Calculate box recall for class-agnostic task."""
        assert len(gt_boxes) == len(pred_boxes)
        if iou_matrices is None:
            iou_matrices = [
                box_iou(gt_boxes_i, pred_boxes_i) for
                gt_boxes_i, pred_boxes_i in zip(gt_boxes, pred_boxes)
            ]
            return self.call(gt_boxes, pred_boxes, iou_matrices=iou_matrices)

        tot_boxes = 0
        tot_detected_boxes = 0

        for iou_matrix in iou_matrices:
            # Groundtruth boxes with no "correct" prediction boxes are
            # considered not detected
            is_detected = (iou_matrix >= self.iou_threshold).any(dim=-1)
            tot_boxes += is_detected.numel()
            tot_detected_boxes += is_detected.sum()

        return float(tot_detected_boxes) / float(tot_boxes)

    def get_str(self):
        return f"recall: {self.last_value:.4f}"


@register("metric")
class MeanAverageBestOverlap(BaseMetric):
    _metric_name = "mABO"
    _metric_description = "Mean Average Best Overlap (mABO)"
    """Calculate mean average best overlap (MABO).

    Reference:
        Uijlings, J. R. R., van de Sande, K. E. A., Gevers, T., & Smeulders,
        A. W. M. (2013). Selective Search for Object Recognition. International
        Journal of Computer Vision, 104(2), 154–171.
    """
    def __init__(self, config=None):
        super(MeanAverageBestOverlap, self).__init__()

    def call(self, gt_boxes, gt_labels, pred_boxes, iou_matrices=None):
        """Calculate box recall for class-agnostic task."""
        assert len(gt_boxes) == len(pred_boxes)
        if iou_matrices is None:
            iou_matrices = [
                box_iou(gt_boxes_i, pred_boxes_i) for
                gt_boxes_i, pred_boxes_i in zip(gt_boxes, pred_boxes)
            ]
            return self.call(gt_boxes, gt_labels, pred_boxes,
                             iou_matrices=iou_matrices)

        best_overlaps = []
        for iou_matrix, gt_labels_i in zip(iou_matrices, gt_labels):
            # In case there is no predicted box
            if iou_matrix.numel() == 0:
                best_overlap = torch.zeros((gt_labels_i.shape[0],)).to(
                    iou_matrix)
            else:
                best_overlap, _ = iou_matrix.max(dim=-1)
            best_overlaps.append(best_overlap)

        best_overlaps = torch.cat(best_overlaps)
        gt_labels = torch.cat(gt_labels).to(torch.int16)
        assert len(best_overlaps) == len(gt_labels)

        # Sort
        gt_labels, sort_idxs = torch.sort(gt_labels)
        best_overlaps = best_overlaps[sort_idxs]

        # Since tensors are sorted, we can use `unique_consecutive` here
        _, counts = torch.unique_consecutive(gt_labels, return_counts=True)
        start = 0
        mabo = []
        for c in counts:
            end = start + c
            abo = best_overlaps[start:end].mean()
            mabo.append(abo)
            start = end

        return sum(mabo) / len(mabo)

    def get_str(self):
        return f"mABO: {self.last_value:.4f}"


@register("metric")
class DRWinCurve(BaseMetric):
    _metric_name = "auc"
    _metric_description = "Area Under the DR-#WIN Curve"
    """Plot the DR-#WIN curve (detection-rate (recall) versus the number of
    windows per image) and calculate area under the DR-#WIN curve approximated
    in a log space (to avoid bias towards too many predicted boxes). Note that
    `max(recalls)` might be slightly different from recall computed from
    `BoxRecall` due to an approximation in `BoxRecall` implementation.

    Reference:
        Alexe, B., Deselaers, T., & Ferrari, V. (2012). Measuring the
        Objectness of Image Windows. IEEE Transactions on Pattern Analysis and
        Machine Intelligence, 34(11), 2189–2202.
    """
    def __init__(self, iou_threshold=0.5, config=None):
        super(DRWinCurve, self).__init__()
        self.config = config
        self.iou_threshold = iou_threshold

    def call(self, gt_boxes, pred_boxes, pred_objectness, iou_matrices=None):
        assert len(gt_boxes) == len(pred_boxes) == len(pred_objectness)
        if iou_matrices is None:
            iou_matrices = [
                box_iou(gt_boxes_i, pred_boxes_i) for
                gt_boxes_i, pred_boxes_i in zip(gt_boxes, pred_boxes)
            ]
            return self.call(gt_boxes, pred_boxes, pred_objectness,
                             iou_matrices=iou_matrices)

        matches = []
        for iou_matrix, gt_boxes_i, pred_boxes_i, pred_objectness_i in \
                zip(iou_matrices, gt_boxes, pred_boxes, pred_objectness):
            # Sort by objectness score
            pred_objectness_i, sort_idxs = pred_objectness_i.sort(
                descending=True)
            pred_boxes_i = pred_boxes_i[sort_idxs]
            iou_matrix = index_argsort(iou_matrix, sort_idxs, dim=-1)

            # Get matches for each predicted boxes (negative = no match)
            if iou_matrix.numel() == 0:
                matches.append([])  # we just need an empty tensor
            else:
                matches_val_i, matches_i = iou_matrix.max(dim=0)
                matches_i[matches_val_i < self.iou_threshold] = -1

                # For a groundtruth box that corresponds to moltiple predicted
                # boxes, keep only the highest scoring one.
                unique, counts = torch.unique(matches_i, return_counts=True)
                mask = unique >= 0  # don't filter false positives
                unique, counts = unique[mask], counts[mask]

                mask_multiple = counts > 1
                unique_multiple = unique[mask_multiple]

                for unique_i in unique_multiple:
                    where = torch.where(matches_i == unique_i)[0]
                    matches_i[where[1:]] = -1  # turn other boxes into FP
                matches.append(matches_i)

        # Now calculate recall over different number of windows per image
        num_win = [len(pred_boxes_i) for pred_boxes_i in pred_boxes]
        max_win = max(num_win)  # maximum number of windows per image
        spaces = torch.exp(  # log space to avoid bias
            torch.linspace(start=0, end=math.log(max_win), steps=30))
        spaces = torch.unique_consecutive(spaces.round().long())  # avoid dups

        tot_boxes = sum(len(gt_boxes_i) for gt_boxes_i in gt_boxes)
        tot_detected_boxes = 0
        recalls = []

        for i, space in enumerate(spaces):
            start = 0 if i == 0 else spaces[i - 1]
            end = space

            for matches_i in matches:
                if start < len(matches_i):
                    tot_detected_boxes += (matches_i[start:end] >= 0).sum()
            recalls.append(float(tot_detected_boxes) / float(tot_boxes))

        # Since we have an increment of window of 1, area under the curve is
        # the mean of all recall values
        auc = sum(recalls) / len(recalls)

        # Plot
        self._plot(spaces, recalls, label=f"auc={auc:.4f}")

        return {"auc": auc, "num_windows": spaces, "recall_scores": recalls}

    def __call__(self, *args, **kwargs):
        outp = self.call(*args, **kwargs)
        self.last_value = outp["auc"]
        return outp

    def _plot(self, x, y, label):
        # Save path
        save_dir = self.config["training"]["save_dir"]
        if save_dir is None or "epoch" not in self.config:
            return

        save_path = os.path.join(
            save_dir, f"DRWinCurve_{self.config['epoch']}.jpg")

        plt.plot(x, y, label=label)
        plt.xscale('log')
        plt.xlabel("# windows (log)")
        plt.ylabel("recall")
        plt.title("DR-#WIN curve")
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Figure saved to {save_path}.")

    def get_str(self):
        return f"area(DR-#WIN): {self.last_value:.4f}"


class MetricHolder:
    """Metric holder"""
    def __init__(self, metrics_config, all_config):
        self.metrics_config = metrics_config
        self.all_config = all_config
        self.last_result_str = None

        self.metrics = {}
        for metric_name, metric_config in metrics_config.items():
            metric_initialized = registry["metric"][metric_name](
                config=all_config, **metric_config)
            self.metrics[metric_name] = metric_initialized

    def __call__(self, gt_boxes, gt_labels, pred_boxes, pred_objectness,
                 pred_classes):
        """Calculate metric.

        Parameters
        ----------
        gt_boxes : list[Tensor]
            List of groundtruth boxes of size `batch_size`. Each element is a
            tensor of shape (M, 4).
        gt_labels : list[Tensor]
            List of groundtruth labels of size `batch_size`. Each element is a
            tensor of shape (M,).
        pred_boxes : list[Tensor]
            List of predictions of size `batch_size`. Each element is a tensor
            of shape (N, 4).
        pred_objectness : list[Tensor]
            Objectness scores (e.g., output of RPN model) corresponding to
            each prediction in `pred_boxes` tensor. List of size `batch_size`,
            where each element is a tensor of shape (N,).
        pred_classes : list[Tensor]
            Predicted classes (e.g., output of Fast R-CNN model) corresponding
            to each prediction in `pred_boxes` tensor. List of size
            `batch_size`, where each element is a tensor of shape (N,).

        Returns
        -------
        results
            A dictionary mapping each metric name (e.g., "BoxRecall") with its
            computed value.
        """
        iou_matrices = [
            box_iou(groundtruths_i, predictions_i) for
            groundtruths_i, predictions_i in zip(gt_boxes, pred_boxes)
        ]
        mapper = {
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "pred_boxes": pred_boxes,
            "pred_objectness": pred_objectness,
            "pred_classes": pred_classes,
            "iou_matrices": iou_matrices,
        }
        results = {}
        self.last_result_str = []

        for metric_name, metric in self.metrics.items():
            # Selectively call metric with its arguments
            kwargs = {}
            call_args = inspect.getfullargspec(
                metric.call)[0][1:]  # not including `self`
            for arg in call_args:
                assert arg in mapper  # function args MUST be in `mapper` keys
                kwargs[arg] = mapper[arg]
            result = metric(**kwargs)
            results[metric_name] = result
            self.last_result_str.append(metric.get_str())

        return results

    def get_str(self):
        return self.last_result_str

    def write_to_tensorboard(self, writer):
        for metric_name, metric in self.metrics.items():
            metric.write_to_tensorboard(writer)
