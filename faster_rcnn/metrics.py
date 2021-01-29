import inspect

from torchvision.ops import box_iou


all_metrics = {}


def register_metric(func):
    if func.__name__ in all_metrics:
        raise RuntimeError(f"Metric {func.__name__} already exists")
    all_metrics[func.__name__] = func
    return func


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


@register_metric
class BoxRecall(BaseMetric):
    """Calculate box recall for class-agnostic task.

    Parameters
    ----------
    iou_threshold : float
        Groundtruths with IoU overlap with any of predictions higher than
        this threshold will be considered detected.
    """
    def __init__(self, iou_threshold=0.5):
        super(BoxRecall, self).__init__()
        self.iou_threshold = iou_threshold

    def call(self, gt_boxes, pred_boxes, pred_objectness, iou_matrices=None):
        """Calculate box recall for class-agnostic task."""
        assert len(gt_boxes) == len(pred_boxes)
        if iou_matrices is None:
            iou_matrices = [
                box_iou(gt_boxes_i, pred_boxes_i) for
                gt_boxes_i, pred_boxes_i in zip(gt_boxes, pred_boxes)
            ]
            return self(gt_boxes, pred_boxes, pred_objectness,
                        iou_matrices=iou_matrices)

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


class MetricHolder:
    """Metric holder"""
    def __init__(self, metrics_config):
        self.metrics_config = metrics_config
        self.last_result_str = None
        self.metrics = {}
        for metric_name, metric_config in metrics_config.items():
            metric_initialized = all_metrics[metric_name](**metric_config)
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
