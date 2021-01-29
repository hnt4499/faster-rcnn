from torchvision.ops import box_iou


all_metrics = {}


def register_metric(func):
    if func.__name__ in all_metrics:
        raise RuntimeError(f"Metric {func.__name__} already exists")
    all_metrics[func.__name__] = func
    return func


class BaseRPNMetric:
    """Base class for all metrics."""
    def __init__(self):
        self.last_value = None

    def call(self, groundtruths, predictions, predictions_score,
             iou_matrices=None):
        raise NotImplementedError

    def __call__(self, groundtruths, predictions, predictions_score,
                 iou_matrices=None):
        value = self.call(
            groundtruths, predictions, predictions_score, iou_matrices)
        self.last_value = value
        return value

    def get_str(self):
        """Used to log results"""
        raise NotImplementedError


@register_metric
class BoxRecall(BaseRPNMetric):
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

    def call(self, groundtruths, predictions, predictions_score,
             iou_matrices=None):
        """Calculate box recall for class-agnostic task.

        Parameters
        ----------
        groundtruths : list[Tensor]
            List of groundtruth boxes of size `batch_size`. Each element is a
            tensor of shape (M, 4).
        predictions : list[Tensor]
            List of predictions of size `batch_size`. Each element is a tensor
            of shape (N, 4).
        predictions_score : list[Tensor]
            Scores corresponding for each prediction in `predictions` tensor.
        iou_matrices : list[Tensor] or None
            List of NxM matrices.

        Returns
        -------
        recall_score
        """
        assert len(predictions) == len(groundtruths)
        if iou_matrices is None:
            iou_matrices = [
                box_iou(groundtruths_i, predictions_i) for
                groundtruths_i, predictions_i in zip(groundtruths, predictions)
            ]
            return self(groundtruths, predictions, predictions_score,
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


class RPNMetric:
    """Metric holder"""
    def __init__(self, metrics_config):
        self.metrics_config = metrics_config
        self.last_result_str = None
        self.metrics = {}
        for metric_name, metric_config in metrics_config.items():
            metric_initialized = all_metrics[metric_name](**metric_config)
            self.metrics[metric_name] = metric_initialized

    def __call__(self, groundtruths, predictions, predictions_score):
        """Calculate all metrics.

        Parameters
        ----------
        groundtruths : list[Tensor]
            List of groundtruth boxes of size `batch_size`. Each element is a
            tensor of shape (M, 4).
        predictions : list[Tensor]
            List of predictions of size `batch_size`. Each element is a tensor
            of shape (N, 4).
        predictions_score : list[Tensor]
            Scores corresponding for each prediction in `predictions` tensor.

        Returns
        -------
        results
            A dictionary mapping each metric name (e.g., "BoxRecall") with its
            computed value.
        """
        iou_matrices = [
            box_iou(groundtruths_i, predictions_i) for
            groundtruths_i, predictions_i in zip(groundtruths, predictions)
        ]
        results = {}
        self.last_result_str = []

        for metric_name, metric in self.metrics.items():
            result = metric(
                groundtruths, predictions, predictions_score, iou_matrices)
            results[metric_name] = result
            self.last_result_str.append(metric.get_str())

        return results

    def get_str(self):
        return self.last_result_str
