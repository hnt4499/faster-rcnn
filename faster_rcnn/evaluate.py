import torch
from loguru import logger


def evaluate(model, dataloader, device, prefix="", testing=False,
             rpn_metrics=None):
    """Evaluate the trained model.

    Parameters
    ----------
    model : nn.Module
        Initialized model.
    dataloader : torch.utils.data.DataLoader
        Test data loader.
    device : torch.device
        Device in which computation is performed.
    prefix : str
        Prefix for logging. (default: "")
    testing : bool
        If True, only run for 10 iterations. Useful for debugging and finding
        batch sizes, etc. (default: False)
    """
    model.eval()
    tot_gt_boxes = []
    tot_gt_labels = []
    tot_pred_boxes = []
    tot_pred_objectness = []

    with torch.no_grad():
        for i, (images, bboxes, labels, image_boundaries) in \
                enumerate(dataloader):
            images = images.to(device)
            bboxes = [bbox.to(device) for bbox in bboxes]
            labels = [label.to(device) for label in labels]
            image_boundaries = image_boundaries.to(device)

            # Forward
            output = model(inp=images, gt_boxes=None, labels=None,
                           image_boundaries=image_boundaries)
            tot_gt_boxes.extend(bboxes)
            tot_gt_labels.extend(labels)
            tot_pred_boxes.extend(output["pred_boxes"])
            tot_pred_objectness.extend(output["pred_probs"])

            # Break when reaching 10 iterations when testing
            if testing and i == 9:
                break

    metric_results = rpn_metrics(tot_gt_boxes, tot_gt_labels, tot_pred_boxes,
                                 tot_pred_objectness, pred_classes=None)
    results = ", ".join(rpn_metrics.get_str())
    logger.info(f"{prefix}{results}")

    model.train()
    return metric_results
