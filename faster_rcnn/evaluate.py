import torch


def evaluate(model, dataloader, device, prefix="", testing=False):
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

    with torch.no_grad():
        for i, (images, bboxes, labels, image_boundaries) in \
                enumerate(dataloader):
            images = images.to(device)
            bboxes = [bbox.to(device) for bbox in bboxes]
            labels = [label.to(device) for label in labels]
            image_boundaries = image_boundaries.to(device)

            # Forward
            output = model(images, bboxes, labels, image_boundaries)

            # Break when reaching 10 iterations when testing
            if testing and i == 9:
                break

    model.train()
    return
