from tqdm import tqdm
from loguru import logger

from .utils import Timer


def train(model, dataloader, optimizer, device, epoch=None, total_epoch=None,
          testing=False):
    """Train the model.

    Parameters
    ----------
    model : nn.Module
        Initialized model.
    dataloader : torch.utils.data.DataLoader
        Training data loader.
    optimizer : torch.optim
        Initialized optimizer for the model.
    epoch : int
        Current epoch index. Used to print meaningful progress bar with tqdm if
        specified.
    total_epoch : int
        Total number of epochs. Used to print meaningful progress bar with tqdm
        if specified.
    testing : bool
        If True, only run for 10 iterations. Useful for debugging and finding
        batch sizes, etc. (default: False)
    """
    model.train()
    timer = Timer()

    total = 10 if testing else len(dataloader)
    with tqdm(dataloader, total=total, leave=False) as t:
        if epoch is not None and total_epoch is not None:
            description = f"Training ({epoch}/{total_epoch})"
        else:
            description = "Training"
        t.set_description(description)

        for i, (images, bboxes, labels, image_boundaries) in enumerate(t):
            timer.start()

            images = images.to(device)
            bboxes = [bbox.to(device) for bbox in bboxes]
            labels = [label.to(device) for label in labels]
            image_boundaries = image_boundaries.to(device)

            optimizer.zero_grad()

            # Forward
            output = model(inp=images, gt_boxes=bboxes, labels=labels,
                           image_boundaries=image_boundaries)
            loss_cls = output["loss_cls"]
            loss_t = output["loss_t"]
            loss = output["loss"]

            # Backward
            loss.backward()
            optimizer.step()

            timer.end()
            speed = timer.get_accumulated_interval()

            t.set_postfix(
                loss=f"{loss.item():.4f}", loss_cls=f"{loss_cls.item():.4f}",
                loss_reg=f"{loss_t.item():.4f}", speed=f"{speed:.3f}s/it")

            # Break when reaching 10 iterations when testing
            if testing and i == 9:
                break

    logger.info(f"{description} took {timer.get_total_time():.2f}s.")
    return
