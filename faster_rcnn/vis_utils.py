import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


def crop_to_image_boundary(image, boxes, image_boundary):
    """Crop transformed image to original image boundaries (e.g., remove
    padding).

    Parameters
    ----------
    image : np.ndarray
        Numpy integer array of shape (H, W, C).
    boxes : np.ndarray
        Numpy array of shape (N, 4), where N is the number of boxes.
    image_boundary : np.ndarray or None
        Numpy array of shape (4,). If None, no cropping will be performed.
    """
    if image_boundary is None:
        return image, boxes

    height, width = image.shape[:2]
    xmin, ymin, xmax, ymax = image_boundary.astype("int")
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, width)
    ymax = min(ymax, height)

    cropped_image = image[ymin:ymax, xmin:xmax]
    cropped_boxes = []

    for box in boxes:
        cropped_boxes.append(box - [xmin, ymin, xmin, ymin])

    return cropped_image, cropped_boxes


def draw_box_on_image(image, box, label=None, color=(0, 255, 0),
                      line_thickness=None):
    """Plot one bounding box on image.

    Adapted from
        https://github.com/ultralytics/yolov5/blob/73a066993051339f6adfe5095a7852a2b9184c16/utils/plots.py#L57

    Parameters
    ----------
    image : np.ndarray
        Numpy integer array of shape (H, W, C).
    box : np.ndarray
        Numpy array of shape (4,).
    label : str
    color : tuple or None
        Tuple of three R, G, B values.
    line_thickness: int or None
    """
    if (not isinstance(image, np.ndarray)) or \
            (not isinstance(box, np.ndarray)):
        raise TypeError(f"`image` and `box` must be numpy arrays. Got "
                        f"{type(image)} and {type(box)} instead")

    if line_thickness is None:
        line_thickness = round(  # line/font thickness
            0.001 * (image.shape[0] + image.shape[1]) / 2) + 1
    if color is None:
        color = (0, 255, 0)

    # Draw box
    box = box.astype("int")
    corner_1, corner_2 = tuple(box[:2]), tuple(box[2:])
    cv2.rectangle(image, corner_1, corner_2, color, thickness=line_thickness,
                  lineType=cv2.LINE_AA)

    # Draw label
    if label is not None:
        font_thickness = max(line_thickness - 1, 1)  # font thickness
        text_size = cv2.getTextSize(
            label, 0, fontScale=line_thickness / 3,
            thickness=font_thickness)[0]
        corner_2 = corner_1[0] + text_size[0], corner_1[1] - text_size[1] - 3

        cv2.rectangle(image, corner_1, corner_2, color, -1, cv2.LINE_AA)
        cv2.putText(
            image, label, (corner_1[0], corner_1[1] - 2), 0,
            line_thickness / 3, [225, 255, 255],
            thickness=font_thickness, lineType=cv2.LINE_AA)


def draw_boxes_on_image(image, boxes, labels=None, color=(0, 255, 0),
                        line_thickness=None):
    """Plot multiple bounding boxes on image

    Parameters
    ----------
    image : np.ndarray
        Numpy integer array of shape (H, W, C).
    boxes : np.ndarray
        Numpy array of shape (N, 4), where N is the number of boxes.
    labels : list-like or None
        List-like object of size N, where N is the number of boxes.
    color : tuple or None
        Tuple of three R, G, B values.
    line_thickness: int or None
    """
    if line_thickness is None:
        line_thickness = round(  # line/font thickness
            0.001 * (image.shape[0] + image.shape[1]) / 2) + 1

    for i, box in enumerate(boxes):
        label = None if labels is None else labels[i]
        draw_box_on_image(image, box, label, color, line_thickness)


def draw_multiple_box_sets_comparison(
        image, boxes, labels=None, image_boundary=None, colors=None,
        line_thickness=None, ncol=None):
    """
    Plot multiple version of the same image, each with a set of boxes.

    Parameters
    ----------
    image : np.ndarray
        Numpy integer array of shape (H, W, C).
    boxes : list[np.ndarray]
        List of size `num_sets` of numpy array, each of shape (N, 4), where
        `num_sets` is the number of box sets being considered, and N is the
        number of boxes.
    labels : list[list] or None
        List of size `num_sets`, where each element is a list-like object of
        size N, where N is the number of boxes.
    image_boundary : np.ndarray or None
        Numpy array of shape (4,) denoting the original image boundary as a
        bounding box in the transformed image.
    colors : list[tuple] or None
        List of size `num_sets`, where each element is a tuple of three R, G, B
        values denoting the box colors for the i-th box set.
    line_thickness: list[int] or None
        List of size `num_sets`, where each element is an integer corresponding
        to the i-th box set.
    ncol : int or None
        Number of columns of the grid. If None, number of **rows** will be 1.
    """
    if labels is None:
        labels = [None] * len(boxes)
    if colors is None:
        colors = [None] * len(boxes)
    if line_thickness is None:
        line_thickness = [None] * len(boxes)
    assert len(boxes) == len(labels) == len(colors) == len(line_thickness)

    image, boxes = crop_to_image_boundary(image, boxes, image_boundary)

    drawn_images = []
    for boxes_i, labels_i, colors_i, line_thickness_i in \
            zip(boxes, labels, colors, line_thickness):
        image_i = image.copy()
        draw_boxes_on_image(image_i, boxes_i, labels_i, colors_i,
                            line_thickness_i)

        drawn_images.append(torch.from_numpy(image_i).permute(2, 0, 1))

    if ncol is None:
        ncol = len(boxes)
    grid = make_grid(drawn_images, nrow=ncol)
    return grid
