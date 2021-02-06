from datetime import datetime

import torch
from torchvision.ops import box_iou
from tensorboardX import SummaryWriter

from .vis_utils import draw_multiple_box_sets_comparison, draw_boxes_on_image


class TensorboardWriter:
    def __init__(self, log_dir, plot_gt_pred_comparison=None):
        self.log_dir = log_dir
        self.plot_gt_pred_comparison = plot_gt_pred_comparison
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        self.writer = SummaryWriter(log_dir)

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

        self.tb_writer_ftns = {
            "add_scalar", "add_scalars",
            "add_image", "add_image_with_boxes", "add_images",
            "add_figure", "add_histogram", "add_pr_curve",
            "add_audio", "add_text", "add_embedding",
        }

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """Wrapper that finds tensorboardX's `add_data()` methods (e.g.,
        `add_scalar`, `add_figure`) and wraps it with additional information
        (i.e., current mode of training)."""
        if self.log_dir is None:
            def do_nothing(*args, **kwargs):
                pass
            return do_nothing

        # Validate
        if self.mode not in ["train", "val", "test"]:
            if self.mode == "":
                raise RuntimeError("Mode must be set.")
            else:
                raise RuntimeError(
                    f"Invalid mode. Expected one of ['train', 'val', 'test'], "
                    f"got '{self.mode}' instead.")

        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # Add mode(train/valid) tag
                    tag = f"{self.mode}/{tag}"
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            try:
                attr = self.writer.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "type object \"{}\" has no attribute "
                    "\"{}\"".format(self.writer.__class__.__name__, name))
            return attr

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def _write_image_comparison(self, data, step, opts):
        if opts["enabled"] and step % opts["interval"] == 0 and \
                all(i in data for i in
                    ["images", "gt_boxes", "pred_boxes", "image_boundaries"]):

            num_boxes = opts.get("num_boxes", 10)
            # Only visualize the first image in the batch
            image = data["images"][0].cpu().permute(1, 2, 0).numpy()
            image = (image * self.image_std + self.image_mean)
            image = (image * 255).astype("uint8")

            image_boundary = data["image_boundaries"][0].cpu().numpy()
            gt_boxes = data["gt_boxes"][0].cpu()
            pred_boxes = data["pred_boxes"][0].cpu()

            # Draw top predicted boxes, side by side with the gt boxes
            grid = draw_multiple_box_sets_comparison(
                image=image.copy(), labels=None, image_boundary=image_boundary,
                boxes=[gt_boxes.numpy(), pred_boxes.numpy()[:num_boxes]],
                ncol=2, colors=[(0, 255, 0), (0, 0, 255)])
            self.add_image("gt_vs_pred", grid)

            # Draw best overlapped boxes
            pred_probs = data["pred_probs"][0].cpu()
            iou = box_iou(gt_boxes, pred_boxes)
            idxs = torch.argmax(iou, dim=-1)

            best_overlap_boxes = torch.index_select(
                pred_boxes, dim=0, index=idxs)
            best_overlap_scores = pred_probs[idxs].tolist()
            best_overlap_scores = [f"{s:.2f}" for s in best_overlap_scores]

            # Labels and colors
            all_boxes = torch.cat([gt_boxes, best_overlap_boxes], dim=0)
            labels = [None] * len(gt_boxes) + best_overlap_scores
            colors = ([(0, 255, 0)] * len(gt_boxes)
                      + [(0, 0, 255)] * len(pred_boxes))

            drawn = draw_boxes_on_image(
                image.copy(), all_boxes.numpy(), labels=labels,
                image_boundary=image_boundary, color=colors,
                line_thickness=None)
            self.add_image(
                "best_overlaps", torch.from_numpy(drawn).permute(2, 0, 1))

    def write_one_step_train(self, data, step):
        self.set_step(step=step, mode="train")

        self.add_scalar(
            "loss", data["loss"].item(), display_name="Total RPN loss")
        self.add_scalar(
            "loss_cls", data["loss_cls"].item(),
            display_name="RPN classification loss")
        self.add_scalar(
            "loss_t", data["loss_t"].item(),
            display_name="RPN regression loss")

        # Draw images
        if isinstance(self.plot_gt_pred_comparison, dict):
            opts = self.plot_gt_pred_comparison["during_training"]
            self._write_image_comparison(data, step, opts)

    def write_one_step_test(self, data, step, mode, rpn_metrics=None):
        self.set_step(step=step, mode=mode)
        if rpn_metrics is not None:
            rpn_metrics.write_to_tensorboard(self)

        # Draw images
        if isinstance(self.plot_gt_pred_comparison, dict):
            opts = self.plot_gt_pred_comparison["during_testing"]
            self._write_image_comparison(data, step, opts)

    def write_one_step(self, data, step, mode, **kwargs):
        """Wrapper that helps to write all necessary information during
        training/testing at a particular step."""
        if mode == "train":
            return self.write_one_step_train(data, step, **kwargs)
        elif mode in ["val", "test"]:
            return self.write_one_step_test(data, step, mode, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")


writers = {}


def get_writer(log_dir, *args, **kwargs):
    if log_dir not in writers:
        writers[log_dir] = TensorboardWriter(log_dir, *args, **kwargs)
    return writers[log_dir]
