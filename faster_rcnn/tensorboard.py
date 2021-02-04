from datetime import datetime

from tensorboardX import SummaryWriter


class TensorboardWriter:
    def __init__(self, log_dir):
        self.log_dir = log_dir
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


writers = {}


def get_writer(log_dir):
    if log_dir not in writers:
        writers[log_dir] = TensorboardWriter(log_dir)
    return writers[log_dir]
