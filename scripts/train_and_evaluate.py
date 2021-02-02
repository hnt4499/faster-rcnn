import sys
import argparse
from functools import partial

import yaml
from torch.utils.data import DataLoader

from faster_rcnn.data import collate_fn
from faster_rcnn.transforms import get_transforms
from faster_rcnn.metrics import MetricHolder
from faster_rcnn.registry import registry
from faster_rcnn.trainer import Trainer


DESCRIPTION = """Train and evaluate a Faster R-CNN model."""


def initialize_dataloaders(config, format, main_args, collate_fn, batch_size,
                           num_workers, shuffle=True):
    Dataset = registry["dataset"][format]
    dataset = Dataset(config, main_args=main_args)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers)

    return dataloader


def main(args):
    with open(args.config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
    config["load_from"] = args.load_from
    config["resume_from"] = args.resume_from
    config["config_path"] = args.config_path

    train_transforms = get_transforms(
        input_size=config["training"]["input_size"],
        transforms_mode=config["training"]["transforms_mode"])
    test_transforms = get_transforms(
        input_size=config["evaluating"]["input_size"],
        transforms_mode=config["evaluating"]["transforms_mode"])

    rpn_metrics = MetricHolder(config["training"]["metrics"]["rpn"], config)

    # Collate function
    train_collate_fn = partial(
        collate_fn, transforms=train_transforms)
    test_collate_fn = partial(
        collate_fn, transforms=test_transforms)

    # Initializer trainer
    trainer = Trainer(
        config, train_collate_fn=train_collate_fn,
        test_collate_fn=test_collate_fn, rpn_metrics=rpn_metrics)

    # Start training
    trainer.train()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        '-c', '--config-path', type=str, required=True,
        help='Path to full config.')
    parser.add_argument(
        '-r', '--resume-from', type=str, required=False, default=None,
        help='Directory to resume from. Mutually exclusive with `load_from`.')
    parser.add_argument(
        '-l', '--load-from', type=str, required=False, default=None,
        help='Path to the model to load from. Mutually exclusive with '
             '`resume_from`.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
