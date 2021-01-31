import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from shutil import copy, SameFileError
from functools import partial

import torch
import yaml
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader

from faster_rcnn.models import BackboneModel, RPNModel, FasterRCNN
from faster_rcnn.train import train
from faster_rcnn.evaluate import evaluate
from faster_rcnn.data import collate_fn
from faster_rcnn.transforms import get_transforms
from faster_rcnn.metrics import MetricHolder
from faster_rcnn.registry import registry
from faster_rcnn.utils import ConfigComparer


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

    # Unpack model hyperparameters
    model_info = config["model"]

    # Unpack training hyperparameters
    training_info = config["training"]
    work_dir = training_info["work_dir"]
    train_transforms = get_transforms(
        input_size=training_info["input_size"],
        transforms_mode=training_info["transforms_mode"])
    device = training_info["device"]
    lr = training_info["learning_rate"]
    train_batch_size = training_info["batch_size"]
    num_epochs = training_info["num_epochs"]
    num_workers = training_info["num_workers"]
    testing = training_info.get("testing", False)

    # Metrics
    metrics_info = training_info["metrics"]
    rpn_metrics = MetricHolder(metrics_info["rpn"], config)

    # Unpack evaluating hyperparameters
    evaluate_info = config["evaluating"]
    test_transforms = get_transforms(
        input_size=evaluate_info["input_size"],
        transforms_mode=evaluate_info["transforms_mode"])

    load_from = args.load_from
    resume_from = args.resume_from

    # Get save directory
    if resume_from is None:
        if work_dir is not None:
            curr_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join(work_dir, curr_time)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = None
    else:
        save_dir = os.path.realpath(resume_from)
        assert os.path.exists(save_dir)
    training_info["save_dir"] = save_dir

    # Get logger
    logger.remove()  # remove default handler
    logger.add(
        sys.stderr, colorize=True,
        format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {message}")
    if save_dir is not None:
        logger_path = os.path.join(save_dir, "training.log")
        logger.add(logger_path, mode="a",
                   format="{time:YYYY-MM-DD at HH:mm:ss} | {message}")
        logger.info(f"Working directory: {save_dir}")

    # Print config
    logger.info(f"Config:\n{json.dumps(config, indent=2)}")

    # Train, val and test data
    data_info = config["data"]
    dataloaders = {}

    # Collate function
    collate_fn_init_train = partial(
        collate_fn, transforms=train_transforms)
    if training_info["transforms_mode"] == "no_pad":
        train_batch_size = 1
    collate_fn_init_test = partial(
        collate_fn, transforms=test_transforms)
    if evaluate_info["transforms_mode"] == "no_pad":
        test_batch_size = 1
    else:
        test_batch_size = train_batch_size

    # Initialize dataloaders
    logger.info("Initializing dataloaders...")
    for set_name, set_info in data_info.items():
        if set_name == "train":
            shuffle = True
            collate = collate_fn_init_train
            batch_size = train_batch_size
        else:
            shuffle = False
            collate = collate_fn_init_test
            batch_size = test_batch_size

        main_args = f"data->{set_name}"
        dataloaders[set_name] = initialize_dataloaders(
            config, format=set_info["format"], main_args=main_args,
            collate_fn=collate, batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle)

    # Initialize model
    device = torch.device(device)
    logger.info("Initializing model...")
    backbone_model = BackboneModel(config).to(device)
    model_info["backbone_model"] = backbone_model

    rpn_model = RPNModel(config).to(device)
    faster_rcnn = FasterRCNN(backbone_model, rpn_model)
    optimizer = optim.Adam(
        [params for params in faster_rcnn.parameters()
         if params.requires_grad],
        lr=lr)

    if load_from is not None and resume_from is not None:
        raise ValueError(
            "`load_from` and `resume_from` are mutually exclusive.")

    # Load from a pretrained model
    if load_from is not None:
        load_from = os.path.realpath(load_from)
        logger.info(f"Loading model at {load_from}")
        # Ensure that the two configs match (with some exclusions)
        load_dir, _ = os.path.split(load_from)
        with open(os.path.join(load_dir, "config.yaml"), "r") as conf:
            resume_config = yaml.load(conf, Loader=yaml.FullLoader)

    if resume_from is not None:
        # Ensure that the two configs match (with some exclusions)
        with open(os.path.join(save_dir, "config.yaml"), "r") as conf:
            resume_config = yaml.load(conf, Loader=yaml.FullLoader)

        # Load the most recent saved model
        model_list = Path(save_dir).glob("checkpoint*.pth")
        load_from = max(model_list, key=os.path.getctime)  # last saved model
        logger.info(f"Loading most recent saved model at {load_from}")
        # Get some more info for resuming training
        _, last_name = os.path.split(load_from)
        last_name, _ = os.path.splitext(last_name)
        last_epoch = int(last_name.split("_")[-1])

    if load_from is not None:  # this also includes `resume_from`
        compare_config = ConfigComparer(config, resume_config)
        compare_config.compare()

        checkpoint = torch.load(load_from, map_location=device)
        backbone_model.load_state_dict(checkpoint["models"]["backbone"])
        rpn_model.load_state_dict(checkpoint["models"]["rpn"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Copy config
    if save_dir is not None:
        copy_from = os.path.realpath(args.config_path)
        copy_to = os.path.realpath(os.path.join(save_dir, "config.yaml"))
        try:
            copy(copy_from, copy_to)
        except SameFileError:
            pass

    # Start training and evaluating
    for epoch in range(1, num_epochs + 1):
        if resume_from is not None:
            if epoch < last_epoch:
                continue

        # Train
        train(
            faster_rcnn, dataloaders["train"], optimizer, device,
            epoch=epoch, total_epoch=num_epochs, testing=testing)

        # Evaluate
        evaluate(
            faster_rcnn, dataloaders["val"], device, rpn_metrics=rpn_metrics,
            prefix=f"Validation (epoch: {epoch}/{num_epochs}): ",
            testing=testing)

        # Save model
        if save_dir is not None:
            save_path = os.path.join(
                save_dir, f"checkpoint_{epoch}.pth")
            checkpoint = {
                "models": {
                    "backbone": backbone_model.state_dict(),
                    "rpn": rpn_model.state_dict(),
                },
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, save_path)
            logger.info(f"Checkpoint saved to {save_path}.")

    # Test
    evaluate(
        rpn_model, dataloaders["test"], device, rpn_metrics=rpn_metrics,
        prefix="Test: ")
    logger.info("Training finished.")


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
