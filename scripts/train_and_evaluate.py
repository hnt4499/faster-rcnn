import os
import sys
import json
import argparse
import datetime
from shutil import copy, SameFileError
from functools import partial

import torch
import yaml
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader

from faster_rcnn.models import BackboneModel, RPNModel
from faster_rcnn.train import train
from faster_rcnn.evaluate import evaluate
from faster_rcnn.data import get_dataset, collate_fn
from faster_rcnn.transforms import get_transforms


DESCRIPTION = """Train and evaluate a Faster R-CNN model."""


def initialize_dataloaders(info, collate_fn, batch_size, num_workers,
                           shuffle=True):
    paths = info["paths"]
    format = info["format"]
    kwargs = info["kwargs"]

    Dataset = get_dataset(format)
    dataset = Dataset(paths, **kwargs)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers)

    return dataloader


def main(args):
    with open(args.config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)

    # Unpack model hyperparameters
    model_info = config["model"]
    backbone_name = model_info["backbone"]
    backbone_freeze_all = model_info["backbone_freeze_all"]
    anchor_areas = model_info["anchor_areas"]
    aspect_ratios = model_info["aspect_ratios"]
    kernel_size = model_info["kernel_size"]
    num_channels = model_info["num_channels"]
    sampler = model_info["sampler"]
    positive_fraction = model_info["positive_fraction"]
    batch_size_per_image = model_info["batch_size_per_image"]
    reg_lambda = model_info["reg_lambda"]
    normalize_offsets = model_info["normalize_offsets"]
    handle_cross_boundary_boxes = model_info["handle_cross_boundary_boxes"]

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

    # Unpack evaluating hyperparameters
    evaluate_info = config["evaluating"]
    evaluate_every = evaluate_info["evaluate_every"]
    test_transforms = get_transforms(
        input_size=evaluate_info["input_size"],
        transforms_mode=evaluate_info["transforms_mode"])
    # Post-processing hyperparameters
    # RPN
    rpn_info = evaluate_info["post_process"]["rpn"]
    rpn_pre_nms_top_n = rpn_info["pre_nms_top_n"]
    rpn_post_nms_top_n = rpn_info["post_nms_top_n"]
    rpn_nms_iou_threshold = rpn_info["nms_iou_threshold"]
    rpn_score_threshold = rpn_info["score_threshold"]
    rpn_min_size = rpn_info["min_size"]

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

        dataloaders[set_name] = initialize_dataloaders(
            info=set_info, collate_fn=collate, batch_size=batch_size,
            num_workers=num_workers, shuffle=shuffle)

    # Initialize model
    device = torch.device(device)
    logger.info("Initializing model...")
    backbone_model = BackboneModel(
        model_name=backbone_name, freeze_all=backbone_freeze_all
    ).to(device)
    rpn_model = RPNModel(
        backbone_model, anchor_areas, aspect_ratios, kernel_size, num_channels,
        sampler, positive_fraction, batch_size_per_image, reg_lambda,
        normalize_offsets=normalize_offsets,
        handle_cross_boundary_boxes=handle_cross_boundary_boxes,
        pre_nms_top_n=rpn_pre_nms_top_n, post_nms_top_n=rpn_post_nms_top_n,
        nms_iou_threshold=rpn_nms_iou_threshold,
        score_threshold=rpn_score_threshold, min_size=rpn_min_size
    ).to(device)
    rpn_optimizer = optim.Adam(
        [params for params in rpn_model.parameters() if params.requires_grad],
        lr=lr)

    if load_from is not None and resume_from is not None:
        raise ValueError(
            "`load_from` and `resume_from` are mutually exclusive.")

    # # Load from a pretrained model
    # if load_from is not None:
    #     load_from = os.path.realpath(load_from)
    #     logger.info(f"Loading model at {load_from}")
    #     model.load_state_dict(torch.load(load_from, map_location=device))
    #
    # if resume_from is not None:
    #     # Ensure that the two configs match (with some exclusions)
    #     with open(os.path.join(save_dir, "config.yaml"), "r") as conf:
    #         resume_config = yaml.load(conf, Loader=yaml.FullLoader)
    #     if not compare_config(config, resume_config):
    #         raise RuntimeError("The two config files do not match.")
    #
    #     # Load the most recent saved model
    #     model_list = Path(save_dir).glob("model*.pth")
    #     last_saved_model = max(model_list, key=os.path.getctime)
    #     logger.info(f"Loading most recent saved model at {last_saved_model}")
    #     model.load_state_dict(
    #         torch.load(last_saved_model, map_location=device))
    #     # Get some more info for resuming training
    #     _, last_name = os.path.split(last_saved_model)
    #     last_name, _ = os.path.splitext(last_name)
    #     _, last_epoch, last_dataloader_i = last_name.split("_")
    #     last_epoch, last_dataloader_i = int(last_epoch), int(last_dataloader_i)

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
        # if resume_from is not None:
        #     if epoch < last_epoch:
        #         continue
        #     elif epoch == last_epoch and dataloader_i <= last_dataloader_i:
        #         continue

        evaluator = partial(
            evaluate, dataloader=dataloaders["val"],
            device=device, testing=testing)

        # Train
        train(
            rpn_model, dataloaders["train"], rpn_optimizer, device,
            epoch=epoch, total_epoch=num_epochs, testing=testing,
            evaluate_every=evaluate_every, evaluator=evaluator)

        # Evaluate
        evaluator(
            model=rpn_model,
            prefix=f"[Validation (epoch: {epoch}/{num_epochs})] ")

        # Save model
        if save_dir is not None:
            save_path = os.path.join(
                save_dir, f"model_{epoch}.pth")
            torch.save(rpn_model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    # Test
    evaluate(
        rpn_model, dataloaders["test"], device, prefix="[Test] ")
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
