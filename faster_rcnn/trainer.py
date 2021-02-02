import os
import sys
import json
import datetime
from pathlib import Path
from shutil import copy, SameFileError

import torch
import yaml
from loguru import logger
from torch import optim

from faster_rcnn.models import BackboneModel, RPNModel, FasterRCNN
from faster_rcnn.train import train
from faster_rcnn.evaluate import evaluate
from faster_rcnn.data import DataLoader
from faster_rcnn.registry import registry
from faster_rcnn.utils import from_config, ConfigComparer


class Trainer:
    @from_config(requires_all=True)
    def __init__(self, train_collate_fn, test_collate_fn, rpn_metrics,
                 work_dir, config_path, resume_from=None):
        # Get save dir
        self._get_save_dir(work_dir, resume_from)
        # Get logger
        self._get_logger()
        # Print config
        logger.info(f"Config:\n{json.dumps(self.config, indent=2)}")

        # Initialize dataloaders
        if self.config["training"]["transforms_mode"] == "no_pad":
            self.config["training"]["batch_size"] = 1
        if self.config["evaluating"]["transforms_mode"] == "no_pad":
            self.config["evaluating"]["batch_size"] = 1
        else:
            self.config["evaluating"]["batch_size"] = self.config[
                "training"]["batch_size"]

        logger.info("Initializing dataloaders...")
        self._initialize_dataloaders(train_collate_fn, test_collate_fn)

        # Copy config
        if self.save_dir is not None:
            copy_from = os.path.realpath(config_path)
            copy_to = os.path.realpath(
                os.path.join(self.save_dir, "config.yaml"))
            try:
                copy(copy_from, copy_to)
            except SameFileError:
                pass

        # Initialize models, optimizers and load state dicts (if possible)
        self._initialize_models(self.config)

        # Set additional attributes
        self.rpn_metrics = rpn_metrics

    def _get_save_dir(self, work_dir, resume_from):
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
        self.config["training"]["save_dir"] = self.save_dir = save_dir

    def _get_logger(self):
        # Get logger
        logger.remove()  # remove default handler
        logger.add(
            sys.stderr, colorize=True,
            format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {message}")
        if self.save_dir is not None:
            logger_path = os.path.join(self.save_dir, "training.log")
            logger.add(logger_path, mode="a",
                       format="{time:YYYY-MM-DD at HH:mm:ss} | {message}")
            logger.info(f"Working directory: {self.save_dir}")
        self.logger = logger

    def _initialize_dataloaders(self, train_collate_fn, test_collate_fn):
        self.dataloaders = {}

        for set_name, set_info in self.config["data"].items():
            if set_name == "train":
                main_args_2 = "training"
                self.config["training"]["shuffle"] = True
                self.config["training"]["collate_fn"] = train_collate_fn
            else:
                main_args_2 = "evaluating"
                self.config["evaluating"]["shuffle"] = False
                self.config["evaluating"]["collate_fn"] = test_collate_fn

            main_args = f"data->{set_name},{main_args_2}"
            self.dataloaders[set_name] = self._initialize_dataloaders_helper(
                format=set_info["format"], main_args=main_args)

    def _initialize_dataloaders_helper(self, format, main_args):
        dataset = registry["dataset"][format](self.config, main_args=main_args)
        dataloader = DataLoader(self.config, dataset=dataset,
                                main_args=main_args)
        return dataloader

    @from_config(requires_all=True)
    def _initialize_models(self, learning_rate, load_from, resume_from,
                           device):
        """Initialize models and optimizer(s), and load state dictionaries, if
        possible."""
        # Initialize model
        self.device = torch.device(device)
        logger.info("Initializing model...")
        self.backbone_model = BackboneModel(self.config).to(self.device)
        self.config["model"]["backbone_model"] = self.backbone_model

        self.rpn_model = RPNModel(self.config).to(self.device)
        self.faster_rcnn = FasterRCNN(self.backbone_model, self.rpn_model)
        self.optimizer = optim.Adam(
            [params for params in self.faster_rcnn.parameters()
             if params.requires_grad],
            lr=learning_rate)

        if load_from is not None and resume_from is not None:
            raise ValueError(
                "`load_from` and `resume_from` are mutually exclusive.")

        # Load from a pretrained model
        self.start_epoch = 0
        if load_from is not None:
            load_from = os.path.realpath(load_from)
            logger.info(f"Loading model at {load_from}")
            # Ensure that the two configs match (with some exclusions)
            load_dir, _ = os.path.split(load_from)
            with open(os.path.join(load_dir, "config.yaml"), "r") as conf:
                resume_config = yaml.load(conf, Loader=yaml.FullLoader)

        if resume_from is not None:
            # Ensure that the two configs match (with some exclusions)
            with open(os.path.join(self.save_dir, "config.yaml"), "r") as conf:
                resume_config = yaml.load(conf, Loader=yaml.FullLoader)

            # Load the most recent saved model
            model_list = Path(self.save_dir).glob("checkpoint*.pth")
            load_from = max(
                model_list, key=os.path.getctime)  # last saved model
            logger.info(f"Loading most recent saved model at {load_from}")
            # Get some more info for resuming training
            _, last_name = os.path.split(load_from)
            last_name, _ = os.path.splitext(last_name)
            self.start_epoch = int(last_name.split("_")[-1])

        if load_from is not None:  # this also includes `resume_from`
            compare_config = ConfigComparer(self.config, resume_config)
            compare_config.compare()

            checkpoint = torch.load(load_from, map_location=self.device)
            self.backbone_model.load_state_dict(
                checkpoint["models"]["backbone"])
            self.rpn_model.load_state_dict(checkpoint["models"]["rpn"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def _save_models(self):
        # Save model
        if self.save_dir is not None:
            save_path = os.path.join(
                self.save_dir, f"checkpoint_{self.epoch}.pth")
            checkpoint = {
                "models": {
                    "backbone": self.backbone_model.state_dict(),
                    "rpn": self.rpn_model.state_dict(),
                },
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, save_path)
            logger.info(f"Checkpoint saved to {save_path}.")

    @from_config(requires_all=True)
    def _train(self, load_from, num_epochs, testing=False):
        # Evaluate model before training starts
        if load_from is not None:
            logger.info("Running validation before training...")
            evaluate(
                self.faster_rcnn, self.dataloaders["val"], self.device,
                rpn_metrics=self.rpn_metrics, testing=testing,
                prefix="Validation (before training): ")

        # Start training and evaluating
        for epoch in range(self.start_epoch, num_epochs + 1):
            self.epoch = epoch

            # Train
            train(
                self.faster_rcnn, self.dataloaders["train"], self.optimizer,
                self.device, epoch=epoch, total_epoch=num_epochs,
                testing=testing)

            # Evaluate
            evaluate(
                self.faster_rcnn, self.dataloaders["val"], self.device,
                rpn_metrics=self.rpn_metrics, testing=testing,
                prefix=f"Validation (epoch: {epoch}/{num_epochs}): ")

            # Checkpoint
            self._save_models()

        # Test
        evaluate(
            self.rpn_model, self.dataloaders["test"], self.device,
            rpn_metrics=self.rpn_metrics, prefix="Test: ")
        logger.info("Training finished.")

    def train(self):
        return self._train(self.config)
