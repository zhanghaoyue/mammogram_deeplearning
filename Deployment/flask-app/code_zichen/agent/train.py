import argparse
import json
import os

import torch

from one_epoch_trainer import Trainer
from utils import Logger


def main(config, resume):
    train_logger = Logger()

    trainer = Trainer(resume=resume,
                      config=config,
                      train_logger=train_logger)

    trainer.train_val()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BE223C')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    # prevent GPU memory leak, GPU0 is the default gpu in pytorch
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
