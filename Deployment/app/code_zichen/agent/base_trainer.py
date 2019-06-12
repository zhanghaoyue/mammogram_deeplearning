import datetime
import logging
import math
import os
import sys

import torch
import torch.optim as optim

sys.path.append('../')

import graph.model.model as module_arch
from dataloader.data_loader import get_INBreast_dataloader
from utils.util import ensure_dir, get_instance
from utils.visualization import WriterTensorboardX


def build_optimizer(model, config):
    optim_config = config['optimizer']
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(optim, optim_config['type'])(trainable_params, **optim_config['args'])
    scheduler = get_instance(optim.lr_scheduler, 'scheduler', config, optimizer)

    return optimizer, scheduler


def model_parallel(model, device, device_ids):
    # The module must have its parameters and buffers on device_ids[0] before running DataParallel module.
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)  # used for displaying logging and warning info

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(self.config['n_gpu'])

        # dataloader
        self.train_set, self.val_set = get_INBreast_dataloader(self.config)

        # class weight for balanced dataset
        self.class_weight = [torch.FloatTensor([1, 20]), torch.FloatTensor([1, 4])]
        self.label_weight = [2, 1]
        self.output_ch = len(self.class_weight)

        # build model architecture
        self.model = module_arch.AttU_Net(img_ch=1, output_ch=self.output_ch)
        # print(self.model)

        # model parallel using muti-gpu
        self.model = model_parallel(self.model, self.device, self.device_ids)

        # build optimizer, learning rate scheduler.
        self.optimizer, self.scheduler = build_optimizer(self.model, self.config)

        self.train_logger = train_logger  # used for saving logging info

        trainer_config = self.config['trainer']
        self.max_epochs = trainer_config['max_epochs']
        self.save_period = trainer_config['save_period']
        self.val_period = trainer_config['val_period']
        self.verbosity = trainer_config['verbosity']
        self.monitor = trainer_config.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = trainer_config.get('early_stop', math.inf)

        self.start_epoch = 1

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        if self.save_period != 0:
            self.checkpoint_dir = os.path.join(trainer_config['save_dir'], self.config['name'], start_time)
            ensure_dir(self.checkpoint_dir)
        # setup visualization writer instance
        writer_dir = os.path.join(trainer_config['log_dir'], self.config['name'], start_time)
        ensure_dir(writer_dir)
        self.writer = WriterTensorboardX(writer_dir, self.logger, trainer_config['tensorboardX'])

        # Save configuration file into logging directory:
        self._save_config(writer_dir)

        if resume:
            self._resume_checkpoint(resume)

    def _save_config(self, writer_dir):
        # save conifg.json
        os.system('cp -r ../config %s/config' % writer_dir)
        # save model architecture
        os.system('cp -r ../graph %s/graph' % writer_dir)
        # save agent
        os.system('cp -r ../agent %s/agent' % writer_dir)
        # save dataloader
        os.system('cp -r ../dataloader %s/dataloader' % writer_dir)
        # save utils
        os.system('cp -r ../utils %s/utils' % writer_dir)

    def _prepare_device(self, n_gpu_use):
        """ 
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _param_init(self):
        pass

    def train_val(self):
        """
        Full training logic
        """
        self._param_init()

        for epoch in range(self.start_epoch, self.max_epochs + 1):

            result = self._train_epoch(epoch)
            result.update({'epoch': epoch})
            # save logged information into log dict
            if self.train_logger is not None:
                self.train_logger.add_entry(result)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if epoch % self.val_period == 0 and self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and result[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and result[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = result[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    # display best mnt metric
                    self.logger.info('Best %s: %f' % (self.mnt_metric, self.mnt_best))
                else:
                    # record period not count number
                    try:
                        not_improved_count += self.val_period
                    except:
                        not_improved_count = 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if self.save_period != 0:  # skip saving checkpoint when debugging model
                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=best)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        """
        state = {
            'exp_name': self.config['name'],
            'epoch': epoch,
            'logger': self.train_logger,
            'model_state_dict': self.model.state_dict(),
            'model_optimizer': self.optimizer.state_dict(),
            'model_scheduler': self.scheduler.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load model params from checkpoint.
        if checkpoint['config']['name'] != self.config['name']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed. 
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        self.optimizer.load_state_dict(checkpoint['model_optimizer'])

        # load scheduler state from checkpoint only when scheduler type is not changed
        if checkpoint['config']['scheduler']['type'] != self.config['scheduler']['type']:
            self.logger.warning('Warning: Scheduler type given in config file is different from that of checkpoint. ' + \
                                'Scheduler parameters not being resumed.')
        self.scheduler.load_state_dict(checkpoint['model_scheduler'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
