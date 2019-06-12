import sys
sys.path.insert(0, '../')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph.metric import *
from utils.visualization import visualize_generated_img

from base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, resume, config, train_logger=None):
        super(Trainer, self).__init__(resume, config, train_logger)

    def _one_epoch(self, phase, epoch_dataloader, epoch):
        """
        Training logic for an epoch
        """
        if phase == 'train':
            self.model.train()
            self.scheduler.step()
        else:
            self.model.eval()

        with torch.set_grad_enabled(phase == 'train'):
            loss_overall_list = []
            loss_list_dict = {'0': [], '1': []}
            dice_list_dict = {'0': [], '1': []}
            for batch_idx, data in enumerate(epoch_dataloader):
                images, GT = data
                images = images.to(self.device)
                GT = GT.to(self.device)

                # SR : Segmentation Result
                SR = self.model(images)
                SR_probs = F.sigmoid(SR)

                loss_overall = torch.FloatTensor([0]).to(self.device)
                for i in range(self.output_ch):
                    GT_sub = GT[:, i, :, :]

                    SR_probs_sub = SR_probs[:, i, :, :]

                    SR_flat = SR_probs_sub.view(SR_probs_sub.size(0), -1)

                    GT_flat = GT_sub.view(GT_sub.size(0), -1)

                    weight = self.class_weight[i][GT_flat.long()].to(self.device)
                    loss_fn = nn.BCELoss(weight=weight)
                    loss = loss_fn(SR_flat, GT_flat)

                    loss_overall += loss * self.label_weight[i]

                    dice = get_DC(SR_probs_sub, GT_sub)

                    loss_list_dict['%d' % i].append(loss.data.cpu().item())
                    dice_list_dict['%d' % i].append(dice)

                loss_overall_list.append(loss_overall.data.cpu().item())

                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss_overall.backward()
                    self.optimizer.step()

                # log iter loss and score
                if batch_idx % self.config['trainer']['log_write_iteration'] == 0:
                    self.writer.set_step((epoch - 1) * len(epoch_dataloader) + batch_idx, mode=phase)
                    self.writer.add_scalars('iter_loss_overall', loss_overall.item())
                    for i in range(self.output_ch):
                        self.writer.add_scalars('iter_loss_%d' % i, loss_list_dict['%d' % i][-1])
                        self.writer.add_scalars('iter_dice_%d' % i, dice_list_dict['%d' % i][-1])

            # write epoch metric to logger and tensorboard
            log = {}
            self.writer.set_step(epoch, mode=phase)
            log.update({'%s_loss_overall' % phase: np.array(loss_overall_list).mean()})
            self.writer.add_scalars('epoch_loss_overall', np.array(loss_overall_list).mean())

            for i in range(self.output_ch):
                log.update({'%s_loss_%d' % (phase, i): np.array(loss_list_dict['%d' % i]).mean()})
                self.writer.add_scalars('epoch_loss_%d' % i, np.array(loss_list_dict['%d' % i]).mean())
                log.update({'%s_dice_%d' % (phase, i): np.array(dice_list_dict['%d' % i]).mean()})
                self.writer.add_scalars('epoch_dice_%d' % i, np.array(dice_list_dict['%d' % i]).mean())

            # visualize generated images
            if epoch % self.config['trainer']['vis_period'] == 0 and phase == 'val':
                visualize_generated_img(self.writer, epoch, images, GT, SR_probs)

            # display epoch logging on the screen
            if self.verbosity >= 2 and epoch % self.config['trainer']['log_display_period'] == 0:
                self.logger.info('%s Epoch %d: %s' % (phase, epoch, str(log)))

        return log

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information of one epoch.

        """

        log = self._one_epoch('train', self.train_set, epoch)

        if self.verbosity >= 2 and epoch % self.config['trainer']['val_period'] == 0:
            val_log = self._one_epoch('val', self.val_set, epoch)
            # add val logging
            log = {**log, **val_log}

        # return log info to base_trainer.train_val()
        return log
