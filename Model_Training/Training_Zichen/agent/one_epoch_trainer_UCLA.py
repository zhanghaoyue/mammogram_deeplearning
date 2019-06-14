import sys

sys.path.insert(0, '../')
import numpy as np
import torch
from sklearn import metrics

from sklearn.metrics import auc
from base_trainer_UCLA import BaseTrainer


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
            loss_list = []
            acc_list = []
            pred_list = []
            gt_list = []
            for i_batch, data in enumerate(epoch_dataloader):

                # train Dis
                images, mass_region_attentions, labels = data
                images, mass_region_attentions, labels = images.to(self.device), \
                                                         mass_region_attentions.to(self.device), labels.to(self.device)

                probs = self.model(images, mass_region_attentions)
                probs = probs.flatten()

                weight = self.class_weight[labels.long()].to(self.device).flatten()

                criterion = torch.nn.BCELoss(weight=weight).to(self.device)

                loss = criterion(probs, labels.float())
                loss_list.append(loss.item())

                pred = (probs.data.cpu().numpy().flatten() > 0.5).astype(np.int).tolist()
                gt = labels.data.cpu().numpy().flatten().tolist()

                pred_list += pred
                gt_list += gt

                acc = np.sum(np.array(pred) == np.array(gt)) / float(len(gt))
                acc_list.append(acc)

                if phase == 'train':
                    # update dis scheduler and optimizer
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # log iter loss and score
                if i_batch % self.config['trainer']['log_write_iteration'] == 0:
                    self.writer.set_step((epoch - 1) * len(epoch_dataloader) + i_batch, mode=phase)
                    self.writer.add_scalars('iter_loss', loss.item())
                    self.writer.add_scalars('iter_acc', acc)

            fpr, tpr, thresholds = metrics.roc_curve(gt_list, pred_list, pos_label=1)
            epoch_auc_score = auc(fpr, tpr)

            epoch_loss = np.array(loss_list).mean()
            epoch_acc = np.array(acc_list).mean()

            # write epoch metric to logger and tensorboard
            log = {}
            self.writer.set_step(epoch, mode=phase)
            log.update({'%s_loss' % phase: epoch_loss})
            log.update({'%s_acc' % phase: epoch_acc})
            log.update({'%s_auc_score' % phase: epoch_auc_score})
            self.writer.add_scalars('epoch_loss', epoch_loss)
            self.writer.add_scalars('epoch_acc', epoch_acc)
            self.writer.add_scalars('epoch_auc_score', epoch_auc_score)

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
