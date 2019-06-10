import numpy as np
import torch
import torch.nn as nn

from base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, resume, config, train_logger=None):
        super(Trainer, self).__init__(resume, config, train_logger)

    def _one_epoch(self, phase, epoch_data_loader, epoch):
        """
        Training logic for an epoch
        """
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        self.model_scheduler.step()

        with torch.set_grad_enabled(phase == 'train'):
            loss_list = []
            score_list = []
            for i_batch, data in enumerate(epoch_data_loader):

                # train Dis
                images, labels = data['image'], data['label']
                images, labels = images.to(self.device), labels.to(self.device)

                probs = self.model(images).flatten()

                weight = self.class_weight[labels.long()].to(self.device).flatten()

                criterion = torch.nn.BCELoss(weight=weight).to(self.device)

                loss = criterion(probs, labels.float())
                loss_list.append(loss.item())

                prediction = (probs > 0.5)
                prediction = prediction.to(self.device)

                # accuracy
                score = (prediction.long() == labels.long()).float().mean()
                score_list.append(score.item())

                if phase == 'train':
                    # update dis scheduler and optimizer
                    self.model_optimizer.zero_grad()
                    loss.backward()
                    self.model_optimizer.step()

                # log iter loss and score
                if i_batch % self.config['trainer']['log_write_iteration'] == 0:
                    self.writer.set_step((epoch - 1) * len(epoch_data_loader) + i_batch, mode=phase)
                    self.writer.add_scalars('iter_loss', loss.item())
                    self.writer.add_scalars('iter_score', score)

            epoch_loss = np.array(loss_list).mean()
            epoch_score = np.array(score_list).mean()

            # write epoch metric to logger and tensorboard
            log = {}
            self.writer.set_step(epoch, mode=phase)
            log.update({'%s_loss' % phase: epoch_loss})
            log.update({'%s_score' % phase: epoch_score})
            self.writer.add_scalars('epoch_loss', epoch_loss)
            self.writer.add_scalars('epoch_score', epoch_score)

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
