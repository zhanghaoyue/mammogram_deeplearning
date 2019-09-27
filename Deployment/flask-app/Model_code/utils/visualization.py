import importlib
import numpy as np
import torch
import torchvision.utils as vutils


def visualize_generated_img(writer, epoch, images, GT, SR_prob):
    images = vutils.make_grid(images, normalize=True, scale_each=True)
    writer.set_step(epoch, mode='train')
    writer.add_image('%s' % 'image', images)

    SR = (SR_prob > 0.5).float()
    for i in range(SR.shape[1]):
        SR_sub = vutils.make_grid(SR[:, i, :, :].unsqueeze(1), normalize=True, scale_each=True)
        GT_sub = vutils.make_grid(GT[:, i, :, :].unsqueeze(1), normalize=True, scale_each=True)
        writer.add_image('%s_%d' % ('ground_truth', i), GT_sub)
        writer.add_image('%s_%d' % ('segmentation_result', i), SR_sub)


class WriterTensorboardX():
    """
    Wrapper of tensorboardX.SummaryWriter
    """

    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                message = "Warning: TensorboardX visualization is configured to use, but currently not installed on this machine. " + \
                          "Please install the package by 'pip install tensorboardx' command or turn off the option in the 'config.json' file."
                logger.warning(message)
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text',
                                        'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing

        __getattr__ will allow you to “catch” references to attributes that don’t exist in this object
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    if name == 'add_scalars':
                        add_data('{}'.format(tag), {'{}'.format(self.mode): data}, self.step, *args, **kwargs)
                    else:
                        add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr
