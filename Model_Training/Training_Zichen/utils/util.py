import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_instance(module, name, config, *args):
    # first get module using getattr and then assign config parameters
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class InfiniteLoopDataloader(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.unlimit_gen = self.generator(True)

    def generator(self, inf=False):
        while True:
            for images, labels in iter(self.data_loader):
                yield images, labels
            if not inf:
                break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()
