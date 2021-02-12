from tensorboardX import SummaryWriter


class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardWriter, self).__init__(logdir)

    def log_training(self, value, step):
        self.add_scalar('train_loss', value, step)

    def log_validation(self, value,  step):
        self.add_scalar('val_loss', value, step)

    def log_best_validation(self, value,  step):
        self.add_scalar('val_best_loss', value, step)
