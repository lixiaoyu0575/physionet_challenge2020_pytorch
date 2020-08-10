import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from logger import setup_logging
from pathlib import Path

class BaseEvaluater:
    """
    Base class for all evaluaters
    """
    def __init__(self, model, criterion, metric_ftns, config, checkpoint_dir, result_dir):
        self.config = config
        self.logger = config.get_logger('evaluater', config['evaluater']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.result_dir = config.result_dir
        else:
            self.checkpoint_dir = config.save_dir

        if result_dir:
            self.result_dir = Path(result_dir)
        else:
            self.result_dir = config.result_dir

        setup_logging(self.result_dir)
        # self._resume_checkpoint(self.checkpoint_dir / 'model_best.pth')
        self._resume_checkpoint(self.checkpoint_dir)

    @abstractmethod
    def evaluate(self):
        """
        Evaluate logic
        """
        raise NotImplementedError


    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])


        # No need for loading optimizer
        # # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        #     self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
        #                         "Optimizer parameters not being resumed.")
        # else:
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded from epoch {}".format(self.start_epoch))
