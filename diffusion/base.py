import os
import pickle
import torch
from prettytable import PrettyTable
import numpy as np


def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table


def get_metric_table(metric_dict, epochs):
    table = PrettyTable()
    table.add_column('Epoch', epochs)
    if len(metric_dict)>0:
        for metric_name, metric_values in metric_dict.items():
            table.add_column(metric_name, metric_values)
    return table

class BaseExperiment(object):

    def __init__(self, model, optimizer, scheduler_iter, scheduler_epoch,
                 log_path, eval_every, check_every, monitoring_statistics, n_patient=10, eval_evaluator=None, test_evaluator=None):

        # Objects
        self.model = model
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch

        # Paths
        self.log_path = log_path
        self.check_path = os.path.join(log_path, 'check')

        # Intervals
        self.eval_every = eval_every
        self.check_every = check_every

        # Initialize
        self.current_epoch = 0
        self.patient = 0
        self.n_patient = n_patient
        self.train_metrics = {}
        self.eval_metrics = {}
        self.test_metrics = {}
        self.eval_epochs = []
        self.test_epochs = []

        self.monitoring_statistics = monitoring_statistics
        self.current_best_eval = None
        self.eval_evaluator = eval_evaluator
        self.test_evaluator = test_evaluator

    def train_fn(self, epoch):
        raise NotImplementedError()

    def eval_fn(self, epoch):
        raise NotImplementedError()

    def test_fn(self, epoch):
        raise NotImplementedError()

    def log_fn(self, epoch, train_dict, eval_dict, test_dict):
        raise NotImplementedError()

    def compare_current_best(self, eval_dict):
        if eval_dict is None:
            return False
        if self.current_best_eval is None:
            self.current_best_eval = np.mean([eval_dict[key] for key in self.monitoring_statistics])
            return True
        else:
            current_eval = np.mean([eval_dict[key] for key in self.monitoring_statistics])
            if current_eval < self.current_best_eval:
               self.current_best_eval = current_eval
               return True
            else:
                return False 
    
    def log_train_metrics(self, train_dict):
        if len(self.train_metrics)==0:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name].append(metric_value)

    def log_eval_metrics(self, eval_dict):
        if len(self.eval_metrics)==0:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name].append(metric_value)
    
    def log_test_metrics(self, test_dict):
        if len(self.test_metrics)==0:
            for metric_name, metric_value in test_dict.items():
                self.test_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in test_dict.items():
                self.test_metrics[metric_name].append(metric_value)

    def create_folders(self):

        # Create log folder
        os.makedirs(self.log_path)
        print("Storing logs in:", self.log_path)

        # Create check folder
        if self.check_every is not None:
            os.makedirs(self.check_path)
            print("Storing checkpoints in:", self.check_path)
            os.makedirs(os.path.join(self.log_path, 'eval'))
            os.makedirs(os.path.join(self.log_path, 'test'))

    def save_args(self, args):

        # Save args
        with open(os.path.join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args, f)

        # Save args table
        args_table = get_args_table(vars(args))
        with open(os.path.join(self.log_path,'args_table.txt'), "w") as f:
            f.write(str(args_table))

    def save_metrics(self):

        # Save metrics
        with open(os.path.join(self.log_path,'metrics_train.pickle'), 'wb') as f:
            pickle.dump(self.train_metrics, f)
        with open(os.path.join(self.log_path,'metrics_eval.pickle'), 'wb') as f:
            pickle.dump(self.eval_metrics, f)
        with open(os.path.join(self.log_path,'metrics_test.pickle'), 'wb') as f:
            pickle.dump(self.test_metrics, f)

        # Save metrics table
        metric_table = get_metric_table(self.train_metrics, epochs=list(range(1, self.current_epoch+2)))
        with open(os.path.join(self.log_path,'metrics_train.txt'), "w") as f:
            f.write(str(metric_table))
        metric_table = get_metric_table(self.eval_metrics, epochs=[e+1 for e in self.eval_epochs])
        with open(os.path.join(self.log_path,'metrics_eval.txt'), "w") as f:
            f.write(str(metric_table))
        metric_table = get_metric_table(self.test_metrics, epochs=[e+1 for e in self.test_epochs])
        with open(os.path.join(self.log_path,'metrics_test.txt'), "w") as f:
            f.write(str(metric_table))

    def checkpoint_save(self, name='checkpoint.pt'):
        checkpoint = {'current_epoch': self.current_epoch,
                      'train_metrics': self.train_metrics,
                      'eval_metrics': self.eval_metrics,
                      'test_metrics': self.test_metrics,
                      'eval_epochs': self.eval_epochs,
                      'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler_iter': self.scheduler_iter.state_dict() if self.scheduler_iter else None,
                      'scheduler_epoch': self.scheduler_epoch.state_dict() if self.scheduler_epoch else None}
        torch.save(checkpoint, os.path.join(self.check_path, name))

    def checkpoint_load(self, check_path, name='checkpoint.pt'):
        checkpoint = torch.load(os.path.join(check_path, name))
        self.current_epoch = checkpoint['current_epoch']
        self.train_metrics = checkpoint['train_metrics']
        self.eval_metrics = checkpoint['eval_metrics']
        self.test_metrics = checkpoint['test_metrics']
        self.eval_epochs = checkpoint['eval_epochs']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler_iter: self.scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
        if self.scheduler_epoch: self.scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])

    def run(self, epochs):

        for epoch in range(self.current_epoch, epochs):

            # Train
            train_dict = self.train_fn(epoch)
            self.log_train_metrics(train_dict)

            # Eval
            if (epoch+1) % self.eval_every == 0:
                eval_dict = self.eval_fn(epoch)
                self.log_eval_metrics(eval_dict)
                self.eval_epochs.append(epoch)
                # Test if eval score a better metrics
                if self.compare_current_best(eval_dict):
                    self.current_best_eval_dict = eval_dict
                    test_dict = self.test_fn(epoch)
                    self.log_test_metrics(test_dict)
                    self.test_epochs.append(epoch)
                    self.patient = 0
                else:
                    self.patient += 1
                
            else:
                eval_dict = None
                test_dict = None
            
            # Log
            self.save_metrics()
            self.log_fn(epoch, train_dict, eval_dict, test_dict)

            # Checkpoint
            self.current_epoch += 1
            if (epoch+1) % self.check_every == 0:
                self.checkpoint_save(f'checkpoint_{epoch}.pt')

            if self.patient > self.n_patient:
                return


class DataParallelDistribution(torch.nn.DataParallel):
    """
    A DataParallel wrapper for Distribution.
    To be used instead of nn.DataParallel for Distribution objects.
    """

    def log_prob(self, *args, **kwargs):
        return self.forward(*args, mode='log_prob', **kwargs)

    def sample(self, *args, **kwargs):
        return self.module.sample(*args, **kwargs)

    def sample_with_log_prob(self, *args, **kwargs):
        return self.module.sample_with_log_prob(*args, **kwargs)
