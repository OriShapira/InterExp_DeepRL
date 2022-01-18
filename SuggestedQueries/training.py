""" module providing basic training utilities"""
import os
from os.path import join
from time import time
from datetime import timedelta
import json
from tqdm import tqdm
import tensorboardX
from rl import A2CPipeline

class BasicTrainer(object):
    """ Basic trainer with minimal function and early stopping"""

    def __init__(self, pipeline, save_dir, ckpt_freq, patience,
                 scheduler=None, val_mode='loss', args=None):
        assert isinstance(pipeline, A2CPipeline)
        assert val_mode in ['loss', 'score']
        self._pipeline = pipeline # A2CPipeline object
        self._save_dir = save_dir
        self._logger = tensorboardX.SummaryWriter(join(save_dir, 'log'))
        os.makedirs(join(save_dir, 'ckpt'))

        self._ckpt_freq = ckpt_freq
        self._patience = patience
        self._sched = scheduler
        self._val_mode = val_mode

        self._step = 0
        self._running_loss = None
        # state vars for early stopping
        self._current_p = 0
        self._best_val = None
        self._best_val_step = None
        self.args = args

    def log(self, log_dict):
        loss = log_dict['mse_loss'] if 'mse_loss' in log_dict else log_dict['avg_reward']
        if self._running_loss is not None:
            self._running_loss = 0.99 * self._running_loss + 0.01 * loss
        else:
            self._running_loss = loss
        log_dict['running_loss'] = self._running_loss
        for key, value in log_dict.items():
            if isinstance(value, str):
                self._logger.add_text('{}_{}'.format(key, self._pipeline.name), value, self._step)
            else:
                self._logger.add_scalar('{}_{}'.format(key, self._pipeline.name), value, self._step)

    def validate(self, name='val'):
        val_log = self._pipeline.validate(save_dir=self._save_dir, name=name)
        for key, value in val_log.items():
            if isinstance(value, str):
                self._logger.add_text(f'{name}_{key}_{self._pipeline.name}', value, self._step)
            else:
                self._logger.add_scalar(f'{name}_{key}_{self._pipeline.name}', value, self._step)

        #if 'avg_reward' in val_log:
        #    val_metric = val_log['reward']
        #else:
        #    #val_metric = (val_log['mse_loss'] if self._val_mode == 'mse_loss'
        #    #              else val_log['score'])
        #    val_metric = val_log['mse_loss']

        return val_log #val_metric

    def checkpoint(self, saveModel=False):
        val_metric = self.validate(name='val')
        test_metric = self.validate(name='test')
        test_kp_metric = self.validate(name='test_kp')
        test_metric_all = {'test': test_metric, 'test_kp': test_kp_metric}
        # NB no saving model
        # self._pipeline.checkpoint(
        #     join(self._save_dir, 'ckpt'), self._step, val_metric)
        # if isinstance(self._sched, ReduceLROnPlateau):
        #     self._sched.step(val_metric)
        # else:
        #     self._sched.step()
        stop = self.check_stop(val_metric['reward'])
        # if the current validation shows an improvement (self._current_p == 0) or we need to stop, save the model:
        if self._current_p == 0 or saveModel or stop:
            self._pipeline.checkpoint(
                join(self._save_dir, 'ckpt'), self._step, val_metric['reward'])
            scoresFile = os.path.join(self._save_dir, f'test_scores_{self._step}_{test_metric["reward"]}.json')
            with open(scoresFile, 'w') as fOut:
                json.dump(test_metric_all, fOut, indent=2)
        return stop

    def check_stop(self, val_metric):
        if self._best_val is None:
            self._best_val = val_metric
            self._best_val_step = self._step
        elif ((val_metric < self._best_val and self._val_mode == 'loss')
              or (val_metric > self._best_val and self._val_mode == 'score')):
            self._current_p = 0
            self._best_val = val_metric
            self._best_val_step = self._step
        else:
            self._current_p += 1
            print(f'not outperforming best for {self._current_p} times', end=', ')
        print(f'current best val: {self._best_val}@{self._best_val_step}')# {self.args.path}')
        return self._current_p >= self._patience

    def train(self):
        try:
            start = time()
            for _ in tqdm(range(20000)): #10000)):
                log_dict = self._pipeline.train_step()
                self.log(log_dict)
                if self._step % 100 == 0:
                    print(log_dict)
                    # print(f"[Train-{self._step}]reward: {log_dict['reward']:.4f}, avg n_sent:{log_dict['avg_n_sent']}")
                if self._step % self._ckpt_freq == 0:
                    #saveModel = self._step % 500 == 0
                    stop = self.checkpoint(saveModel=True)
                    if stop:
                        break
                self._step += 1
            print('Training finished in ', timedelta(seconds=time() - start))
        finally:
            self._pipeline.terminate()