""" RL training utilities"""
import math
import pickle
from time import time
from datetime import timedelta
import os
from os.path import join
from cytoolz import concat, curry
from sklearn.metrics import auc
import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_
from nltk.tokenize import word_tokenize

from dataset_k.utils import prepare_data_for_model_input
from decoding import decode_full_dataset



def a2c_validate(agent, loader, save_dir, num_train_batches_until_now, split_name, num_steps):#, reward_type='auc'):
    """
    Generate the list of KPs and evaluate it againt the full reference summary.
    :param agent: The ActorCritic agent from which to get the summary
    :param loader: the batch loader
    :param save_dir: where to save the output summaries
    :param num_train_batches_until_now: just for printing out at what step we are in training
    :param split_name: split name ('test' or 'val' or 'test_kp')
    :param num_steps: the number of kps to output
    #:param reward_type: 'auc' or 'r1f' or 'r1r' - what score to go by for the validation score
    :return: reward in a dictionary (key is 'reward')
    """
    agent.eval()
    start = time()
    print('start running validation...', end='')
    base_save_dirpath = join(save_dir, 'dec_all', f'{split_name}-{str(num_train_batches_until_now-1)}')

    #full_dataset_scores, avg_num_kps_in_summ, avg_num_tokens_in_summ = \
    full_dataset_scores = decode_full_dataset(agent, loader, base_save_dirpath, split_name, num_steps)

    message = f'finished in {timedelta(seconds=int(time() - start))}! '
    message += ', '.join([f'{metric}: {full_dataset_scores[metric]:.4f}' for metric in full_dataset_scores])
    message += f'avg # tokens noInitial: {full_dataset_scores["kps_tokens_len_mean_noInitial"]:.2f}, '
    message += f'avg # KPs withInitial: {full_dataset_scores["num_kps_mean_full"]:.2f}'
    #message += f'avg # tokens: {avg_num_tokens_in_summ:.2f}, '
    #message += f'avg # KPs: {avg_num_kps_in_summ:.2f}'
    print(message)

    #print(f'finished in {timedelta(seconds=int(time() - start))}! '
    #      f'AUCNORM_OVERLAP: {full_dataset_scores["auc_norm_overlap"]:.4f}, '
    #      f'AUCNORM_TFIDF: {full_dataset_scores["auc_norm_tfidf"]:.4f}, '
    #      f'SCORE_OVERLAP: {full_dataset_scores["mean_kp_score_overlap"]:.4f}, '
    #      f'SCORE_TFIDF: {full_dataset_scores["mean_kp_score_tfidf"]:.4f}, '
    #      f'avg # tokens: {avg_num_tokens_in_summ:.2f}, '
    #      f'avg # KPs: {avg_num_kps_in_summ:.2f}')

    metric = {}
    metric.update(full_dataset_scores)
    #metric['summ_len_full'] = avg_num_tokens_in_summ
    #metric['num_KPs_full'] = avg_num_kps_in_summ
    metric['reward'] = full_dataset_scores["auc_norm_ppf_full"] #auc_norm_overlap_full"]

    #metric['auc_norm_overlap'] = full_dataset_scores['auc_norm_overlap']
    #metric['mean_kp_score_overlap'] = full_dataset_scores['mean_kp_score_overlap']
    #metric['auc_norm_tfidf'] = full_dataset_scores['auc_norm_tfidf']
    #metric['mean_kp_score_tfidf'] = full_dataset_scores['mean_kp_score_tfidf']
    #metric['summ_len_full'] = avg_num_tokens_in_summ
    #metric['num_KPs_full'] = avg_num_kps_in_summ
    #metric['reward'] = full_dataset_scores["auc_norm_overlap"]

    return metric


def a2c_train_step(agent, loader, opt, grad_fn, num_steps, compute_reward_func=None, discount_gamma=0.99):
    opt.zero_grad()
    batch_kp_indices = [] # list of list of indices for this batch (one per summary)
    batch_probs = [] # list of probabilities object for all KPs (in order of the kp indices) for each summary in the batch - from the actor
    batch_baselines = [] # list of baseline scores for each summary in the batch (for subtracting from the reward in the A2C model ("advantage")) this is the critic value
    batch_summ_kps = [] # list of list of KP strings (one summary per batch instance)
    batch_num_kps_in_summ = [] # list of the summary lengths (num KPs) one per summary in the batch
    batch_initialSumms_kps = [] # list of list of strings (list per instance in the batch)

    docs_batch, refs_batch, ids_batch, topicIdx_batch, initialsumm_batch, refKps_batch = next(loader)
    for topic_docs, datum_id, topic_id, initialsumm_indices, _ in \
            zip(docs_batch, ids_batch, topicIdx_batch, initialsumm_batch, refKps_batch):

        topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted = \
            prepare_data_for_model_input(topic_docs, initialsumm_indices)

        # get the predicted summary:
        (chosen_indices, chosen_scores, distributions), critic_scores = \
            agent(topic_docs_filtered, num_steps, datum_name=topic_id, split='train', initial_summ=initial_summ_indices_adjusted)
        batch_num_kps_in_summ.append(len(chosen_indices))
        batch_baselines.append(critic_scores)
        batch_kp_indices.append(chosen_indices)
        batch_probs.append(distributions)
        summ_kps = [topic_docs_filtered_flat[idx.item()] for idx in chosen_indices]
        initialSumm_kps = [topic_docs_filtered_flat[idx] for idx in initial_summ_indices_adjusted]
        batch_summ_kps.append(summ_kps)
        batch_initialSumms_kps.append(initialSumm_kps)

    avg_reward, avg_advantage, mse_loss, batch_num_rewards = \
        a2c_train_step_with_reward(batch_kp_indices, refs_batch, batch_summ_kps,
                                   batch_initialSumms_kps, batch_baselines, batch_probs,
                                   compute_reward_func=compute_reward_func, discount_gamma=discount_gamma)
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['avg_reward'] = avg_reward
    log_dict['avg_advantage'] = avg_advantage
    log_dict['mse_loss'] = mse_loss.item()
    log_dict['avg_num_kp'] = np.mean(batch_num_rewards)
    log_dict['reward_func'] = compute_reward_func.__name__
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict

def a2c_train_step_with_reward(summ_indices_batch, refsumms_batch, summ_kps_batch, batch_initialSumms_kps,
                               baselines_batch, probabilities_batch,
                               compute_reward_func=None, discount_gamma=0.99):
    discounted_rewards_for_batch = []
    avg_reward = 0 # just for logging
    num_rewards = []
    for sampleIdx, (summ_kps, refs_sents, initialSumm_kps) in \
            enumerate(zip(summ_kps_batch, refsumms_batch, batch_initialSumms_kps)):

        # NOTE: the returned list of rewards may be shortened, so that it is no longer the length
        # of summ_sents. Therefore we take care to shorten the needed relevant vectors in the below if statement.
        rewards_for_summary, summary_overall_reward = compute_reward_func(summ_kps, refs_sents, initialSumm_kps)
        avg_reward += summary_overall_reward

        # re-compute rewards as discounted rewards:
        R = 0
        discounted_rewards = []
        for r in rewards_for_summary[::-1]:
            R = r + discount_gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards_for_batch += discounted_rewards
        # in case the number of rewards is less than the number of sentences, shorten the needed vectors:
        num_rewards.append(len(rewards_for_summary))
        if len(rewards_for_summary) != len(summ_kps):
            summ_indices_batch[sampleIdx] = summ_indices_batch[sampleIdx][:len(rewards_for_summary)]
            probabilities_batch[sampleIdx] = probabilities_batch[sampleIdx][:len(rewards_for_summary)]
            baselines_batch[sampleIdx] = baselines_batch[sampleIdx][:len(rewards_for_summary)]

    indices = list(concat(summ_indices_batch))  # all indices of the batch in one list
    probs = list(concat(probabilities_batch))  # all probability objects of the batch in one list
    baselines = list(concat(baselines_batch))  # all baseline scores of the batch in one list
    # standardize rewards
    reward = torch.Tensor(discounted_rewards_for_batch).to(baselines[0].device)
    reward = (reward - reward.mean()) / (reward.std() + float(np.finfo(np.float32).eps))
    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    for action, p, r, b in zip(indices, probs, reward, baseline):
        # action is the index of the sentence to use in this iteration
        # p is the categorical probability distribution for all sentences in this iteration
        # r is the reward for this iteration
        # b is the baseline score from the Critic (CriticScorer) for this iteration
        advantage = r - b
        avg_advantage += advantage
        # losses.append(-p.log_prob(action) * advantage)
        losses.append(-p.log_prob(action) * (advantage / len(indices)))  # divide by T*B
    critic_loss = F.mse_loss(baseline, reward).reshape(1)
    # backprop and update
    autograd.backward(
        [critic_loss] + losses,
        [torch.ones(1).to(critic_loss.device)] * (1 + len(losses))
    )

    avg_reward /= len(summ_indices_batch)  # average reward over the batch
    avg_advantage = avg_advantage.item() / len(indices)  # average advantage over the batch

    return avg_reward, avg_advantage, critic_loss, num_rewards


@curry
def compute_reward_summary(summ_kps, refs_sents, initialSumm_kps, reward_fn):
    """
    :param summ_kps:
    :param refs_sents:
    :param initialSumm_kps:
    :param reward_fn:
    :return:
    """

    # compute the score for each KP in the list against the reference summaries:
    num_kps = len(summ_kps)
    rewards_for_summary = []
    for j in range(num_kps):
        r = reward_fn(summ_kps[j], refs_sents, initialSumm_kps, summ_kps[:j])
        if r == None:
            break # if we've reached a bad reward, then stop rewarding at this point
        rewards_for_summary.append(r)

    # get the accumulating token-length and score of the summary:
    accum_lens = [0]
    accum_scores = [0.]
    for kp, r in zip(summ_kps, rewards_for_summary):
        accum_lens.append(accum_lens[-1] + len(kp))  # add on the lengths kp by kp
        accum_scores.append(accum_scores[-1] + r)

    # the overall reward is the AUC of the keyphrases (we want better KPs earlier),
    # normalized by num KPs for comparison
    summary_overall_reward = auc(list(range(num_kps + 1)), accum_scores) / num_kps

    #if reward_fn.__name__ == compute_rouge_n.__name__:
    #    # for logging, the reported reward for this summary is the average token-length-normalized reward:
    #    summary_overall_reward = np.mean([r / l for r, l in zip(rewards_for_summary, accum_lens[1:])])
    #else:
    #    # for the delta-ROUGE reward function, the overall reward is the AUC of the accumulating summary scores:
    #    summary_overall_reward = auc(accum_lens, accum_scores)

    return rewards_for_summary, summary_overall_reward


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]

    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1 / 2)
            grad_log['grad_norm' + n] = tot_grad.item()
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        # grad_norm = grad_norm.item()
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log

    return f


class A2CPipeline(object):
    def __init__(self, name, net,
                 train_batcher, val_batcher, test_batcher, test_kp_batcher,
                 optim, grad_fn,
                 reward_summary_fn, discounting_gamma, max_num_steps, summexp_model_dir):
        self.name = name
        self._net = net # ActorCritic object
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._test_batcher = test_batcher
        self._test_kp_batcher = test_kp_batcher
        self._opt = optim
        self._grad_fn = grad_fn
        self._discount_gamma = discounting_gamma
        self._max_num_steps = max_num_steps
        #self._validate_score_type = 'auc' if generation_mode == 'qsummary' else 'r1f'
        self._compute_reward_func = compute_reward_summary(reward_fn=reward_summary_fn)
        self._summexp_model_dir = summexp_model_dir
        self.train_batch_count = 0

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(self._net, self._train_batcher, self._opt, self._grad_fn, self._max_num_steps,
                                  compute_reward_func=self._compute_reward_func)
        self.train_batch_count += 1
        return log_dict

    def validate(self, save_dir, name):
        if name == 'val':
            return a2c_validate(self._net, self._val_batcher, save_dir, self.train_batch_count, name,
                                self._max_num_steps)
        elif name == 'test':
            return a2c_validate(self._net, self._test_batcher, save_dir, self.train_batch_count, name,
                                self._max_num_steps)
        elif name == 'test_kp':
            return a2c_validate(self._net, self._test_kp_batcher, save_dir, self.train_batch_count, name,
                                self._max_num_steps)

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        pass  # No extra processs so do nothing