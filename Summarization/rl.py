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

from metric import compute_rouge_l, compute_rouge_n, compute_similarity_query_to_text
from dataset.utils import prepare_data_for_model_input
from decoding import decode_full_dataset



def a2c_validate(agent, loader, save_dir, num_train_batches_until_now, split_name,
                 max_num_steps=-1, testing_summary_len=250, reward_type='auc', max_sent_len=999):
    """
    The validation functionality is somewhat different from the training objective. Here
    we just generate a full length summary and evaluate it against the full reference summary.
    This gives us a feel of how well the system is at generating a generic summary.
    The downside of this is that we have no way of validating the query-biased extraction.
    We also show the AUC of the incrementally aggregating summary to see how fast important information
    is presented. Again, this does not measure the query-responsiveness at all.
    :param agent: The ActorCritic agent from which to get the summary
    :param loader: the batch loader
    :param save_dir: where to save the output summaries
    :param num_train_batches_until_now: just for printing out at what step we are in training
    :param split_name: split name ('test' or 'val')
    :param max_num_steps: the max number of sentences to use in the sample (if -1, uses all queries in the sample datum)
    :param testing_summary_len: summary length to generate
    :param reward_type: 'auc' or 'r1f' or 'r1r' - what score to go by for the validation score
    :return: reward in a dictionary (key is 'reward')
    """
    agent.eval()
    start = time()
    print('start running validation...', end='')
    base_save_dirpath = join(save_dir, 'dec_all', f'{split_name}-{str(num_train_batches_until_now-1)}')

    full_dataset_scores, avg_num_sents_in_summ, avg_num_tokens_in_summ = \
        decode_full_dataset(agent, loader, base_save_dirpath, split_name, max_num_steps=max_num_steps,
                            testing_summary_len=testing_summary_len, max_sent_len=max_sent_len)

    print(f'finished in {timedelta(seconds=int(time() - start))}! '
          f'AVG_AUC_NORM: {full_dataset_scores["auc_norm"]:.4f}, '
          f'R1_F1: {full_dataset_scores["R1"]["f1"]:.4f}, '
          f'R1_Recall: {full_dataset_scores["R1"]["recall"]:.4f}, '
          f'avg # tokens: {avg_num_tokens_in_summ:.2f}, '
          f'avg # sents: {avg_num_sents_in_summ:.2f}')

    metric = {}
    metric['R1_F1'] = full_dataset_scores["R1"]["f1"]
    metric['R1_Recall'] = full_dataset_scores["R1"]["recall"]
    metric['R2_F1'] = full_dataset_scores["R2"]["f1"]
    metric['R2_Recall'] = full_dataset_scores["R2"]["recall"]
    metric['RL_F1'] = full_dataset_scores["RL"]["f1"]
    metric['RL_Recall'] = full_dataset_scores["RL"]["recall"]
    metric['RSU4_F1'] = full_dataset_scores["RSU4"]["f1"]
    metric['RSU4_Recall'] = full_dataset_scores["RSU4"]["recall"]
    metric['summ_len_full'] = avg_num_tokens_in_summ
    metric['num_sents_full'] = avg_num_sents_in_summ

    if reward_type == 'auc':
        metric['reward'] = full_dataset_scores["auc_norm"]
    elif reward_type == 'r1f':
        metric['reward'] = full_dataset_scores["R1"]["f1"]
    elif reward_type == 'r1r':
        metric['reward'] = full_dataset_scores["R1"]["recall"]
    else:
        metric['reward'] = full_dataset_scores["auc_norm"]

    return metric


def a2c_train_step(agent, loader, opt, grad_fn, compute_reward_func=None, discount_gamma=0.99,
                   ignore_queries_summ_reward=False, max_num_steps=-1, max_sent_len=999):
    opt.zero_grad()
    batch_sent_indices = [] # list of list of indices for this batch (one per summary)
    batch_probs = [] # list of probabilities object for all sentences (in order of the sentence indices) for each summary in the batch - from the actor
    batch_baselines = [] # list of baseline scores for each summary in the batch (for subtracting from the reward in the A2C model ("advantage")) this is the critic value
    batch_summ_sents = [] # list of list of sentence strings (one summary per batch instance)
    batch_queries = [] # list of list of queries (list of queries per batch instance)
    batch_num_sents_in_summ = [] # list of the summary lengths (num sentences) one per summary in the batch
    batch_initialSumms_sents = [] # list of list of sentences (list per instance in the batch)
    isFullSummaryReward = (compute_reward_func.__name__ == compute_reward_summary.__name__) # are we using the summary rewrad now?

    docs_batch, refs_batch, ids_batch, topicIdx_batch, initialsumm_batch, queriesInfo_batch = next(loader)
    for topic_docs, datum_id, topic_id, initialsumm_indices, queriesInfo in \
            zip(docs_batch, ids_batch, topicIdx_batch, initialsumm_batch, queriesInfo_batch):

        topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted, queriesList = \
            prepare_data_for_model_input(topic_docs, initialsumm_indices, queriesInfo,
                                         ignore_queries=(isFullSummaryReward and ignore_queries_summ_reward),
                                         max_num_steps=max_num_steps)
        queriesStrList = [qStr for qStr, qGroupIdx in queriesList]

        # get the predicted summary:
        (chosen_indices, distributions), critic_scores = \
            agent(topic_docs_filtered, datum_name=topic_id, split='train', initial_summ=initial_summ_indices_adjusted,
                  queries_list=queriesStrList, max_sent_len=max_sent_len) #, num_sentences_to_add=num_sents_needed)
        batch_num_sents_in_summ.append(len(chosen_indices))
        batch_baselines.append(critic_scores)
        batch_sent_indices.append(chosen_indices)
        batch_probs.append(distributions)
        summ_sents = [topic_docs_filtered_flat[idx.item()] for idx in chosen_indices]
        initialSumm_sents = [topic_docs_filtered_flat[idx] for idx in initial_summ_indices_adjusted]
        batch_summ_sents.append(summ_sents)
        batch_queries.append(queriesList)
        batch_initialSumms_sents.append(initialSumm_sents)

    avg_reward, avg_advantage, mse_loss, batch_num_rewards = \
        a2c_train_step_with_reward(batch_sent_indices, refs_batch, batch_summ_sents,
                                   batch_initialSumms_sents, batch_baselines, batch_probs, batch_queries,
                                   compute_reward_func=compute_reward_func, discount_gamma=discount_gamma)
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['avg_reward'] = avg_reward
    log_dict['avg_advantage'] = avg_advantage
    log_dict['mse_loss'] = mse_loss.item()
    log_dict['avg_num_sent'] = np.mean(batch_num_rewards)
    log_dict['reward_func'] = compute_reward_func.__name__
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict

def a2c_train_step_with_reward(summ_indices_batch, refsumms_batch, summ_sents_batch, batch_initialSumms_sents,
                               baselines_batch, probabilities_batch, queries_batch,
                               compute_reward_func=None, discount_gamma=0.99):
    discounted_rewards_for_batch = []
    avg_reward = 0 # just for logging
    num_rewards = []
    for sampleIdx, (summ_sents, refs_sents, queries_list, initialSumm_sents) in \
            enumerate(zip(summ_sents_batch, refsumms_batch, queries_batch, batch_initialSumms_sents)):

        # get the rewards for this summary output:
        # either compute_reward_summary or compute_reward_query
        # NOTE: the returned list of rewards may be shortened, so that it is no longer the length
        # of summ_sents. Therefore we take care to shorten the needed relevant vectors in the below if statement.
        rewards_for_summary, summary_overall_reward = compute_reward_func(summ_sents, refs_sents,
                                                                          queries_list, initialSumm_sents)
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
        if len(rewards_for_summary) != len(summ_sents):
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
def compute_reward_summary(summ_sents, refs_sents, queries_list, initialSumm_sents,
                           reward_fn=compute_rouge_n(n=1, mode='r')):
    """

    :param summ_sents:
    :param refs_sents:
    :param queries_list: list of tuples (queryStr, queryGroupIdx)
    :param initialSumm_sents:
    :param reward_fn:
    :return:
    """

    # map summary sentence index to reference summary index according to the queryInfo (in case needed):
    summSentToRefSent = {}
    curSentIdx = 0
    for qIdx, qInfo in enumerate(queries_list):
        for i in range(qInfo[1]):
            summSentToRefSent[curSentIdx + i] = qIdx
        curSentIdx += qInfo[1]

    initialSummTokens = list(concat(initialSumm_sents))
    # get the reward per summary sentence:
    rewards_for_summary = []
    for j in range(len(summ_sents)):
        r = reward_fn(summ_sents[j], refs_sents,
                      preceding_text=list(concat(initialSummTokens+summ_sents[:j])),
                      ref_sent_idx=(0, summSentToRefSent[j]))  # ref_sent_idx is the (refSumIdx, sentIdx) of the current query, in case needed
        if r == None:
            break # if we've reached a bad reward, then stop rewarding at this point
        rewards_for_summary.append(r)

    # get the accumulating token-length of the summary (for computing the average reward):
    accum_lens = [0]
    accum_scores = [0.]
    for s, r in zip(summ_sents, rewards_for_summary):
        accum_lens.append(accum_lens[-1] + len(s))  # add on the lengths sentence by sentence
        accum_scores.append(accum_scores[-1] + r)

    if reward_fn.__name__ == compute_rouge_n.__name__:
        # for logging, the reported reward for this summary is the average token-length-normalized reward:
        summary_overall_reward = np.mean([r / l for r, l in zip(rewards_for_summary, accum_lens[1:])])
    else:
        # for the delta-ROUGE reward function, the overall reward is the AUC of the accumulating summary scores:
        summary_overall_reward = auc(accum_lens, accum_scores)

    return rewards_for_summary, summary_overall_reward


@curry
def compute_reward_query(summ_sents, ref_sents, queries_list, initialSumm_sents,
                         reward_fn=compute_similarity_query_to_text(useSemSim=True, useLexSim=True)):
    """

    :param summ_sents:
    :param ref_sents:
    :param queries_list: list of tuples (queryStr, queryGroupIdx)
    :param initialSumm_sents:
    :param reward_fn:
    :return:
    """
    # Reward is the similarity of a sentence to the query
    # (last item in the list is the "stop" signal sentence)
    #rewards_for_summary = [reward_fn(query_str, summ_sents[j]) for j in range(len(summ_sents) - 1)]
    rewards_for_summary = [reward_fn(word_tokenize(queryInfo[0]), summ_sent) for summ_sent, queryInfo in zip(summ_sents, queries_list)]

    # for logging, the reported reward for this summary is the average reward:
    summary_overall_reward = np.mean(rewards_for_summary)

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
                 train_batcher, val_batcher, test_batcher,
                 optim, grad_fn,
                 reward_summary_fn, reward_query_fn, discounting_gamma, ignore_queries_summ_reward,
                 reward_query_every, max_num_steps, generation_mode, max_sent_len, testing_summary_len=250):
        self.name = name
        self._net = net # ActorCritic object
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._test_batcher = test_batcher
        self._opt = optim
        self._grad_fn = grad_fn
        self._reward_summary_fn = reward_summary_fn
        self._reward_query_fn = reward_query_fn
        self._discount_gamma = discounting_gamma
        self._testing_summary_len = testing_summary_len
        self._ignore_queries_summ_reward = ignore_queries_summ_reward
        self._reward_query_every = reward_query_every
        self._max_num_steps = max_num_steps
        self._max_sent_len = max_sent_len
        self._validate_score_type = 'auc' if generation_mode == 'qsummary' else 'r1f'

        self.train_batch_count = 0 # to keep track of what reward to use (alternates)

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # set the reward functionality to use for training, alternating from one batch to the next
        # according to the given argument value:
        if (self._reward_query_every > 0) and ((self.train_batch_count + 1) % self._reward_query_every == 0):
            compute_reward_func = compute_reward_query(reward_fn=self._reward_query_fn)
        else:
            compute_reward_func = compute_reward_summary(reward_fn=self._reward_summary_fn)

        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(self._net, self._train_batcher, self._opt, self._grad_fn,
                                  compute_reward_func=compute_reward_func,
                                  ignore_queries_summ_reward=self._ignore_queries_summ_reward,
                                  max_num_steps=self._max_num_steps, max_sent_len=self._max_sent_len)

        self.train_batch_count += 1

        return log_dict

    def validate(self, save_dir, name):
        if name == 'val':
            return a2c_validate(self._net, self._val_batcher, save_dir, self.train_batch_count, name,
                                max_num_steps=self._max_num_steps, testing_summary_len=self._testing_summary_len,
                                reward_type=self._validate_score_type, max_sent_len=self._max_sent_len)
        if name == 'test':
            return a2c_validate(self._net, self._test_batcher, save_dir, self.train_batch_count, name,
                                max_num_steps=self._max_num_steps, testing_summary_len=self._testing_summary_len,
                                reward_type=self._validate_score_type, max_sent_len=self._max_sent_len)

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