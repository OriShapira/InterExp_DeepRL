""" Main training code file for the interactive summarization model. """
import argparse
import json
import pickle as pkl
import os
from datetime import datetime
from itertools import cycle
import random
import numpy as np
from toolz.sandbox.core import unzip

from utils import get_gpu_memory_map, print_config, str2bool

'''
# For distributing jobs on the multiple GPUs of a server:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sorted_gpu_info = get_gpu_memory_map()
for gpu_id, (mem_left, util) in sorted_gpu_info:
    if mem_left >= 2000:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # args.cuda = f'cuda:{gpu_id}'
        print('use gpu:{} with {} MB left, util {}%'.format(gpu_id, mem_left, util))
        break
else:
    print(f'no gpu has memory left >=  MB, exiting...')
    exit()
'''

import torch
from torch import optim
from torch.utils.data import DataLoader

from model.rl import ActorCritic
from model.extract import PtrExtractSumm

from training import BasicTrainer
from rl import get_grad_fn
from rl import A2CPipeline
from utils import load_best_checkpoint
from dataset.batching import ArticleBatcher, data_loader_collate
from metric import compute_rouge_l, compute_rouge_n, compute_delta_rouge_l, compute_delta_rouge_n
from metric import compute_similarity_query_to_text
from dataset.data_manage import DatasetManager

# to enable consistent runs, use a constant seed value:
seed = 112
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def configure_net(ext_dir, database_base_path, beta, query_encode, cuda,
                  method_mode, importance_mode, diversity_mode):
    """ load pretrained sub-modules and build the actor-critic network"""

    # load the config info on the model
    ext_meta = json.load(open(os.path.join(ext_dir, 'meta.json')))

    # the base model can can either be the pre-trained single-doc summarization extractor
    # or the pre-trained generic MDS summarizer:
    assert ext_meta['net'] in ['ml_rnn_extractor', 'base-rl_mmr']

    # in case it is the extractor:
    if ext_meta['net'] == 'ml_rnn_extractor':
        # load pre-trained extractor net:
        ext_ckpt = load_best_checkpoint(ext_dir)
        ext_args = ext_meta['net_args']
        agent_vocab = pkl.load(open(os.path.join(ext_dir, 'vocab.pkl'), 'rb'))
        extractor = PtrExtractSumm(**ext_args)
        extractor.load_state_dict(ext_ckpt)
        # build RL agent
        agent = ActorCritic(extractor._sent_enc,
                            extractor._art_enc,
                            extractor._extractor,
                            ArticleBatcher(agent_vocab, cuda),
                            database_base_path, beta, query_encode,
                            method_mode, importance_mode, diversity_mode)
        net_args = {'extractor': json.load(open(os.path.join(ext_dir, 'meta.json')))}

    # in case it is the MDS summarizer:
    elif ext_meta['net'] == 'base-rl_mmr':
        # load pre-trained extractor net:
        ext_args = ext_meta['net_args']['extractor']['net_args']
        agent_vocab = pkl.load(open(os.path.join(ext_dir, 'agent_vocab.pkl'), 'rb'))
        extractor = PtrExtractSumm(**ext_args)
        # build RL agent
        agent = ActorCritic(extractor._sent_enc,
                            extractor._art_enc,
                            extractor._extractor,
                            ArticleBatcher(agent_vocab, cuda),
                            database_base_path, beta, query_encode,
                            method_mode, importance_mode, diversity_mode)
        # load pre-trained generic MDS summarizer net:
        ext_ckpt = load_best_checkpoint(ext_dir, reverse=True)
        agent.load_state_dict(ext_ckpt)
        net_args = {'summarizer': json.load(open(os.path.join(ext_dir, 'meta.json')))}

    if cuda:
        agent = agent.cuda()

    return agent, agent_vocab, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size,
                       discount_gamma, reward_summ, reward_query):
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer'] = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size'] = batch_size
    train_params['lr_decay'] = lr_decay
    train_params['gamma'] = discount_gamma
    train_params['reward_summ'] = reward_summ
    train_params['reward_query'] = reward_query

    return train_params

def get_reward_funcs(reward_summ_arg, reward_query_arg):

    def get_reward_func(reward_name):
        # if _norm, then the score should be normalized by length:
        normalize = '_norm' in reward_name

        # if _sent, then compute against a specified sentence instead of the whole ref summary:
        if '_sent' in reward_name:
            if reward_name.startswith('dr'):
                raise Exception("Reward cannot be on the sentence level when using the delta scoring.")
            sentence_level = True
        else:
            sentence_level = False

        # if _persent, then compute score for each sentence, ignoring preceding sentences:
        if '_persent' in reward_name:
            if reward_name.startswith('dr'):
                raise Exception("Reward cannot be on the sentence level when using the delta scoring.")
            ignore_preceding = True
        else:
            ignore_preceding = False

        # stemming and stopword removal
        stem = '_stem' in reward_name
        remove_stop = '_nostop' in reward_name

        if reward_name.startswith('dr'): # starts with 'dr', delta ROUGE. e.g. dr1f
            nextInd = 2
        else: # starts with 'r', just ROUGE. e.g. r1f
            nextInd = 1
        type = reward_name[nextInd] # l,1,2
        mode = reward_name[nextInd+1] # r,p,f

        # if starts with 'dr', delta ROUGE:
        if reward_name.startswith('dr'):
            if type == 'l': # delta ROUGE-L
                reward_fn = compute_delta_rouge_l(mode=mode, normalize=normalize, stem=stem, remove_stop=remove_stop)
            else: # delta ROUGE-N
                reward_fn = compute_delta_rouge_n(n=int(type), mode=mode, normalize=normalize,
                                                  stem=stem, remove_stop=remove_stop)
        elif reward_name.startswith('r'):
            if type == 'l': # ROUGE-L
                reward_fn = compute_rouge_l(mode=mode, normalize=normalize, sentence_level=sentence_level,
                                            stem=stem, remove_stop=remove_stop, ignore_preceding=ignore_preceding)
            else: # ROUGE-N
                reward_fn = compute_rouge_n(n=int(type), mode=mode, normalize=normalize, sentence_level=sentence_level,
                                            stem=stem, remove_stop=remove_stop, ignore_preceding=ignore_preceding)
        #else:
        #    raise('Error: Unknown arg for reward_summ. E.g. r1r, r1f, rlr, rlf, dr1r, r1r_norm, etc.')

        # query-similarity reward
        elif reward_name == 'lex':
            reward_fn = compute_similarity_query_to_text(useSemSim=False, useLexSim=True)
        elif reward_name == 'sem':
            reward_fn = compute_similarity_query_to_text(useSemSim=True, useLexSim=False)
        elif reward_name == 'lexsem':
            reward_fn = compute_similarity_query_to_text(useSemSim=True, useLexSim=True)
        else:
            raise ('Error: Unknown arg for reward_query. Expecting any of lex, sem, lexsem.')

        return reward_fn

    reward_summary_fn = get_reward_func(reward_summ_arg)
    reward_query_fn = get_reward_func(reward_query_arg)
    return reward_summary_fn, reward_query_fn


def build_batchers(batch_size, dataset_base_path):
    """
    Prepares the train/val/test batchers with which to read in the data.
    :param batch_size:
    :param dataset_base_path:
    :return:
    """

    loader = DataLoader(
        DatasetManager('train', dataset_base_path),
        batch_size=batch_size,
        shuffle=True, num_workers=0,#4,
        collate_fn=data_loader_collate
    )
    val_loader = DataLoader(
        DatasetManager('val', dataset_base_path),
        batch_size=batch_size,
        shuffle=False, num_workers=0,#4,
        collate_fn=data_loader_collate
    )
    test_loader = DataLoader(
        DatasetManager('test', dataset_base_path),
        batch_size=batch_size,
        shuffle=False, num_workers=0,#4,
        collate_fn=data_loader_collate
    )
    return cycle(loader), val_loader, test_loader


def train(args):
    # make net
    agent, agent_vocab, net_args = configure_net(args.ext_dir, args.data_dir,
                                                 args.beta, args.query_encoding_in_input, args.cuda,
                                                 args.method_mode, args.importance_mode, args.diversity_mode)

    # configure training setting
    train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch_size,
        args.gamma, args.reward_summ, args.reward_query
    )
    train_batcher, val_batcher, test_batcher = build_batchers(args.batch_size, args.data_dir)
    reward_summary_fn, reward_query_fn = get_reward_funcs(args.reward_summ, args.reward_query)

    # save configuration
    meta = {}
    if args.generation_mode == 'qsummary':
        meta['net'] = 'q-rl_mmr'
    elif args.generation_mode == 'summary':
        meta['net'] = 'base-rl_mmr'
    meta['net_args'] = net_args
    meta['train_params'] = train_params
    with open(os.path.join(args.model_out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    with open(os.path.join(args.model_out_dir, 'agent_vocab.pkl'), 'wb') as f:
        pkl.dump(agent_vocab, f)

    # prepare trainer
    grad_fn = get_grad_fn(agent, args.clip)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None

    pipeline = A2CPipeline(meta['net'], agent,
                           train_batcher, val_batcher, test_batcher,
                           optimizer, grad_fn,
                           reward_summary_fn, reward_query_fn, args.gamma, args.ignore_queries_summ_reward,
                           args.reward_query_every, args.max_num_steps, args.generation_mode, args.max_sent_len,
                           testing_summary_len=args.testing_summary_len)
    trainer = BasicTrainer(pipeline, args.model_out_dir,
                           args.ckpt_freq, args.patience, scheduler,
                           val_mode='score', args=args)

    print('Starting training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model options
    parser.add_argument('--ext_dir', action='store',
                        help='root of the extractor model', default='saved_model/extractive')
    parser.add_argument('--ckpt', type=int, action='store', default=None,
                        help='ckeckpoint used decode')
    parser.add_argument('--data_dir', action='store', default='./dataset/DUC',
                        help='directory of the database')
    parser.add_argument('--model_out_dir', action='store', default='./saved_model',
                        help='where to save the learned model')
    parser.add_argument('--generation_mode', action='store', default='qsummary',
                        help='should the model learn to generate generic summaries or query-biased summaries: summary, qsummary')
    parser.add_argument('--testing_summary_len', type=int, action='store', default=250,
                        help='the token-length of the summary to output during validation and testing')
    parser.add_argument('--max_num_steps', type=int, action='store', default=10,
                        help='how many sentences should be extracted during train/val/test out of the specified samples.')
    parser.add_argument('--max_sent_len', type=int, action='store', default=9999,
                        help='what is the max sentence token-length to use')
    # training options
    parser.add_argument('--reward_summ', action='store', default='r1r',
                        help='reward function for the summary: r1r (ROUGE-1 recall), r1f (ROUGE-1 F1), '
                             'rlr (ROUGE-L recall), rlf (ROUGE-L F1), dr1r (delta-ROUGE-1 recall), '
                             'dr1f (delta-ROUGE-1 F1), drlr, drlf. Add "_norm" at the end to also normalize'
                             'the score by the token-length of the evaluated text.'
                             'Add _stem for stemming and _nostop for removing stop words'
                             'Add _persent to reward each output sentence separately against the references'
                        )
    parser.add_argument('--reward_query', action='store', default='lexsem',
                        help='reward function for the query: lexsem (ROUGE based + w2v based), lex, sem')
    parser.add_argument('--ignore_queries_summ_reward', type=str2bool, default=True,
                        help='should queries be ignored on a batch where the summary reward is used')
    parser.add_argument('--reward_query_every', type=int, action='store', default=2,
                        help='every how many batches should the query reward be applied (0=never 1=always >1=alternating)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='1-weight for the query in the MMR score (1=noWeightForQuery, 0=fullWeightForQuery)')
    parser.add_argument('--query_encoding_in_input', type=str2bool, default=True,
                        help='should a query be encoded into the input')
    parser.add_argument('--method_mode', action='store', default='soft-attn',
                        help='mode for use of MMR: soft-attn provides MMR scores as soft attention for sentences, and no-mmr does not')
    parser.add_argument('--importance_mode', action='store', default='tfidf',
                        help='MMR importance function (tfidf or w2v)')
    parser.add_argument('--diversity_mode', action='store', default='tfidf',
                        help='MMR diversity function (tfidf or w2v)')
    parser.add_argument('--lr', type=float, action='store', default=5e-4, #5e-5
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, action='store', default=0,
                        help='weight_decay')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=5,
                        help='patience for learning rate decay')
    parser.add_argument('--gamma', type=float, action='store', default=0.99,
                        help='discount factor of RL')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch_size', type=int, action='store', default=8,
                        help='the training batch size')
    parser.add_argument('--ckpt_freq', type=int, action='store', default=100,
                        help='number of update steps for checkpoint and validation')
    parser.add_argument('--patience', type=int, action='store', default=30,
                        help='patience for early stopping (how many batches to continue after there is no improvement in validation.')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.model_out_dir = os.path.join('saved_model', current_time)

    if not os.path.exists(args.model_out_dir):
        os.makedirs(args.model_out_dir)

    # set the appropriate arguments for generic summarization generation model:
    if args.generation_mode == 'summary':
        args.ignore_queries_summ_reward = True # do not consider the queries in the data
        args.reward_query_every = 0 # only use the summary reward
        args.beta = 1.0 # do not consider the query in the MMR score

    print_config(args, outjsonpath=os.path.join(args.model_out_dir, 'config.json'))

    train(args)
