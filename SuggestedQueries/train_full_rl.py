""" full training """
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
from dataset_k.prepare_mds_kps import prepare_data_in_datadir

'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sorted_gpu_info = get_gpu_memory_map()
for gpu_id, (mem_left, util) in sorted_gpu_info:
    if mem_left >= 7000:
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

from model_k.rl import ActorCritic
from model_k.extract import PtrExtractSumm

from training import BasicTrainer
from rl import get_grad_fn
from rl import A2CPipeline
from utils import load_best_checkpoint
from dataset_k.batching import ArticleBatcher, data_loader_collate
from metric_k import compute_kp_overlap, compute_kp_tfidf, compute_kp_tf, compute_partial_phrase_overlap_score
from dataset_k.data_manage import DatasetManager


seed = 111
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def configure_net(ext_dir, database_base_path, beta, cuda,
                  method_mode, importance_mode, diversity_mode):
    """ load pretrained sub-modules and build the actor-critic network"""

    # load the config info on the model
    ext_meta = json.load(open(os.path.join(ext_dir, 'meta.json')))

    # the base model is the pre-trained SDS extractor:
    assert ext_meta['net'] == 'ml_rnn_extractor'

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
                        database_base_path, beta,
                        method_mode, importance_mode, diversity_mode)
    net_args = {'extractor': json.load(open(os.path.join(ext_dir, 'meta.json')))}

    if cuda:
        agent = agent.cuda()

    return agent, agent_vocab, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size, discount_gamma, reward_fn):
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer'] = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size'] = batch_size
    train_params['lr_decay'] = lr_decay
    train_params['gamma'] = discount_gamma
    train_params['reward_fn'] = reward_fn

    return train_params

def get_reward_func(reward_fn_arg, stem, remove_stop, weight_initial, weight_preceding):

    if reward_fn_arg == 'overlap':
        reward_fn = compute_kp_overlap(stem=stem, remove_stop=remove_stop, weight_initial=weight_initial,
                                       weight_preceding=weight_preceding)
    elif reward_fn_arg == 'ppf':
        reward_fn = compute_partial_phrase_overlap_score(stem=stem, remove_stop=remove_stop,
                                                         mmr_lambda=weight_initial, redundancy_alpha=weight_preceding)
    elif reward_fn_arg == 'tfidf':
        reward_fn = compute_kp_tfidf(stem=stem, remove_stop=remove_stop, weight_initial=weight_initial,
                                     weight_preceding=weight_preceding)
    elif reward_fn_arg == 'tf':
        reward_fn = compute_kp_tf(stem=stem, remove_stop=remove_stop, weight_initial=weight_initial,
                                  weight_preceding=weight_preceding)
    else:
        raise Exception('Reward function not supported: ' + reward_fn_arg)

    return reward_fn


def build_batchers(batch_size, dataset_base_path, input_mode, phrasing_method, max_steps_phrases, max_phrase_len):
    """
    Prepares the train/val/test batchers with which to read in the data.
    :param batch_size:
    :param dataset_base_path:
    :param input_mode: '' or 'noduplicates' or 'singledoc' or 'noduplicates_singledoc' - manipulation on the input phrases
    :param phrasing_method: method of preparing the candidate keyphrases ('nounchunks', 'posregex')
    :param max_steps: relevant for keyphrase generation only - how many should be generated
    :return:
    """

    if len(input_mode) > 0:
        modeParts = input_mode.split('_')
        if len(modeParts) == 1 and modeParts[1] == 'noduplicates':
            input_mode_id = DatasetManager.INPUT_MODE_PHRASES_NODUPLICATES
        elif len(modeParts) == 1 and modeParts[1] == 'singledoc':
            input_mode_id = DatasetManager.INPUT_MODE_PHRASES_SINGLEDOC
        elif len(modeParts) == 2 and 'noduplicates' in modeParts and 'singledoc' in modeParts:
            input_mode_id = DatasetManager.INPUT_MODE_PHRASES_SINGLEDOC_NODUPLICATES
        else:
            raise Exception(f'Error: input mode not supported: {input_mode}')
    else:
        input_mode_id = DatasetManager.INPUT_MODE_PHRASES

    loader = DataLoader(
        DatasetManager('train', dataset_base_path, input_mode_id, phrasing_method, num_output_phrases=max_steps_phrases,
                       max_phrase_len=max_phrase_len),
        batch_size=batch_size,
        shuffle=True, num_workers=0,#4,
        collate_fn=data_loader_collate
    )
    val_loader = DataLoader(
        DatasetManager('val', dataset_base_path, input_mode_id, phrasing_method, num_output_phrases=max_steps_phrases,
                       max_phrase_len=max_phrase_len),
        batch_size=batch_size,
        shuffle=False, num_workers=0,#4,
        collate_fn=data_loader_collate
    )
    test_loader = DataLoader(
        DatasetManager('test', dataset_base_path, input_mode_id, phrasing_method, num_output_phrases=max_steps_phrases,
                       max_phrase_len=max_phrase_len),
        batch_size=batch_size,
        shuffle=False, num_workers=0,#4,
        collate_fn=data_loader_collate
    )
    test_kp_loader = DataLoader(
        DatasetManager('test_kp', dataset_base_path, input_mode_id, phrasing_method, num_output_phrases=max_steps_phrases,
                       max_phrase_len=max_phrase_len),
        batch_size=batch_size,
        shuffle=False, num_workers=0,  # 4,
        collate_fn=data_loader_collate
    )
    return cycle(loader), val_loader, test_loader, test_kp_loader


def train(args):
    # make net
    agent, agent_vocab, net_args = configure_net(args.ext_dir, args.data_dir,
                                                 args.beta, args.cuda,
                                                 args.method_mode, args.importance_mode, args.diversity_mode)

    # configure training setting
    train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch_size, args.gamma, args.reward_func)
    train_batcher, val_batcher, test_batcher, test_kp_batcher = \
        build_batchers(args.batch_size, args.data_dir, args.input_mode,
                       args.phrasing_method, args.max_num_steps, args.max_phrase_len)

    reward_fn = get_reward_func(args.reward_func, args.stem, args.remove_stop, args.weight_initial,
                                args.weight_preceding)

    # save configuration
    meta = {}
    meta['net'] = 'kp-rl_mmr'
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

    pipeline = A2CPipeline(meta['net'], agent, train_batcher, val_batcher, test_batcher, test_kp_batcher,
                           optimizer, grad_fn, reward_fn, args.gamma, args.max_num_steps, args.summexp_model_dir)
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
    parser.add_argument('--data_dir', action='store', default='./dataset_k/DUC',
                        help='directory of the database')
    parser.add_argument('--model_out_dir', action='store', default='./saved_model',
                        help='where to save the learned model')
    parser.add_argument('--phrasing_method', action='store', default='posregex',
                        help='How to prepare keyphrases: nounchunks, posregex')
    parser.add_argument('--input_mode', action='store', default='',
                        help='Any manipulation on the keyphrases: <>, <singledoc>')#, <noduplicates>, <noduplicates_singledoc>')
    parser.add_argument('--max_num_steps', type=int, action='store', default=20,
                        help='How many keyphrases should be extracted during train/val/test.')
    parser.add_argument('--max_phrase_len', type=int, action='store', default=999,
                        help='The max phrase token-length to use.')
    # training options
    parser.add_argument('--reward_func', action='store', default='overlap',
                        help='reward function for the summary: overlap or tfidf or tf')
    parser.add_argument('--weight_initial', type=float, action='store', default=0.5,
                        help='the weight of the KP match in the initial summary on the reward function')
    parser.add_argument('--weight_preceding', type=float, action='store', default=0.5,
                        help='the weight of the KP match in the preceding extracted KPs on the reward function')
    parser.add_argument('--stem', type=str2bool, default=True,
                        help='should text and KPs be stemmed when matched')
    parser.add_argument('--remove_stop', type=str2bool, default=True,
                        help='should stop words be removed from text and KPs when matched')
    parser.add_argument('--method_mode', action='store', default='soft-attn',
                        help='mode for use of MMR: soft-attn provides MMR scors as soft attention for sentences, and no-mmr does not')
    parser.add_argument('--importance_mode', action='store', default='tfidf',
                        help='MMR importance function (tfidf)')
    parser.add_argument('--diversity_mode', action='store', default='tfidf',
                        help='MMR diversity function (tfidf)')
    parser.add_argument('--beta', type=float, action='store', default=0.5,
                        help='the weight of importance (vs diversity) in the MMR component')
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
                        help='patience for early stopping')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--summexp_model_dir', action='store', default='',
                        help='A RLMMR_Q model to use for generating summary expansions during training. '
                             'Empty if not training with "initial summaries".')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.model_out_dir = os.path.join('saved_model', f'kp_{current_time}')

    if not os.path.exists(args.model_out_dir):
        os.makedirs(args.model_out_dir)

    print_config(args, outjsonpath=os.path.join(args.model_out_dir, 'config.json'))

    # Create a multi-doc keyphrase list set with the DUC 2001 KP dataset:
    prepare_data_in_datadir(args.data_dir, 'test_kp')

    train(args)
