""" utility functions"""
import re
import os
import json
from os.path import basename
import subprocess
import argparse

import gensim
import torch
from torch import nn


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


PAD = 0
UNK = 1
START = 2
END = 3


def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
    return word2id


def make_embedding(id2word, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  # word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == END:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_info = [eval(x) for x in result.strip().split('\n')]
    gpu_info = dict(zip(range(len(gpu_info)), gpu_info))
    sorted_gpu_info = sorted(gpu_info.items(), key=lambda kv: kv[1][0], reverse=True)
    sorted_gpu_info = sorted(sorted_gpu_info, key=lambda kv: kv[1][1])
    print(f'gpu_id, (mem_left, util): {sorted_gpu_info}')
    return sorted_gpu_info



def print_config(config, logger=None, outjsonpath=None):
    config = vars(config)
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    if not logger:
        print("\n" + info + "\n")
    else:
        logger.info("\n" + info + "\n")
    if outjsonpath:
        with open(outjsonpath, 'w') as fOut:
            json.dump(config, fOut, indent=2)



def load_best_checkpoint(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward/score"""
    ckpts = os.listdir(os.path.join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    print('loading checkpoint {}...'.format(ckpts[0]))
    # ckpt = torch.jit.load(
    ckpt = torch.load(
        os.path.join(model_dir, 'ckpt/{}'.format(ckpts[0])), map_location=torch.device('cpu')
    )['state_dict']
    return ckpt


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


'''
def get_summary_snapshot_info(initial_summary_info, only_initial_summary=False):
    # initial_summary_info is a list of tuples of (query, [list of sentence indices added for the query])
    # the first query will likely be an empty string (a general summary without a query)

    # if None, return None
    if initial_summary_info == None or initial_summary_info == []:
        if only_initial_summary:
            return None
        return None, None, None, None

    queryList = [] # list of queries until the last one in the summary snapshot
    fullInitialSummIndices = [] # list of sentence indices accumulated as a result of all queries until the last one
    for queryInfo in initial_summary_info[:-1]:
        queryList.append(queryInfo[0])
        fullInitialSummIndices.extend(queryInfo[1])
    # get the last query in the snapshot, and the last summary addition as a result of the last query:
    lastQuery, lastSummaryAddition = initial_summary_info[-1]

    if only_initial_summary:
        return fullInitialSummIndices + lastSummaryAddition # the full initial summary available
    else:
        return fullInitialSummIndices, lastQuery, queryList, lastSummaryAddition
'''