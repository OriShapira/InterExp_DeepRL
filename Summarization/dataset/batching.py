import torch
from collections import defaultdict
from cytoolz import curry
from nltk.tokenize import word_tokenize
from toolz.sandbox.core import unzip

from utils import PAD, UNK, START, END


class ArticleBatcher(object):
    def __init__(self, word2id, cuda=True):
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._word2id = word2id

    def __call__(self, raw_article_sents, min_len_pad=5):
        articles = _conver2id(UNK, self._word2id, raw_article_sents)
        article = _pad_batch_tensorize(articles, PAD, cuda=False, min_len_pad=min_len_pad).to(self._device)
        return article


def _conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]


def _pad_batch_tensorize(inputs, pad, cuda=True, min_len_pad=5):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :param min_len_pad: if the inputs are shorter than this length, then the returned tensors are padded to that length. -1 means not relevant.
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max(len(ids) for ids in inputs)
    tensor_shape = (batch_size, max(max_len, min_len_pad))
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    return tensor

@curry
def data_loader_collate(batch):
    id_batch, topicId_batch, docs_batch, refs_batch, initial_batch, queriesInfo_batch = unzip(batch)
    all_docs_sents = [list(map(_tokenize_sentence_list, docs)) for docs in docs_batch] # list(filter(bool, map(tokenize(None), art_batch)))
    all_refs_sents = [list(map(_tokenize_sentence_list, refs)) for refs in refs_batch] # list(filter(bool, map(tokenize(None), abs_batch)))
    return all_docs_sents, all_refs_sents, list(id_batch), list(topicId_batch), list(initial_batch), list(queriesInfo_batch)

@curry
def _tokenize_sentence_list(sent_list):
    #tokenized_sentence_list = []
    #for s in sent_list:
    #    tokenized_sentence = word_tokenize(s.lower())
    #    # if the sentence is 2 or 3 tokens long, and the last token is a '.', then put the '.' back to the
    #    # second to last token (example: u.s . -> u.s.)
    #    if 2 <= len(tokenized_sentence) <= 3 and tokenized_sentence[-1] == '.':
    #        tokenized_sentence = tokenized_sentence[:-1]
    #        tokenized_sentence[-1] += '.'
    #    tokenized_sentence_list.append(tokenized_sentence)
    #return tokenized_sentence_list
    return [word_tokenize(s.lower()) for s in sent_list]