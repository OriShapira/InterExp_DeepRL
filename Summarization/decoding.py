""" decoding utilities"""
import json
import os
from os.path import join
import pickle as pkl
import numpy as np
from cytoolz import concat
import torch
from datetime import timedelta
from time import time
from sklearn.metrics import auc
from metric import compute_rouge_n
from model.extract import PtrExtractSumm
from model.rl import ActorCritic
from dataset.batching import ArticleBatcher
from dataset.utils import prepare_data_for_model_input

from evaluate import calc_official_rouge
from utils import load_best_checkpoint



def decode_single_datum(model_dir, topic_docs_sents, initial_summ_inds, queries_info, beta, query_encode,
                        importance_mode, diversity_mode, max_sent_len, cuda=False):
    """
    Decode a single instance (initial summary + query -> output expansion)
    :param model_dir: Path to the folder of the saved model
    :param topic_docs_sents: list of list of strings (sentences per doc in the topic)
    :param initial_summ_inds: list of pair-lists [doc_idx, sent_idx]
    :param queries_info: list of pair-lists [query_str, num_sentences_for_query]
    :param query_encode: should queries be encoded into the input
    :param cuda: is cuda
    :return: two lists of strings: initial_summ_sents, expansion_sents and list of indices of expansion sentences
    """
    start = time()
    # setup model
    with open(os.path.join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())

    # initialize the decoding agent:
    topic_docs = [' '.join(list(concat(doc_sents))) for doc_sents in topic_docs_sents] # one text per document (used as the base data)
    agent = RLExtractor(model_dir, topic_docs, beta, query_encode, importance_mode, diversity_mode,
                        cuda=(torch.cuda.is_available() and cuda))

    # prepare the sentences and indices for the agent:
    topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted, queriesList, sent_idx_mapper = \
        prepare_data_for_model_input(topic_docs_sents, initial_summ_inds, queries_info, get_index_mapper=True,
                                     max_num_steps=-1)
    queriesStrList = [qStr for qStr, qGroupIdx in queriesList]

    # get the chosen sentence indices (with respect to topic_docs_filtered_flat):
    chosen_sent_indices = agent(topic_docs_filtered, initial_summ=initial_summ_indices_adjusted,
                                queries_list=queriesStrList, max_sent_len=max_sent_len)

    # the summary expansion:
    expansion_sents_indices = [idx.item() for idx in chosen_sent_indices]
    expansion_sents = list([topic_docs_filtered_flat[idx] for idx in expansion_sents_indices])
    # the initial summary sentences:
    initial_summ_sents = list([topic_docs_filtered_flat[sent_idx_mapper[orig_idx]] for orig_idx in initial_summ_inds])

    print(f'finished in {timedelta(seconds=int(time() - start))}!')

    # get the original indices (pair-lists) from the absolute indices:
    orig_idxs = list(sent_idx_mapper.keys())
    new_idxs = list(sent_idx_mapper.values())
    expansion_sents_indices_orig = [orig_idxs[new_idxs.index(val)] for val in expansion_sents_indices]

    return initial_summ_sents, expansion_sents, expansion_sents_indices_orig


def decode_full_dataset(agent, loader, save_dir, split_name, max_num_steps=-1, testing_summary_len=250,
                        reset_rouge_evaluator=False, max_sent_len=999):
    """

    :param agent:
    :param loader:
    :param save_dir:
    :param split_name:
    :param max_num_steps:
    :param testing_summary_len:
    :param reset_rouge_evaluator: the ROUGE evaluator reuses the reference summary temp folder so that it isn't
    re-prepared on every evaluation. This is the default behavior, but if you need to re-prepare the reference
    summaries on every evaluation, set as True.
    :return:
    """
    outdirpath_dec = join(save_dir, 'dec')
    if not os.path.exists(outdirpath_dec):
        os.makedirs(outdirpath_dec)
    num_sents_in_summ = []  # flat list of the summary lengths (num sentences) over all batches and instances
    summ_lengths = []  # flat list of the summary lengths (num tokens) over all batches and instances
    refsumms_by_topic = {}  # topicId -> list of summaries
    auc_score_per_topic = {}
    with torch.no_grad():
        # for each batch:
        for docs_batch, refs_batch, ids_batch, topicIdx_batch, initialsumm_batch, queriesInfo_batch in loader:
            # for each sample (topic):
            for topic_docs, topic_refsumms, datum_id, topic_id, initialsumm_indices, queriesInfo in \
                    zip(docs_batch, refs_batch, ids_batch, topicIdx_batch, initialsumm_batch, queriesInfo_batch):
                # prepare the sentences and indices for the agent:
                topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted, queriesList = \
                    prepare_data_for_model_input(topic_docs, initialsumm_indices, queriesInfo, max_num_steps=max_num_steps)
                queriesStrList = [qStr for qStr, qGroupIdx in queriesList]

                # for validation, we get a full summary without any queries and without an initial summary
                chosen_sent_indices = agent(topic_docs_filtered, datum_name=topic_id, split=split_name,
                                            initial_summ=initial_summ_indices_adjusted, queries_list=queriesStrList,
                                            max_sent_len=max_sent_len)
                                            #max_num_sentences_to_add=round(testing_summary_len / 15.0))
                num_sents_in_summ.append(len(initialsumm_indices) + len(chosen_sent_indices))

                # get the intial_summ sentences and the chosen sentences:
                initial_summ_sents_tokens = [topic_docs_filtered_flat[idx.item()] for idx in initial_summ_indices_adjusted]
                chosen_summ_sents_tokens = [topic_docs_filtered_flat[idx.item()] for idx in chosen_sent_indices]

                # write out the system summary:
                sys_summ = '\n'.join([' '.join(sent_tokens)
                                      for sent_tokens in initial_summ_sents_tokens + chosen_summ_sents_tokens])
                fname = join(outdirpath_dec, f'{datum_id}.dec')
                with open(fname, 'w') as fOut:
                    fOut.write(sys_summ)

                # get the initial summary score and token-length:
                initial_summ_score = 0.
                initial_summ_len = 0
                if initial_summ_sents_tokens:
                    initial_summ_score = compute_rouge_n(initial_summ_sents_tokens, topic_refsumms, #refs_tokens,
                                                         n=1, mode='r', stem=True, preceding_text=[])
                    initial_summ_len = sum(len(sent) for sent in initial_summ_sents_tokens)

                # get the ROUGE-1 Recall scores at each of the added sentences:
                sents_accumulating = initial_summ_sents_tokens # list of sentences so far in the tested summary
                scores_accumulating = [initial_summ_score]
                lengths_accumulating = [initial_summ_len]
                for sent_tokens in chosen_summ_sents_tokens:
                    score = compute_rouge_n(sent_tokens, topic_refsumms, #refs_tokens,
                                            n=1, mode='r', stem=True, preceding_text=sents_accumulating)
                    sents_accumulating.extend(sent_tokens)
                    scores_accumulating.append(score)
                    lengths_accumulating.append(lengths_accumulating[-1] + len(sent_tokens))

                # compute the AUC for the accumulating summary:
                aucScore = auc(lengths_accumulating, scores_accumulating)
                # the normalized AUC score (AUC per word):
                auc_score_per_topic[topic_id] = aucScore / lengths_accumulating[-1] # divide by full summary length
                summ_lengths.append(lengths_accumulating[-1])

                # for the topic, keep the reference summaries as one full text per ref:
                refsumms_by_topic[datum_id] = [' '.join(' '.join(sent) for sent in refsumm) for refsumm in topic_refsumms]

    official_scores = calc_official_rouge(refsumms_by_topic, outdirpath_dec, dataset_name=split_name, summ_len_limit=testing_summary_len, reset_refsumms=reset_rouge_evaluator)
    official_scores['auc_norm'] = np.mean(list(auc_score_per_topic.values()))

    return official_scores, np.mean(num_sents_in_summ), np.mean(summ_lengths)



class RLExtractor(object):
    """
    Class for decoding (inference time). Loads the extractor given and loads it into an ActorCritic.
    """
    def __init__(self, extractor_model_dir, data_source, beta, query_encode,
                 importance_mode, diversity_mode, method_name='soft-attn', cuda=True):
        """
        :param extractor_model_dir: The path to where the model is
        :param data_source: Either the base directory of the database, or a list of texts (text per source "document")
        :param cuda:
        """
        ext_meta = json.load(open(join(extractor_model_dir, 'meta.json')))
        assert ext_meta['net'] == 'q-rl_mmr'
        self._device = torch.device('cuda' if cuda else 'cpu')
        # the model can either be trained straight on the q-rlmmr data:
        if 'extractor' in ext_meta['net_args']:
            ext_args = ext_meta['net_args']['extractor']['net_args']
        # or the model can be trained on the q-rlmmr data after being pretrined on generic MDS summarization:
        elif 'summarizer' in ext_meta['net_args']:
            ext_args = ext_meta['net_args']['summarizer']['net_args']['extractor']['net_args']
        word2id = pkl.load(open(join(extractor_model_dir, 'agent_vocab.pkl'), 'rb'))
        extractor = PtrExtractSumm(**ext_args)
        agent = ActorCritic(extractor._sent_enc,
                            extractor._art_enc,
                            extractor._extractor,
                            ArticleBatcher(word2id, cuda),
                            data_source, beta, query_encode, method_name,
                            importance_mode, diversity_mode)
        ext_ckpt = load_best_checkpoint(extractor_model_dir, reverse=True)
        agent.load_state_dict(ext_ckpt)

        self.agent = agent.to(self._device)

    def __call__(self, raw_article_sents, datum_name='', split='', initial_summ=None, queries_list=None,
                 max_sent_len=999):
        self.agent.eval()
        indices = self.agent(raw_article_sents, datum_name=datum_name, split=split,
                             initial_summ=initial_summ, queries_list=queries_list, max_sent_len=max_sent_len)
        return indices
