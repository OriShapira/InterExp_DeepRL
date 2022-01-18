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
from metric_k import compute_kp_overlap, compute_kp_tfidf, evaluateKpList, compute_partial_phrase_overlap_score, compute_kp_tf
from model_k.extract import PtrExtractSumm
from model_k.rl import ActorCritic
from dataset_k.batching import ArticleBatcher
from dataset_k.utils import prepare_data_for_model_input

#from evaluate import calc_official_rouge
from utils import load_best_checkpoint



def decode_single_datum(model_dir, topic_docs_sents, initial_summ_inds, num_kps_needed, beta=None, cuda=False):
    """
    Decode a single instance of the K_RLMMR
    :param model_dir: Path to the folder of the saved model
    :param topic_docs_sents: list of list of strings (sentences per doc in the topic)
    :param initial_summ_inds: list of pair-lists [doc_idx, sent_idx]
    :param cuda: is cuda
    :return: two lists of strings: initial_summ_sents, expansion_sents
    """
    start = time()
    # get the model config arguments for initializing the RLExtractor obj:
    with open(os.path.join(model_dir, 'config.json')) as f:
        model_config = json.loads(f.read())
    method_mode_toUse = model_config['method_mode']
    importance_mode_toUse = model_config['importance_mode']
    diversity_mode_toUse = model_config['diversity_mode']
    beta_toUse = model_config['beta'] if beta == None else beta
    #phrasing_method = model_config['phrasing_method']
    #input_mode = model_config['input_mode'] if model_config['input_mode'] != '' else '""'

    # load the agent:
    topic_docs = [' '.join(list(concat(doc_sents))) for doc_sents in topic_docs_sents]  # one text per document (used as the base data)
    agent = RLExtractor(model_dir, topic_docs, beta_toUse, method_mode_toUse, importance_mode_toUse, diversity_mode_toUse,
                        cuda=(torch.cuda.is_available() and cuda))

    # prepare the sentences and indices for the agent:
    topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted = \
        prepare_data_for_model_input(topic_docs_sents, initial_summ_inds)

    # invoke the agent to get inferred KPs for the topic docs and initial summary specified:
    chosen_kp_indices, chosen_kp_scores = agent(topic_docs_filtered, num_kps_needed, initial_summ=initial_summ_indices_adjusted)

    # get the chosen and initial KPs from their indices:
    initial_summ_kps_tokens = [topic_docs_filtered_flat[idx] for idx in initial_summ_indices_adjusted]
    initial_summ_kps = [' '.join(kp_tokens) for kp_tokens in initial_summ_kps_tokens]
    chosen_summ_kps_tokens = [topic_docs_filtered_flat[idx.item()] for idx in chosen_kp_indices]
    chosen_summ_kps = [' '.join(kp_tokens) for kp_tokens in chosen_summ_kps_tokens]

    print(f'finished in {timedelta(seconds=int(time() - start))}!')

    return chosen_summ_kps, chosen_kp_scores, initial_summ_kps



def _read_output_dec_file(dec_file_path):
    chosen_summ_kps_tokens = []
    with open(dec_file_path, 'r') as fIn:
        read_vals = False
        for line in fIn:
            line = line.strip()
            if line == '---': # start reading after the line with this symbol
                read_vals = True
                continue
            if read_vals:
                chosen_summ_kps_tokens.append(line.split())
    return chosen_summ_kps_tokens

def decode_full_dataset(agent, loader, save_dir, split_name, num_steps, recompute_if_exists=True):
    outdirpath_dec = join(save_dir, 'dec')
    if not os.path.exists(outdirpath_dec):
        os.makedirs(outdirpath_dec)
    # unofficial_scores = [] # flat list of the summary scores over all batches and instances
    num_kps_in_summ_full = []  # flat list of the summary lengths (num kps) over all batches and instances
    num_kps_in_summ_noInitial = []  # flat list of the summary lengths (num kps) over all batches and instances
    summ_lengths_full = []  # flat list of the summary lengths (num tokens) over all batches and instances (including initial summs)
    summ_lengths_noInitial = []  # flat list of the summary lengths (num tokens) over all batches and instances (w/out initial summs)
    refsumms_by_topic = {}  # topicId -> list of summaries
    auc_score_per_topic_overlap_full = {}
    auc_score_per_topic_overlap_noInitial = {}
    auc_score_per_topic_tfidf_full = {}
    auc_score_per_topic_ppf_full = {}
    auc_score_per_topic_tf_full = {}
    auc_score_per_topic_tfidf_noInitial = {}
    auc_score_per_topic_ppf_noInitial = {}
    auc_score_per_topic_tf_noInitial = {}
    kp_eval_scores_per_topic_full = {}
    kp_eval_scores_per_topic_noInitial = {}
    kp_mean_score_per_topic_overlap = {}
    kp_mean_score_per_topic_tfidf = {}
    kp_mean_score_per_topic_ppf = {}
    kp_mean_score_per_topic_tf = {}
    with torch.no_grad():
        # for each batch:
        sampleIdsCovered = {} # the train set is cycled infinitely, so stop once a sample repeats
        for docs_batch, refs_batch, ids_batch, topicIdx_batch, initialsumm_batch, refKps_batch in loader:
            # in case the loader is re-cycled (train set), stop the loader:
            if ids_batch[0] in sampleIdsCovered:
                break
            # for each sample (topic):
            for topic_docs, topic_refsumms, datum_id, topic_id, initialsumm_indices, ref_kps in \
                    zip(docs_batch, refs_batch, ids_batch, topicIdx_batch, initialsumm_batch, refKps_batch):
                dec_file_path = join(outdirpath_dec, f'{datum_id}.dec')

                # prepare the sentences and indices for the agent:
                topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted = \
                    prepare_data_for_model_input(topic_docs, initialsumm_indices)

                if not recompute_if_exists and os.path.exists(dec_file_path):
                    chosen_summ_kps_tokens = _read_output_dec_file(dec_file_path)
                else:
                    # for validation, we get a full summary without an initial summary
                    chosen_kp_indices, chosen_kp_scores = \
                        agent(topic_docs_filtered, num_steps, datum_name=topic_id, split=split_name,
                              initial_summ=initial_summ_indices_adjusted)
                    chosen_summ_kps_tokens = [topic_docs_filtered_flat[idx.item()] for idx in chosen_kp_indices]

                #num_kps_in_summ_full.append(len(initialsumm_indices) + len(chosen_kp_indices))
                #num_kps_in_summ_noInitial.append(len(chosen_kp_indices))
                num_kps_in_summ_full.append(len(initialsumm_indices) + len(chosen_summ_kps_tokens))
                num_kps_in_summ_noInitial.append(len(chosen_summ_kps_tokens))

                # get the initial_summ kps:
                initial_summ_kps_tokens = [topic_docs_filtered_flat[idx] for idx in initial_summ_indices_adjusted]


                # write out the system summary:
                sys_summ = '\n'.join([' '.join(kp_tokens)
                                      for kp_tokens in initial_summ_kps_tokens + [['---']] + chosen_summ_kps_tokens])
                with open(dec_file_path, 'w') as fOut:
                    fOut.write(sys_summ)

                #refs_tokens = [list(concat(ref_sents)) for ref_sents in topic_refsumms]  # concatenated tokens per refsumm

                # get the initial summary KPs scores and token-lengths:
                scores_accumulating_overlap = [0.]
                scores_accumulating_tfidf = [0.]
                scores_accumulating_ppf = [0.]
                scores_accumulating_tf = [0.]
                lengths_accumulating = [0]
                indices_accumulating = [0]
                if initial_summ_kps_tokens:
                    for kpIdx in range(len(initial_summ_kps_tokens)):
                        curKp = initial_summ_kps_tokens[kpIdx]
                        precedingKps = initial_summ_kps_tokens[0:kpIdx]
                        score_overlap = compute_kp_overlap(curKp, topic_refsumms, [], precedingKps, stem=True,
                                                           remove_stop=True, weight_initial=0, weight_preceding=1)
                        scores_accumulating_overlap.append(scores_accumulating_overlap[-1] + score_overlap)
                        score_tfidf = compute_kp_tfidf(curKp, topic_refsumms, [], precedingKps, stem=True,
                                                       remove_stop=True, weight_initial=0, weight_preceding=1)
                        scores_accumulating_tfidf.append(scores_accumulating_tfidf[-1] + score_tfidf)
                        score_ppf = compute_partial_phrase_overlap_score(curKp, topic_refsumms, [], precedingKps,
                                                                         stem=True, remove_stop=True,
                                                                         mmr_lambda=0.5, redundancy_alpha=0.5)
                        scores_accumulating_ppf.append(scores_accumulating_ppf[-1] + score_ppf)
                        score_tf = compute_kp_tf(curKp, topic_refsumms, [], precedingKps,
                                                 stem=True, remove_stop=True,
                                                 weight_initial=0, weight_preceding=1)
                        scores_accumulating_tf.append(scores_accumulating_tf[-1] + score_tf)
                        lengths_accumulating.append(lengths_accumulating[-1] + len(curKp))
                        indices_accumulating.append(indices_accumulating[-1] + 1)

                # get the KP scores at each of the added KPs:
                scores_per_kp_overlap = []
                scores_per_kp_tfidf = []
                scores_per_kp_ppf = []
                scores_per_kp_tf = []
                for kpIdx, kp_tokens in enumerate(chosen_summ_kps_tokens):
                    precedingKps = chosen_summ_kps_tokens[0:kpIdx]
                    score_overlap = compute_kp_overlap(kp_tokens, topic_refsumms, initial_summ_kps_tokens, precedingKps,
                                                       stem=True, remove_stop=True,
                                                       weight_initial=0.5, weight_preceding=0.5)
                    scores_per_kp_overlap.append(score_overlap)
                    scores_accumulating_overlap.append(scores_accumulating_overlap[-1] + score_overlap)
                    score_tfidf = compute_kp_tfidf(kp_tokens, topic_refsumms, initial_summ_kps_tokens, precedingKps,
                                                   stem=True, remove_stop=True, weight_initial=0.5, weight_preceding=0.5)
                    scores_per_kp_tfidf.append(score_tfidf)
                    scores_accumulating_tfidf.append(scores_accumulating_tfidf[-1] + score_tfidf)
                    score_ppf = compute_partial_phrase_overlap_score(kp_tokens, topic_refsumms,
                                                                     initial_summ_kps_tokens, precedingKps,
                                                                     stem=True, remove_stop=True,
                                                                     mmr_lambda=0.5, redundancy_alpha=0.5)
                    scores_per_kp_ppf.append(score_ppf)
                    scores_accumulating_ppf.append(scores_accumulating_ppf[-1] + score_ppf)
                    score_tf = compute_kp_tf(kp_tokens, topic_refsumms,
                                             initial_summ_kps_tokens, precedingKps,
                                             stem=True, remove_stop=True,
                                             weight_initial=0.5, weight_preceding=0.5)
                    scores_per_kp_tf.append(score_tf)
                    scores_accumulating_tf.append(scores_accumulating_tf[-1] + score_tf)

                    lengths_accumulating.append(lengths_accumulating[-1] + len(kp_tokens))
                    indices_accumulating.append(indices_accumulating[-1] + 1)

                # compute the AUC for the accumulating summary:
                #aucScore = auc(indices_accumulating, scores_accumulating)

                # scores including the initial summary:
                aucScore_overlap_full = auc(lengths_accumulating, scores_accumulating_overlap)
                aucScore_tfidf_full = auc(lengths_accumulating, scores_accumulating_tfidf)
                aucScore_ppf_full = auc(indices_accumulating, scores_accumulating_ppf)
                aucScore_tf_full = auc(indices_accumulating, scores_accumulating_tf)
                auc_score_per_topic_overlap_full[topic_id] = aucScore_overlap_full / (len(lengths_accumulating) - 1)
                auc_score_per_topic_tfidf_full[topic_id] = aucScore_tfidf_full / (len(lengths_accumulating) - 1)
                auc_score_per_topic_ppf_full[topic_id] = aucScore_ppf_full / (len(lengths_accumulating) - 1)
                auc_score_per_topic_tf_full[topic_id] = aucScore_tf_full / (len(lengths_accumulating) - 1)
                summ_lengths_full.append(lengths_accumulating[-1])

                # scores not including the initial summary:
                initialCnt = len(initial_summ_kps_tokens)
                aucScore_overlap_noInitial = auc(lengths_accumulating[initialCnt:], scores_accumulating_overlap[initialCnt:])
                aucScore_tfidf_noInitial = auc(lengths_accumulating[initialCnt:], scores_accumulating_tfidf[initialCnt:])
                aucScore_ppf_noInitial = auc(indices_accumulating[initialCnt:], scores_accumulating_ppf[initialCnt:])
                aucScore_tf_noInitial = auc(indices_accumulating[initialCnt:], scores_accumulating_tf[initialCnt:])
                auc_score_per_topic_overlap_noInitial[topic_id] = aucScore_overlap_noInitial / (len(lengths_accumulating) - initialCnt - 1)
                auc_score_per_topic_tfidf_noInitial[topic_id] = aucScore_tfidf_noInitial / (len(lengths_accumulating) - initialCnt - 1)
                auc_score_per_topic_ppf_noInitial[topic_id] = aucScore_ppf_noInitial / (len(lengths_accumulating) - initialCnt - 1)
                auc_score_per_topic_tf_noInitial[topic_id] = aucScore_tf_noInitial / (len(lengths_accumulating) - initialCnt - 1)
                summ_lengths_noInitial.append(lengths_accumulating[-1] - lengths_accumulating[initialCnt])
                kp_mean_score_per_topic_overlap[topic_id] = np.mean(scores_per_kp_overlap)
                kp_mean_score_per_topic_tfidf[topic_id] = np.mean(scores_per_kp_tfidf)
                kp_mean_score_per_topic_ppf[topic_id] = np.mean(scores_per_kp_ppf)
                kp_mean_score_per_topic_tf[topic_id] = np.mean(scores_per_kp_tf)

                if len(ref_kps) > 0:
                    initial_kps = [(' '.join(kpTokens), len(initial_summ_kps_tokens) - kpIdx - 1)
                                   for kpIdx, kpTokens in enumerate(initial_summ_kps_tokens)]
                    pred_kps = [(' '.join(kpTokens), len(initial_kps) + len(chosen_summ_kps_tokens) - kpIdx - 1)
                                for kpIdx, kpTokens in enumerate(chosen_summ_kps_tokens)]
                    kp_eval_scores_per_topic_noInitial[topic_id] = evaluateKpList(pred_kps, ref_kps)
                    kp_eval_scores_per_topic_full[topic_id] = evaluateKpList(initial_kps + pred_kps, ref_kps)
                else:
                    kp_eval_scores_per_topic_noInitial = None
                    kp_eval_scores_per_topic_full = None

                sampleIdsCovered[datum_id] = True

                ## for the topic, keep the reference summaries as one full text per ref:
                #refsumms_by_topic[datum_id] = [' '.join(' '.join(sent) for sent in refsumm) for refsumm in topic_refsumms]

    #official_scores = calc_official_rouge(refsumms_by_topic, outdirpath_dec, dataset_name=split_name, summ_len_limit=testing_summary_len)
    official_scores = {}
    official_scores['auc_norm_ppf_full'] = np.mean(list(auc_score_per_topic_ppf_full.values()))
    official_scores['auc_norm_ppf_noInitial'] = np.mean(list(auc_score_per_topic_ppf_noInitial.values()))
    official_scores['mean_kp_score_ppf'] = np.mean(list(kp_mean_score_per_topic_ppf.values()))
    official_scores['auc_norm_overlap_full'] = np.mean(list(auc_score_per_topic_overlap_full.values()))
    official_scores['auc_norm_overlap_noInitial'] = np.mean(list(auc_score_per_topic_overlap_noInitial.values()))
    official_scores['mean_kp_score_overlap'] = np.mean(list(kp_mean_score_per_topic_overlap.values()))
    official_scores['auc_norm_tfidf_full'] = np.mean(list(auc_score_per_topic_tfidf_full.values()))
    official_scores['auc_norm_tfidf_noInitial'] = np.mean(list(auc_score_per_topic_tfidf_noInitial.values()))
    official_scores['mean_kp_score_tfidf'] = np.mean(list(kp_mean_score_per_topic_tfidf.values()))
    official_scores['auc_norm_tf_full'] = np.mean(list(auc_score_per_topic_tf_full.values()))
    official_scores['auc_norm_tf_noInitial'] = np.mean(list(auc_score_per_topic_tf_noInitial.values()))
    official_scores['mean_kp_score_tf'] = np.mean(list(kp_mean_score_per_topic_tf.values()))
    # if there are KP eval scores (against DUC01), log those too:
    if kp_eval_scores_per_topic_full:
        metrics = list(list(kp_eval_scores_per_topic_full.values())[0].keys())
        for metric in metrics: # e.g. f1@1, f1@5, etc.
            official_scores[f'mean_{metric}_full'] = \
                np.mean([kp_eval_scores_per_topic_full[topic_id][metric] for topic_id in kp_eval_scores_per_topic_full])
            official_scores[f'mean_{metric}_noInitial'] = \
                np.mean([kp_eval_scores_per_topic_noInitial[topic_id][metric] for topic_id in kp_eval_scores_per_topic_noInitial])

    official_scores['num_kps_mean_full'] = np.mean(num_kps_in_summ_full)
    official_scores['num_kps_mean_noInitial'] = np.mean(num_kps_in_summ_noInitial)
    official_scores['kps_tokens_len_mean_full'] = np.mean(summ_lengths_full)
    official_scores['kps_tokens_len_mean_noInitial'] = np.mean(summ_lengths_noInitial)

    return official_scores



class RLExtractor(object):
    """
    Class for decoding (inference time). Loads the extractor given and loads it into an ActorCritic.
    """
    def __init__(self, extractor_model_dir, data_source, beta, method_mode,
                 importance_mode, diversity_mode, cuda=True):
        """
        :param extractor_model_dir: The path to where the model is
        :param data_source: Either the base directory of the database, or a list of texts (text per source "document")
        :param cuda:
        """
        ext_meta = json.load(open(join(extractor_model_dir, 'meta.json')))
        assert ext_meta['net'] == 'kp-rl_mmr'
        self._device = torch.device('cuda' if cuda else 'cpu')
        # the model is trained on the k-rlmmr data:
        if 'extractor' in ext_meta['net_args']:
            ext_args = ext_meta['net_args']['extractor']['net_args']
        word2id = pkl.load(open(join(extractor_model_dir, 'agent_vocab.pkl'), 'rb'))
        extractor = PtrExtractSumm(**ext_args)
        agent = ActorCritic(extractor._sent_enc,
                            extractor._art_enc,
                            extractor._extractor,
                            ArticleBatcher(word2id, cuda),
                            data_source, beta,
                            method_mode, importance_mode, diversity_mode)
        ext_ckpt = load_best_checkpoint(extractor_model_dir, reverse=True)
        agent.load_state_dict(ext_ckpt)

        self.agent = agent.to(self._device)
        #self._word2id = word2id
        #self._id2word = {i: w for w, i in word2id.items()}

    def __call__(self, raw_article_sents, num_steps, datum_name='', split='', initial_summ=None):
        self.agent.eval()
        indices, scores = self.agent(raw_article_sents, num_steps, datum_name=datum_name, split=split,
                                     initial_summ=initial_summ)
        return indices, scores
