""" ROUGE/query-similarity utils"""
from collections import Counter, deque
import numpy as np
import spacy
from cytoolz import concat, curry
from nltk.stem.porter import *
from nltk.corpus import stopwords
import string

STOP_WORDS = stopwords.words('english')
STOP_TOKENS = {t:1 for t in STOP_WORDS + list(string.punctuation) + ['``', "''", '--', "'s", "'d", "'ve", "'ll", "'re", "'t"]}
_stemmer = PorterStemmer()

def make_n_grams(seq, n):
    """ return iterator """
    ngrams = (tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    return ngrams


def _n_gram_match(summ_tokens, ref_tokens, n):
    summ_grams = Counter(make_n_grams(summ_tokens, n))
    ref_grams = Counter(make_n_grams(ref_tokens, n))
    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count


def prepareTextsForRouge(output, reference, stem, remove_stop, output_max_len, preceding_text,
                         ref_sent_idx, sentence_level, ignore_preceding):

    # prepend the preceding text:
    if not ignore_preceding:
        sourceTextTokens = preceding_text + output if preceding_text else output
    else:
        sourceTextTokens = output

    # truncate the output at the max len:
    if output_max_len > 0:
        sourceTextTokens = sourceTextTokens[:output_max_len]

    # if there aren't several references, make the reference a list of one reference:
    if not isinstance(reference[0][0], list):  # a single reference is a list of lists of strings (tokens)
        references = [reference]
    else:
        references = reference

    # if we need to compare on the sentence level, then just get the specified sentence as the reference:
    if sentence_level and ref_sent_idx != None:
        refIdx, sentIdx = ref_sent_idx
        if refIdx >= len(references) or sentIdx >= len(references[refIdx]):
            return None  # if we're on the sentence level, but there is no matching reference sentence, return None
        references = [[references[refIdx][sentIdx]]]

    # remove stop words/tokens if required:
    if remove_stop:
        sourceTextTokens = [t for t in sourceTextTokens if t not in STOP_TOKENS]
        references = [[[t for t in sent if t not in STOP_TOKENS] for sent in ref] for ref in references]

    # stem if needed:
    if stem:
        sourceTextTokens = [_stemmer.stem(t) for t in sourceTextTokens]
        references = [[[_stemmer.stem(t) for t in sent] for sent in ref] for ref in references]

    return sourceTextTokens, references


@curry
def compute_rouge_n(output, reference, n=1, mode='f', stem=False, remove_stop=False, output_max_len=-1,
                    preceding_text=None, ref_sent_idx=None, normalize=False, sentence_level=False,
                    ignore_preceding=False):
    """
    Compute ROUGE-N for a single pair of summary and reference(s). The reference can be a list of references.
    :param output: the text to evaluate (list of tokens)
    :param reference: the reference to evaluate against (list of list of tokens (sentences)) or list of references
    :param n: the n-gram size
    :param mode: 'r' 'p' or 'f'
    :param output_max_len: truncate length (0 or lower means no truncation)
    :param preceding_text: optional list of tokens preceding the output. This is simply prepended to the output list of tokens.
    :param ref_sent_idx: the sentence index (tuple of (ref_idx, sent_idx)) to use instead of the full refSumms
    :param normalize: should the score be normalized by the output length
    :param sentence_level: should only the specified reference summary sentence index be used for computing rouge
    :param ignore_preceding: should the preceding text be ignored or prepended when computing the score
    :return:
    """
    assert mode in list('fpr')  # F-1, precision, recall

    sourceTextTokens, references = prepareTextsForRouge(output, reference, stem, remove_stop, output_max_len,
                                                        preceding_text, ref_sent_idx, sentence_level, ignore_preceding)

    # get the scores over all references:
    recall_per_ref = []
    precision_per_ref = []
    for ref in references:
        refTokens = list(concat(ref)) # list of list of tokens (sentence level) -> list of tokens (ref level)

        match = _n_gram_match(sourceTextTokens, refTokens, n)#, stem=stem)
        recall_per_ref.append(float(match) / len(refTokens) if len(refTokens) > 0 else 0.)
        precision_per_ref.append(float(match) / len(sourceTextTokens) if len(sourceTextTokens) > 0 else 0.)

    if mode == 'p':
        score = np.mean(precision_per_ref)
    elif mode == 'r':
        score = np.mean(recall_per_ref)
    else: # micro f1
        precision = np.mean(precision_per_ref)
        recall = np.mean(recall_per_ref)
        if precision + recall > 0:
            score = 2 * (precision * recall) / (precision + recall)
        else:
            score = 0.0

    if normalize and len(sourceTextTokens) > 0:
        score /= len(sourceTextTokens)

    return score



@curry
def compute_delta_rouge_n(output, reference, n=1, mode='f', stem=False, remove_stop=False, preceding_text=None,
                          ref_sent_idx=None, normalize=False):
    """
    Compute delta ROUGE-N between preceding_text+output to output against the reference(s). The reference can be a list of references.
    :param output: the additional text (list of tokens) for which to get the delta score
    :param reference: the reference to evaluate against (list of list of tokens (sentences)) or list of references
    :param n: the n-gram size
    :param mode: 'r' 'p' or 'f'
    :param output_max_len: truncate length (0 or lower means no truncation)
    :param preceding_text: the base summary (list of tokens) against which to get the delta score
    :param ref_sent_idx: NOT USABLE
    :param normalize: should the score be normalized by the output length
    :return:
    """
    if preceding_text:
        baseScore = compute_rouge_n(preceding_text, reference, n=n, mode=mode, stem=stem, remove_stop=remove_stop)
        fullScore = compute_rouge_n(preceding_text + output, reference, n=n, mode=mode, stem=stem, remove_stop=remove_stop)
        score = fullScore - baseScore
    else:
        score = compute_rouge_n(output, reference, n=n, mode=mode, stem=stem, remove_stop=remove_stop)

    if normalize and len(output) > 0:
        score /= len(output)
    return score


def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b) + 1)]
          for _ in range(0, len(a) + 1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]


@curry
def compute_rouge_l(output, reference, mode='f', stem=False, remove_stop=False, output_max_len=-1, preceding_text=None,
                    ref_sent_idx=None, normalize=False, sentence_level=False, ignore_preceding=False):
    """
    Compute ROUGE-L for a single pair of summary and reference(s), reference are list of words. The reference
    can be a list of references.
    :param output: the text to evaluate (list of tokens)
    :param reference: the reference to evaluate against (list of list of tokens (sentences)) or list of references
    :param mode: 'r' 'p' or 'f'
    :param output_max_len: truncate length (0 or lower means no truncation)
    :param preceding_text: optional list of tokens preceding the output. This is simply prepended to the output list of tokens.
    :param ref_sent_idx: the sentence index (tuple of (ref_idx, sent_idx)) to use instead of the full refSumms
    :param normalize: should the score be normalized by the output length
    :param sentence_level: should only the specified reference summary sentence index be used for computing rouge
    :param ignore_preceding: should the preceding text be ignored or prepended when computing the score
    :return:
    """
    assert mode in list('fpr')  # F-1, precision, recall

    sourceTextTokens, references = prepareTextsForRouge(output, reference, stem, remove_stop, output_max_len,
                                                        preceding_text, ref_sent_idx, sentence_level, ignore_preceding)

    # get the scores over all references:
    scores_per_ref = []
    for ref in references:
        refTokens = list(concat(ref))  # list of list of tokens (sentence level) -> list of tokens (ref level)

        lcs = _lcs_len(sourceTextTokens, refTokens)#, stem=stem)
        if lcs == 0:
            score = 0.0
        else:
            if mode == 'p':
                score = float(lcs) / len(sourceTextTokens) if len(sourceTextTokens) > 0 else 0.0
            elif mode == 'r':
                score = float(lcs) / len(refTokens) if len(refTokens) > 0 else 0.0
            else:
                precision = float(lcs) / len(sourceTextTokens) if len(sourceTextTokens) > 0 else 0.0
                recall = float(lcs) / len(refTokens) if len(refTokens) > 0 else 0.0
                if precision + recall > 0:
                    score = 2 * (precision * recall) / (precision + recall)
                else:
                    score = 0.0
        scores_per_ref.append(score)

    # the average score over the references:
    score = np.mean(scores_per_ref)

    if normalize and len(sourceTextTokens) > 0:
        score /= len(sourceTextTokens)
    return score

@curry
def compute_delta_rouge_l(output, reference, mode='f', stem=False, remove_stop=False, preceding_text=None,
                          ref_sent_idx=None, normalize=False):
    """
    Compute delta ROUGE-L between preceding_text+output to output against the reference(s). The reference can be a list of references.
    :param output: the additional text (list of list of tokens (sentences)) for which to get the delta score
    :param reference: the reference to evaluate against (list of tokens) or list of references
    :param n: the n-gram size
    :param mode: 'r' 'p' or 'f'
    :param output_max_len: truncate length (0 or lower means no truncation)
    :param preceding_text: the base summary (list of tokens) against which to get the delta score
    :param ref_sent_idx: NOT USABLE
    :param normalize: should the score be normalized by the output length
    :return:
    """
    if preceding_text:
        baseScore = compute_rouge_l(preceding_text, reference, mode=mode, stem=stem, remove_stop=remove_stop)
        fullScore = compute_rouge_l(preceding_text + output, reference, mode=mode, stem=stem, remove_stop=remove_stop)
        score = fullScore - baseScore
    else:
        score = compute_rouge_l(output, reference, mode=mode, stem=stem, remove_stop=remove_stop)

    if normalize and len(output) > 0:
        score /= len(output)
    return score


@curry
def compute_similarity_query_to_text(query_tokens, text_tokens, useSemSim=True, useLexSim=True):
    """
    :param query_tokens: List of tokens of the query
    :param text_tokens: list of tokens for the sentence
    :param useSemSim: should include semantic similarity
    :param useLexSim: should include lexical similarity
    :return:
    """
    # If there's no query, then we consider the similarity to be perfect (since the query should
    # not influence the model's choice of sentence).
    if len(query_tokens) == 0:
        return 1.0
    score = 0.
    num_scores = 0
    if useSemSim:
        query = ' '.join(query_tokens)
        text = ' '.join(text_tokens)
        score += _similarity_sem_score(query, text)
        num_scores += 1
    if useLexSim:
        score += _similarity_lex_score(query_tokens, text_tokens)
        num_scores += 1
    # the average of the scores used:
    return score / num_scores if num_scores > 0 else 0.0


_spacy_nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])
_spacy_rep_cache = {}
def _similarity_sem_score(query, text):
    if text not in _spacy_rep_cache:
        _spacy_rep_cache[text] = _spacy_nlp(text)
    if query not in _spacy_rep_cache:
        _spacy_rep_cache[query] = _spacy_nlp(query)
    return _spacy_rep_cache[text].similarity(_spacy_rep_cache[query])
    #return {'spacy': _spacy_rep_cache[text].similarity(_spacy_rep_cache[query])}

def _similarity_lex_score(query_tokens, text_tokens):
    r1p = compute_rouge_n(query_tokens, [text_tokens], n=1, mode='p', stem=True)
    r2p = compute_rouge_n(query_tokens, [text_tokens], n=2, mode='p', stem=True)
    rlp = compute_rouge_l(query_tokens, [text_tokens], mode='p', stem=True)
    return (r1p + r2p + rlp) / 3.0
    #return {'r1p': r1p, 'r2p': r2p, 'rlp': rlp}