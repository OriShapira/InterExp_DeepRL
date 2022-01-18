""" ROUGE utils"""
from collections import Counter, deque
import numpy as np
import spacy
from cytoolz import concat, curry
from nltk.stem.porter import *
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import auc

STOP_WORDS = stopwords.words('english')
STOP_TOKENS = {t:1 for t in STOP_WORDS + list(string.punctuation) + ['``', "''", '--', "'s", "'d", "'ve", "'ll", "'re", "'t"]}
_stemmer = PorterStemmer()
_spacy_nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

_cache_prepared_tokenLists = {} # cache of prepared token lists for prepareTokenList function
_cache_tfidf_vectorizer_references = {} # cache of tfidf vectorizers for the reference summaries
_cache_tf_vectorizer_references = {} # cache of tf vectorizers for the reference summaries


def prepareTokenList(tokenList, stem, remove_stop):
    request_strKey = str(tokenList) + str(stem) + str(remove_stop) # the tokenList raw string + stem and stop requirement representation
    if not request_strKey in _cache_prepared_tokenLists:
        tokenListPrepared = [t.lower() for t in tokenList]
        if remove_stop:
            tokenListPrepared = [t for t in tokenListPrepared if t not in STOP_TOKENS]
        if stem:
            tokenListPrepared = [_stemmer.stem(t) for t in tokenListPrepared]
        _cache_prepared_tokenLists[request_strKey] = tokenListPrepared
    return _cache_prepared_tokenLists[request_strKey]

def prepareTextsForScoringFunc(output_kp, references, initial_kps, preceding_kps, stem, remove_stop):

    output_kpPrepared = prepareTokenList(output_kp, stem, remove_stop)
    initial_kpsPrepared = [prepareTokenList(kp, stem, remove_stop) for kp in initial_kps]
    preceding_kpsPrepared = [prepareTokenList(kp, stem, remove_stop) for kp in preceding_kps]

    # if there aren't several references, make the reference a list of one reference:
    if not isinstance(references[0][0], list):  # a single reference is a list of lists of strings (tokens)
        referencesPrepared = [references]
    else:
        referencesPrepared = references
    referencesPrepared = [[prepareTokenList(sent, stem, remove_stop) for sent in ref] for ref in referencesPrepared]

    return output_kpPrepared, referencesPrepared, initial_kpsPrepared, preceding_kpsPrepared


def score_partial_match(tokens_check, tokens_ref):
    # gets the maximum overlap % of the check list to the ref list, and the earliest relative position where found
    # e.g. score_partial_match(['aa', 'bb', 'd'], ['c', 'aa', 'c', 'd', 'aa']) == 2/3, 1/5
    max_token_match_count = 0
    max_token_match_position = -1
    for i in range(len(tokens_ref) - len(tokens_check) + 1):
        token_match_count = 0
        for j in range(len(tokens_check)):
            if tokens_check[j] == tokens_ref[i + j]:
                token_match_count += 1
        if token_match_count > max_token_match_count:
            max_token_match_count = token_match_count
            max_token_match_position = i

    if len(tokens_check) > 0:
        match_score = float(max_token_match_count) / len(tokens_check)
    else:
        match_score = 0.
        #print('warning: checking empty candidate keyphrase')

    if len(tokens_ref) > 0:
        position_score = float(max_token_match_position) / len(tokens_ref)
    else:
        position_score = 0.
        #print('warning: checking keyphrase against empty reference')

    return match_score, position_score


@curry
def compute_kp_overlap(output_kp, refs_sents, initial_kps, preceding_kps, stem=False, remove_stop=False,
                       weight_initial=0.5, weight_preceding=0.5, position_weighting_exponent=0.1):
                       #length_weighting_exponent_base=1.0):
    """
    Computes the score for the given keyphrase against the reference summaries, given initial KPs and preceding KPs.
    Looks at how many reference summaries it appears in and whether it was already presented previously.
    The score is: max(0, score_refs - (weight_initial * score_initial) - (weight_preceding * score_preceding))
    E.g., if the KP is in 3 of 4 refsumms and appears in the initial KPs and not in the preceding,
    then score_refs=0.75 score_initial=1.0 and score_preceding=0.0 -> final_score=0.25
    :param output_kp: the keyphrase to score (List of tokens)
    :param refs_sents: the reference to evaluate against (list of list of tokens (sentences)) or list of references
    :param initial_kps: information seen by the user (list of list of tokens (kps))
    :param preceding_kps: list of keyphrases already output (list of list of tokens (kps))
    :param stem: should the tokens be stemmed for the scoring procedure
    :param remove_stop: should stop words be removed for the scoring procedure
    :param weight_initial: the weight of the initial_kps score on the overall score
    :param weight_preceding: the weight of the preceding_kps score on the overall score
    :param position_weighting_exponent: this is the exponent used for the relative position, multiplied by the match score
    #:param length_weighting_exponent_base: the base for the weight of the length of the chosen kp, which is raised to the power of the token length.
    :return: Score between 0 and 1 (1 means the KP is a very strong candidate)
    """
    output_kpPrepared, referencesPrepared, initial_kpsPrepared, preceding_kpsPrepared =\
        prepareTextsForScoringFunc(output_kp, refs_sents, initial_kps, preceding_kps, stem, remove_stop)

    # get the KP score against each of the ref summs (ref summ as a list of tokens):
    matchScoresPositionsRefs = [score_partial_match(output_kpPrepared, list(concat(ref))) for ref in referencesPrepared]
    # a score for a ref summ is (match * (1-relative_pos)^0.1), so that the relative position is
    # weighted in lightly for the score (closer to the beginning is better):
    matchScoresRefs = [matchScoreRef * pow((1 - matchPositionRef), position_weighting_exponent)
                       for matchScoreRef, matchPositionRef in matchScoresPositionsRefs]
    score_refs = np.mean(matchScoresRefs)
    #score_refs = score_refs * pow(length_weighting_exponent_base, len(output_kpPrepared))
    #match_score_refs, match_position_refs = np.mean(matchScoresPositionsRefs, axis=0)

    # get the KP score against the initial and preceding KPs (max match score for each):
    score_initial = float(max([score_partial_match(output_kpPrepared, k)[0] for k in initial_kpsPrepared] + [0.]))
    score_preceding = float(max([score_partial_match(output_kpPrepared, k)[0] for k in preceding_kpsPrepared] + [0.]))

    # the final score is the refsScore - (0.5*initialScore) - (0.5*precedingScore)
    final_score = max(0, score_refs - (weight_initial * score_initial) - (weight_preceding * score_preceding))
    return final_score


def _get_tfidf_for_refs(refs_tokens):
    refsStrKey = str(refs_tokens)  # the key for the reference summaries in the cache of TFIDF vectorizers
    if refsStrKey not in _cache_tfidf_vectorizer_references:
        # concatenate the reference summary tokens with spaces, and then set the tokenizer as regular splitter:
        referencesStrs = [' '.join([' '.join(r_sent) for r_sent in r]) for r in refs_tokens]
        refsTfidf = TfidfVectorizer(max_features=50000, tokenizer=lambda s: s.split())
        # vectorize the references and cache the vectorizer:
        refsX = refsTfidf.fit_transform(referencesStrs)
        vocab_scores_dicts = [dict(zip(refsTfidf.get_feature_names(), refX)) for refX in refsX.toarray()]
        _cache_tfidf_vectorizer_references[refsStrKey] = {
            'vectorizer': refsTfidf, 'X': refsX, 'vocab_scores_dicts': vocab_scores_dicts}
    return _cache_tfidf_vectorizer_references[refsStrKey]

def score_with_tfidf(kp_tokens, refs_tokens):
    """
    Gets the TF-IDF score of the keyphrase with respect to the list reference summaries given.
    The average of the kp's tfidf token scores over the reference summaries.
    :param kp_tokens: the list of tokens of the keyphrase to score
    :param refs_tokens: the reference summaries to score against (as a list of list of list of tokens)
    :return: score between 0 and 1
    """
    tfidf_dict_for_refs = _get_tfidf_for_refs(refs_tokens)

    vocab_scores_dicts = tfidf_dict_for_refs['vocab_scores_dicts']
    # the per token score, where each token's score is the average tfidf score over each of the reference summaries:
    kpTokensScores = [np.mean(
        [ref_vocab_scores_dict[kpTok] if kpTok in ref_vocab_scores_dict else 0.
         for ref_vocab_scores_dict in vocab_scores_dicts])
        for kpTok in kp_tokens]
    return np.mean(kpTokensScores)

def most_similar_with_tfidf(kp_tokens, other_kp_tokens, refs_tokens):
    """
    Gets the TF-IDF cosine similarity between the given kp and the list of other keyphrases with respect to the
    list reference summaries given. Returns the most similar score to all other kps.
    :param kp_tokens: the list of tokens of the keyphrase to compare
    :param other_kp_tokens: the list of kps to find similarity against (list of list of tokens)
    :param refs_tokens: the list of list of tokens of the reference summaries to score against
    :return: cosine similarity score between 0 and 1
    """
    if len(other_kp_tokens) == 0:
        return 0.

    tfidf_dict_for_refs = _get_tfidf_for_refs(refs_tokens)

    # convert kps to strings (concat with spaces):
    kp_str = ' '.join(kp_tokens)
    other_kps_strs = [' '.join(kp_t) for kp_t in other_kp_tokens]
    # vectorize the keyphrases, the first being the main kp:
    kps_vec = tfidf_dict_for_refs['vectorizer'].transform([kp_str] + other_kps_strs)
    # get the cossim between the main kp and all kps:
    sim_scores = cosine_similarity(kps_vec[0], kps_vec).squeeze()
    # return the score of the kp that is most similar to the main kp (the main KP sim to itself is at index 0):
    return max(sim_scores[1:])


@curry
def compute_kp_tfidf(output_kp, refs_sents, initial_kps, preceding_kps, stem=False, remove_stop=False,
                       weight_initial=0.5, weight_preceding=0.5):
    """
    Computes the score for the given keyphrase against the reference summaries, given initial KPs and preceding KPs.
    Looks at the tf-idf score of the kp in the reference summaries, and whether it was already presented previously.
    The score is between 0 and 1.
    :param output_kp: the keyphrase to score (List of tokens)
    :param refs_sents: the reference to evaluate against (list of list of tokens (sentences)) or list of references
    :param initial_kps: information seen by the user (list of list of tokens (kps))
    :param preceding_kps: list of keyphrases already output (list of list of tokens (kps))
    :param stem: should the tokens be stemmed for the scoring procedure
    :param remove_stop: should stop words be removed for the scoring procedure
    :param weight_initial: the weight of the initial_kps score on the overall score
    :param weight_preceding: the weight of the preceding_kps score on the overall score
    :return:
    """
    output_kpPrepared, referencesPrepared, initial_kpsPrepared, preceding_kpsPrepared =\
        prepareTextsForScoringFunc(output_kp, refs_sents, initial_kps, preceding_kps, stem, remove_stop)

    # get the KP score against the ref summs (ref summ as a list of tokens):
    score_refs = score_with_tfidf(output_kpPrepared, referencesPrepared)

    # get the KP score against the initial and preceding KPs (max similarity score for each):
    score_initial = most_similar_with_tfidf(output_kpPrepared, initial_kpsPrepared, referencesPrepared)
    score_preceding = most_similar_with_tfidf(output_kpPrepared, preceding_kpsPrepared, referencesPrepared)

    # the final score is the refsScore - (0.5*initialScore) - (0.5*precedingScore)
    final_score = max(0, score_refs - (weight_initial * score_initial) - (weight_preceding * score_preceding))
    return final_score




def _get_tf_for_refs(refs_tokens):
    refsStrKey = str(refs_tokens)  # the key for the reference summaries in the cache of TFIDF vectorizers
    if refsStrKey not in _cache_tf_vectorizer_references:
        # concatenate the reference summary tokens with spaces, and then set the tokenizer as regular splitter:
        referencesStrs = [' '.join([' '.join(r_sent) for r_sent in r]) for r in refs_tokens]
        referencesLens = [len(list(concat(r))) for r in refs_tokens]
        refsTf = CountVectorizer(max_features=50000, tokenizer=lambda s: s.split())
        # vectorize the references and cache the vectorizer:
        refsX = refsTf.fit_transform(referencesStrs)
        vocab_scores_dicts = [dict(zip(refsTf.get_feature_names(), refX)) for refX in refsX.toarray()]
        _cache_tf_vectorizer_references[refsStrKey] = {
            'vectorizer': refsTf, 'X': refsX, 'vocab_scores_dicts': vocab_scores_dicts, 'summ_lens': referencesLens}
    return _cache_tf_vectorizer_references[refsStrKey]

def score_with_tf(kp_tokens, refs_tokens):
    """
    Gets the TF score of the keyphrase with respect to the list reference summaries given.
    The average of the kp's tf token scores over the reference summaries.
    :param kp_tokens: the list of tokens of the keyphrase to score
    :param refs_tokens: the reference summaries to score against (as a list of list of list of tokens)
    :return: score between 0 and 1
    """
    tf_dict_for_refs = _get_tf_for_refs(refs_tokens)

    vocab_scores_dicts = tf_dict_for_refs['vocab_scores_dicts']
    # the per token score, where each token's score is the average tf score over each of the reference summaries:
    kpTokensScores = [np.mean(
        [ref_vocab_scores_dict[kpTok] / float(tf_dict_for_refs['summ_lens'][refIdx])
         if kpTok in ref_vocab_scores_dict else 0.
         for refIdx, ref_vocab_scores_dict in enumerate(vocab_scores_dicts)])
        for kpTok in kp_tokens]
    return np.mean(kpTokensScores)


def most_similar_with_tf(kp_tokens, other_kp_tokens, refs_tokens):
    """
    Gets the TF cosine similarity between the given kp and the list of other keyphrases with respect to the
    list reference summaries given. Returns the most similar score to all other kps.
    :param kp_tokens: the list of tokens of the keyphrase to compare
    :param other_kp_tokens: the list of kps to find similarity against (list of list of tokens)
    :param refs_tokens: the list of list of tokens of the reference summaries to score against
    :return: cosine similarity score between 0 and 1
    """
    if len(other_kp_tokens) == 0:
        return 0.

    tf_dict_for_refs = _get_tf_for_refs(refs_tokens)

    # get the intersections between the main kp and each of the other kps:
    matchings = []
    for o_kp_tokens in other_kp_tokens:
        token_intersection = list(set(o_kp_tokens) & set(kp_tokens)) # intersecting tokens
        if len(token_intersection) > 0:
            matchings.append(token_intersection)

    # get the score for each token intersection:
    vocab_scores_dicts = tf_dict_for_refs['vocab_scores_dicts']
    matchingsScores = []
    for match in matchings:
        # the tf score of the intersecting tokens:
        matchTokensScores = [np.mean([ref_vocab_scores_dict[matchToken] / float(tf_dict_for_refs['summ_lens'][refIdx])
                                      if matchToken in ref_vocab_scores_dict else 0.
                                      for refIdx, ref_vocab_scores_dict in enumerate(vocab_scores_dicts)])
                             for matchToken in match]
        # the average token match score for the main kp (for tokens not intersecting, it's like a score of 0 for that token):
        matchingsScores.append(sum(matchTokensScores)/len(kp_tokens))

    return max(matchingsScores + [0.])


@curry
def compute_kp_tf(output_kp, refs_sents, initial_kps, preceding_kps, stem=False, remove_stop=False,
                  weight_initial=0.5, weight_preceding=0.5):
    """
    Computes the score for the given keyphrase against the reference summaries, given initial KPs and preceding KPs.
    Looks at the term frequency score of the kp in the reference summaries, and whether it was already presented previously.
    The score is between 0 and 1.
    :param output_kp: the keyphrase to score (List of tokens)
    :param refs_sents: the reference to evaluate against (list of list of tokens (sentences)) or list of references
    :param initial_kps: information seen by the user (list of list of tokens (kps))
    :param preceding_kps: list of keyphrases already output (list of list of tokens (kps))
    :param stem: should the tokens be stemmed for the scoring procedure
    :param remove_stop: should stop words be removed for the scoring procedure
    :param weight_initial: the weight of the initial_kps score on the overall score
    :param weight_preceding: the weight of the preceding_kps score on the overall score
    :return:
    """
    output_kpPrepared, referencesPrepared, initial_kpsPrepared, preceding_kpsPrepared =\
        prepareTextsForScoringFunc(output_kp, refs_sents, initial_kps, preceding_kps, stem, remove_stop)

    # get the KP score against the ref summs (ref summ as a list of tokens):
    score_refs = score_with_tf(output_kpPrepared, referencesPrepared)

    # get the KP score against the initial and preceding KPs (max similarity score for each):
    score_initial = most_similar_with_tf(output_kpPrepared, initial_kpsPrepared, referencesPrepared)
    score_preceding = most_similar_with_tf(output_kpPrepared, preceding_kpsPrepared, referencesPrepared)

    # the final score is the refsScore - (0.5*initialScore) - (0.5*precedingScore)
    final_score = max(0, score_refs - (weight_initial * score_initial) - (weight_preceding * score_preceding))
    return final_score




def _rankKPs(kps):
    """
    Returns a new list of KPs with rank instead of score. A repeated score gives the same rank.
    Assumes KPs are ordered by score in descending order.
    :param kps: List of tuples (<kp>,<score>) in descending order of score.
    :return: List of tuples (<kp>,<rank>) in ascending order of rank.
    """
    kpsWithRank = []
    lastScore = 99999
    lastRank = -1
    for (kpText, kpScore) in kps:
        if kpScore < lastScore: # only advance the rank when a lower score is found (same score = same rank)
            lastScore = kpScore
            lastRank += 1
        elif kpScore > lastScore:
            raise Exception('Error: The given KPs list is not ordered by descending score.')
        kpsWithRank.append((kpText, lastRank))
    return kpsWithRank

def _cutKpsAtK(kps, k, hardCut):
    if hardCut or k >= len(kps): # don't consider repeating rank, just cut at k
        cutIdx = k
    else:
        i = 0
        while i < len(kps): # go until the rank is actually found (so KPs with the same rank are kept)
            if kps[i][1] == k:
                break
            i += 1
        cutIdx = i
    return [kpTxt for (kpTxt, kpRank) in kps[:cutIdx]] # return just a list of KPs without ranks anymore

def _getMatchCount(kpsPredRanked, kpsGoldRanked, cutoffPred, cutoffGold): #, match_fn):
    """
    Returns the number of KPs in kpsPred that are in kpsGold (both in the top <cutoff> ranking KPs).
    If KPs rank the same in kpsGold, then any of those can be used for the match.
    The kpsPred are cutoff hard, so just the top-k are kept, regardless of possible repeated ranks.
    :param kpsPredRanked: List of tuples (<kp_text>,<rank>) in ascending order of rank.
    :param kpsGoldRanked: List of tuples (<kp_text>,<rank>) in ascending order of rank.
    :param cutoffPred: Using the top-k ranked KPs for the Pred KPs (hard cut)
    :param cutoffGold: Using the top-k ranked KPs for the Gold KPs (soft cut by repeating rank)
    :return:
    """
    # prepare the lists of KPs:
    kpsPred = _cutKpsAtK(kpsPredRanked, cutoffPred, hardCut=True)  # do not allow KPs with same rank, just cut at k
    # in the gold, allow KPs with same rank to stay
    kpsGold = _cutKpsAtK(kpsGoldRanked, cutoffGold, hardCut=False)

    # find number of matches (number of elements in their intersection):
    # remove redundant predicted KPs after cutting at k so that we don't count a match more than once
    # there shouldn't be repetitions in the gold, but just in case
    matchCount = len(set(kpsPred) & set(kpsGold))

    return matchCount, len(kpsPred), len(kpsGold)

def _getMatchCountUnigram(kpsPredRanked, kpsGoldRanked, cutoffPred, cutoffGold): #, match_fn):
    """
    Returns the number of unigrams from kpsPred that are in kpsGold (both in the top <cutoff> ranking KPs).
    If KPs rank the same in kpsGold, then any of those can be used for the match.
    The kpsPred are cutoff hard, so just the top-k are kept, regardless of possible repeated ranks.
    :param kpsPredRanked: List of tuples (<kp_text>,<rank>) in ascending order of rank.
    :param kpsGoldRanked: List of tuples (<kp_text>,<rank>) in ascending order of rank.
    :param cutoffPred: Using the top-k ranked KPs for the Pred KPs (hard cut)
    :param cutoffGold: Using the top-k ranked KPs for the Gold KPs (soft cut by repeating rank)
    :return:
    """
    # prepare the lists of KPs:
    kpsPred = _cutKpsAtK(kpsPredRanked, cutoffPred, hardCut=True)  # do not allow KPs with same rank, just cut at k
    # in the gold, allow KPs with same rank to stay
    kpsGold = _cutKpsAtK(kpsGoldRanked, cutoffGold, hardCut=False)

    # get bag of unigrams from the non-redundant lists:
    # (remove redundant predicted KPs after cutting at k so that we don't count a match more than once)
    kpsUniquePredUnigrams = list(concat([kp.split() for kp in list(set(kpsPred))]))
    kpsUniqueGoldUnigrams = list(concat([kp.split() for kp in list(set(kpsGold))]))

    # find number of matches (number of common unigrams with possible repeating ones in their intersection):
    unigramMatchCount = len(list((Counter(kpsUniquePredUnigrams) & Counter(kpsUniqueGoldUnigrams)).elements()))

    numUnigramsInKpsPred = len(list(concat([kp.split() for kp in kpsPred])))
    numUnigramsInKpsGold = len(list(concat([kp.split() for kp in kpsGold])))

    return unigramMatchCount, numUnigramsInKpsPred, numUnigramsInKpsGold


def evaluateKpList(kpsPred, kpsGold, cutoffs=[1,5,10,15,20], cutGold=False): #match_fn=kps_match_exact
    """
    Score the given predicted KPs against the gold KPs.
    Currently with F1@k scores.
    :param kpsPred: list of tuples (<kp_text>,<kp_score>)
    :param kpsGold: list of tuples (<kp_text>,<kp_score>)
    :param cutoffs: list of ks for the F1@k score
    :param cutGold: for the F1@k score, should the gold list be cut as well (default false)
    :return: A dictionary of scores {<score_name>: <score>}, currently f1@k scores
    """
    # recreate the list of (kp,score) tuples with stemmed KPs:
    kpsPredStem = [(' '.join([_stemmer.stem(t) for t in kp_text.split()]), kp_score) for (kp_text, kp_score) in kpsPred]
    kpsGoldStem = [(' '.join([_stemmer.stem(t) for t in kp_text.split()]), kp_score) for (kp_text, kp_score) in kpsGold]

    # recreate the list of (kp,score) tuples with ranks instead of scores:
    kpsPredWithRanks = _rankKPs(kpsPredStem)
    kpsGoldWithRanks = _rankKPs(kpsGoldStem)

    all_scores = {} # metric -> score

    # compute f1@k scores
    for cutoff in cutoffs:
        if cutGold:
            matchCount, predCount, goldCount = _getMatchCount(kpsPredWithRanks, kpsGoldWithRanks, cutoff, cutoff)
        else:
            matchCount, predCount, goldCount = _getMatchCount(kpsPredWithRanks, kpsGoldWithRanks, cutoff, 1000)
        precision = matchCount / predCount
        recall = matchCount / goldCount
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.
        all_scores[f'rec@{cutoff}'] = recall
        all_scores[f'prec@{cutoff}'] = precision
        all_scores[f'f1@{cutoff}'] = f1

    # compute f1@k scores on the combined unigram level:
    for cutoff in cutoffs:
        if cutGold:
            matchCount, predCount, goldCount = _getMatchCountUnigram(kpsPredWithRanks, kpsGoldWithRanks, cutoff, cutoff)
        else:
            matchCount, predCount, goldCount = _getMatchCountUnigram(kpsPredWithRanks, kpsGoldWithRanks, cutoff, 1000)
        precision = matchCount / predCount
        recall = matchCount / goldCount
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.
        all_scores[f'rec_unigram@{cutoff}'] = recall
        all_scores[f'prec_unigram@{cutoff}'] = precision
        all_scores[f'f1_unigram@{cutoff}'] = f1

    return all_scores





@curry
def compute_partial_phrase_overlap_score(output_kp, ref_summs_sents, initial_kps, preceding_kps,
                                         stem=False, remove_stop=False,
                                         mmr_lambda=0.5, redundancy_alpha=0.5):
    """
    Computes the score for the given keyphrase against the reference summaries, given an initial summary
    and preceding KPs. Considers its overlap in the reference summaries, preceding KPs and initial summary.
    score = max(0., (mmr_lambda * refScore) - ((1 - mmr_lambda) * max(prevScore, initialScore)))
    where refScore = (avg over refs) the partial-match overlap score of pred in ref, that gives the highest coverage within ref
          prevScore = a weighted sum (redundancy_alpha) of the max pred overlap in the prev list and the overall pred token coverage in prev list
          initialScore = a weighted sum (redundancy_alpha) of the max pred overlap in the initial summary and the overall pred token coverage in the initial summary
    :param output_kp: the keyphrase to score (List of tokens)
    :param ref_summs_sents: the reference to evaluate against (list of list of tokens (sentences)) or list of references
    :param initial_kps: information seen by the user (list of list of tokens (kps))
    :param preceding_kps: list of keyphrases already output (list of list of tokens (kps))
    :param stem: should the tokens be stemmed for the scoring procedure
    :param remove_stop: should stop words be removed for the scoring procedure
    :param mmr_lambda: the weight of the initial_kps score on the overall score
    :param redundancy_alpha: the weight of the preceding_kps score on the overall score
    :return: Score between 0 and 1 (the higher the better)
    """
    output_kpPrepared, referencesPrepared, initial_kpsPrepared, preceding_kpsPrepared = \
        prepareTextsForScoringFunc(output_kp, ref_summs_sents, initial_kps, preceding_kps, stem, remove_stop)

    # the stop word removal may have diminished the output_kp to an empty list:
    if len(output_kpPrepared) == 0:
        return 0.

    referencesPrepared_flat = [[tok for sent in ref for tok in sent] for ref in referencesPrepared]
    refsScores = [_match_overlap_score(output_kpPrepared, ref_tokens) for ref_tokens in referencesPrepared_flat]
    refScore = np.mean(refsScores)
    prevScore = _match_list_overlap_score(output_kpPrepared, preceding_kpsPrepared, weightAlphaParam=redundancy_alpha)
    initialScore = _match_list_overlap_score(output_kpPrepared, initial_kpsPrepared, weightAlphaParam=redundancy_alpha)

    finalKpScore = max(0., (mmr_lambda * refScore) - ((1 - mmr_lambda) * max(prevScore, initialScore)))
    return finalKpScore

def compute_partial_phrase_overlap_list_auc_score(output_kps, ref_summs_sents, initial_summ,
                                                  stem=False, remove_stop=False,
                                                  mmr_lambda=0.5, redundancy_alpha=0.5):
    _, referencesPrepared, initialPrepared, _ = \
        prepareTextsForScoringFunc([], ref_summs_sents, [initial_summ], [], stem, remove_stop)
    output_kpsPrepared = [prepareTokenList(kp, stem, remove_stop) for kp in output_kps]
    # note that initial_summ is sent within a list to be pre-processed since the prep function expects a list of
    # initial kps, while we have a single long text here. The initialPrepared is then also returned as a list (with
    # one item with a list of preped tokens). We can then send it to _match_list_overlap_score already as a list, as
    # expected by _match_list_overlap_score.

    kps_scores = []
    for output_kp_idx, output_kp_tokens in enumerate(output_kpsPrepared):
        refsScores = [_match_overlap_score(output_kp_tokens, ref_tokens) for ref_tokens in referencesPrepared]
        refScore = np.mean(refsScores)
        prevScore = _match_list_overlap_score(output_kp_tokens, output_kpsPrepared[:output_kp_idx], weightAlphaParam=redundancy_alpha)
        initialScore = _match_list_overlap_score(output_kp_tokens, initialPrepared, weightAlphaParam=redundancy_alpha)

        finalKpScore = max(0., (mmr_lambda * refScore) - ((1 - mmr_lambda) * max(prevScore, initialScore)))
        kps_scores.append(finalKpScore)

    acummulatingKpsScores = [0.]
    for kpScore in kps_scores:
        acummulatingKpsScores.append(acummulatingKpsScores[-1] + kpScore)
    scoreAUC = auc(acummulatingKpsScores, [i for i in range(len(acummulatingKpsScores))])

    return scoreAUC



class OverlapMatch:
    def __init__(self, spanOffsetStart, spanOffsetEnd, matchSpanTokens, matchPercent):
        self.span_offset_start = spanOffsetStart # the begin index of the match in the reference token list
        self.span_offset_end = spanOffsetEnd # the end index of the match in the reference token list
        self.match_span_tokens = matchSpanTokens # the match tokens in the ref the have the overlap (start to end indices)
        self.match_percent = matchPercent # the percentage of tokens of pred that overlap in ref
    # Example: for pred ['a', 'b', 'c'] and ref ['c', 'a', 'b', 'b', 'a'], a possible match could be:
    # span_offset_start, span_offset_end = 1, 2
    # match_span_tokens = ['a', 'b']
    # match_percent = 0.666


def _find_matches(testTokens, refTokens):
    """
    Gets a list of partial-phrase-matches of testTokens in refTokens.
    Example: _find_matches(['a', 'b', 'c'], ['c', 'a', 'b', 'b', 'a']) =
             (0, 0), ['c'], 0.3333333333333333
             (1, 2), ['a', 'b'], 0.6666666666666666
             (3, 3), ['b'], 0.3333333333333333
             (4, 4), ['a'], 0.3333333333333333
    :param testTokens: List of tokens to test
    :param refTokens: list of tokens in which to find the overlap
    :return: list of OverlapMatch objects
    """
    # pad the reference before and after with garbage tokens to allow partial overlap at the beginning and end as well:
    padOffset = len(testTokens) - 1
    refTokensPadded = ['|-|'] * padOffset + refTokens + ['|-|'] * padOffset

    # find the partial matches of testTokens in refTokens:
    matches = []
    for i in range(len(refTokensPadded) - len(testTokens) + 1):
        token_match_count = 0
        token_match_idxs = []
        for j in range(len(testTokens)):
            if testTokens[j] == refTokensPadded[i + j]:
                token_match_count += 1
                token_match_idxs.append(i+j)
        if token_match_count > 0:
            matchLocation = (token_match_idxs[0] - padOffset, token_match_idxs[-1] - padOffset)
            matches.append(OverlapMatch(matchLocation[0], matchLocation[1],
                                        refTokensPadded[token_match_idxs[0]: token_match_idxs[-1]+1],
                                        float(token_match_count) / len(testTokens)))

    # return a list of OverlapMatch objects:
    return matches


def _match_overlap_score(predTokens, refTokens, actualRefLen=-1):
    """
    Gets the partial-match overlap score of pred in ref, that gives the highest coverage within ref.
    :param predTokens: list of tokens to test
    :param refTokens: list of tokens to test in
    :param actualRefLen: if the refTokens given was modified before being sent here, provide the original token length,
    to be used for computing the relative coverage of ref with the real length. Otherwise don't send this value.
    :return: the best overlap score (product of the average pred overlap and the ref coverage)
    """
    _allMatchPathsDict = {}  # keep for caching in _build_best_match_path (dynamic programming for recursion)
    def _build_best_match_path(matches, baseMatchesIdx):
        """
        Recursivelly compute all match paths and return the best one.
        A match path is a subset of matches that do not overlap within themselves.
        :param matches: list of OverlapMatch objects (as generated by the _find_matches function)
        :param baseMatchesIdx: since this is a recursive function, the actual index from which the matches list starts
        :return: bestOverlapPath list, bestMatchPathScore float
        """
        # if there are no matches, just return empty:
        if len(matches) == 0:
            return [], 0.

        # check if we already calculated the subpaths of the given matches list (dynamic programming):
        if baseMatchesIdx in _allMatchPathsDict:
            return _allMatchPathsDict[baseMatchesIdx]

        # compute all the possible paths of the given option list:
        allPaths = [[matches[0]]]  # all the ordered paths available from the options (non-overlapping ngram matches)
        allPathsScores = [matches[0].match_percent]
        indicesChecked = []  # keep track of subpaths tried, so we don't go down that path again recursively
        for i in range(len(matches)):
            if i in indicesChecked:
                continue
            for j in range(i + 1, len(matches)):
                # if the j match does not overlap with the i match, compute that subpath:
                if matches[i].span_offset_end < matches[j].span_offset_start:
                    bestSubpath, bestSubpathScore = _build_best_match_path(matches[j:], baseMatchesIdx + j)
                    # add the best subpath with the current option appended at the beginning of the path
                    allPaths.append([matches[i]] + bestSubpath)
                    allPathsScores.append(matches[i].match_percent + bestSubpathScore)
                    indicesChecked.append(j)  # no need to recompute this subpath again

        # send back only the best scoring path from the given options:
        maxScoreIdx = allPathsScores.index(max(allPathsScores))
        bestMatchPath, bestMatchPathScore = allPaths[maxScoreIdx], allPathsScores[maxScoreIdx]
        _allMatchPathsDict[baseMatchesIdx] = (bestMatchPath, bestMatchPathScore)  # keep in DP cache
        return bestMatchPath, bestMatchPathScore

    # get all possible matches between the pred and the ref:
    matches = _find_matches(predTokens, refTokens)
    # get the non-overlapping match combination that gives the most overall overlap of pred in ref:
    bestMatchPath, bestMatchPathScore = _build_best_match_path(matches, 0)
    # the average overlap percentage of all matches in the best overall match combination:
    avgPercentCoverageOfPred = float(bestMatchPathScore) / len(bestMatchPath) if len(bestMatchPath) > 0 else 0.
    # the percent of tokens in ref that are covered by partial-phrase-matches with pred:
    refLen = len(refTokens) if actualRefLen < 0 else actualRefLen
    percentCoverageRef = bestMatchPathScore * len(predTokens) / refLen if refLen > 0 else 0. # the partial-phrase-frequency
    # The final score is the product of the average pred overlap and the ref coverage.
    # We weight the ppf with the avgPredCoverage so that a predicted phrase that appears more wholly is given
    # a better score, while phrases that appear more in parts is weakened (i.e. reward for more fully matching phrases.)
    # So if a long pred phrase covers more of ref, but appears more in parts, but a shorter pred phrase covers less of
    # ref but fully overlaps its matches, they may recieve an equal chance.
    # This is a form of token recall (cover more words of ref) and precision (use more full-matching pred phrase) balancing.
    #finalScore = avgPercentCoverageOfPred * percentCoverageRef
    if avgPercentCoverageOfPred + percentCoverageRef > 0:
        finalScore = (2 * avgPercentCoverageOfPred * percentCoverageRef) / (avgPercentCoverageOfPred + percentCoverageRef)
    else:
        finalScore = 0.
    return finalScore


def _match_list_overlap_score(predTokens, prevList, weightAlphaParam=0.5):
    """
    Gets a match score of pred phrase in the list of previous phrases.
    The score is a weighted sum of the max pred overlap in the list and the overall pred token coverage.
    Repeated tokens are considered once.
    Example: for pred ['a','b','c','d'], if the best overlap is ['a','g','c'] in prev, and 'b' also appears somewhere
    in prev, then the final score will be 0.5*0.75.
    The prevList can also be a single longer list of tokens if the same metric is needed against a long text.
    :param predTokens: list of tokens to test for
    :param prevList: list of list of tokens to test against
    :param weightParam: the weight of the maxOverlapPercent against predCoverage (alpha and 1-alpha)
    :return: score of pred in the list of prev (a weighted sum of the max pred overlap in the list
    and the overall pred token coverage)
    """
    uniquePredTokens = set(predTokens)  # just coverage without repetition (it's enough to be covered once)
    # go over all the previous phrases and keep the maximum overlapping phrase:
    predTokensCovered = {} # keep track of pred tokens appearing in all of prev list (once is enough to be covered)
    maxPredOverlapPercent = 0
    for prevTokens in prevList:
        matches = _find_matches(predTokens, prevTokens) # all matches of pred in prev
        # find the maximum overlapping match of pred in prev:
        for match in matches:
            for tok in match.match_span_tokens:
                if tok in uniquePredTokens:
                    predTokensCovered[tok] = 1
            if match.match_percent > maxPredOverlapPercent:
                maxPredOverlapPercent = match.match_percent
        # if we have already covered all tokens of pred and reached a maximum possible overlap, then no need to keep going:
        if len(predTokensCovered) == len(uniquePredTokens) and maxPredOverlapPercent == 1.0:
            break

    # compute how many of the tokens in the pred phrase were covered overall in prevList, regardless of phrase overlap:
    predCoverageCount = len(predTokensCovered) #len([t for t in uniquePredTokens if t in predTokensCovered]) # how many unique tokens of pref are in all of prevList
    if len(uniquePredTokens) > 0:
        predCoveragePercent = float(predCoverageCount) / len(uniquePredTokens) # unique coverage percentage
    else:
        predCoveragePercent = 0.
        print(f'Warning: Empty predTokens in _match_list_overlap_score "{predTokens}"')
    # print(maxOverlapPercent, predCoverage)

    if weightAlphaParam >= 0:
        # the final score is a weighted sum of the max overlap and the overall pred token coverage:
        finalScore = (weightAlphaParam * maxPredOverlapPercent) + ((1 - weightAlphaParam) * predCoveragePercent)
    else:
        # harmonic mean
        if (predCoveragePercent + maxPredOverlapPercent) > 0:
            finalScore = (2 * predCoveragePercent * maxPredOverlapPercent) / (predCoveragePercent + maxPredOverlapPercent)
        else:
            finalScore = 0.

    return finalScore