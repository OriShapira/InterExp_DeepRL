import sys
sys.path.append('SuggestedQueries')
from SuggestedQueries.decoding import decode_single_datum
from SuggestedQueries.dataset_k.data_manage import getPhrasesInDocset
from QFSE.SuggestedQueriesBase import SuggestedQueriesBase
from nltk.tokenize import word_tokenize
import torch
import os

KRLMMR_MODELS_BASE_DIR = 'SuggestedQueries/saved_model'

class SuggestedQueriesKRLMMR(SuggestedQueriesBase):

    def __init__(self, corpus, modelName):
        super().__init__(corpus)
        self._initDataForKRLMMR()
        self.model_name = modelName
        self.model_dir = os.path.join(KRLMMR_MODELS_BASE_DIR, modelName)
        # The list of most highly scored sentences within the RLMMR system, that was returned in the last generation:
        self.previousQRLMMRSentenceIdxs = []
        self.is_cuda = torch.cuda.is_available()
        self.name = 'SuggestedQueriesKRLMMR__' + modelName

    def _initDataForKRLMMR(self):
        # TODO: need to initialize sentences as keyphrases like in DataManage, so that everything is processed at the KP level in this class.

        self.allDocSentTokensForKRLMMR = [] # list of list of list of tokens (doc, sent, tok)
        self.allDocSentencesIdToIdx = {} # sentId -> [list of kpIdx] within self.allDocKpsForKRLMMR
        #self.allDocSentencesIdxToId = {} # sentIdx within self.allDocSentencesForKRLMMR -> sentId
        #self.allDocSentencesIdxToObjKRLMMR = {}  # KRLMMR sentence index -> sent object
        allDocSentencesIdToText = {}  # sentId -> sentence as passed to KRLMMR
        allDocSents = []  # list of list strs (doc, sent)
        # prepare the sentence info for KRLMMR
        for docIdx, doc in enumerate(self.corpus.documents):
            for sentIdx, sent in enumerate(doc.sentences):
                sentIdxKRLMMR = (docIdx, sentIdx)
                self.allDocSentencesIdToIdx[sent.sentId] = sentIdxKRLMMR
                #self.allDocSentencesIdxToId[sentIdxKRLMMR] = sent.sentId
                #self.allDocSentencesIdxToObjKRLMMR[sentIdxKRLMMR] = sent
                sentTextForKRLMMR = word_tokenize(sent.text)
                allDocSentencesIdToText[sent.sentId] = sentTextForKRLMMR
            self.allDocSentTokensForKRLMMR.append([allDocSentencesIdToText[sent.sentId] for sent in doc.sentences])
            allDocSents.append([sent.text for sent in doc.sentences])

        self.allDocPhrasesForKRLMMR, self.topicSentIdToPhrasesMappings = self._convertDocsSentsToPhrases(allDocSents)
        self.allDocPhrasesTokensForKRLMMR = [[phrase.split() for phrase in doc] for doc in self.allDocPhrasesForKRLMMR]

    def _getTopSuggestions(self, numKeywordsNeeded, presentedSentsSoFar=[], getScores=False):
        # get the next numKeywords keywords:

        # convert the sentences IDs to the KRLMMR sentence indices to KRLMMR phrase indices:
        initialSummSentIdxs = [self.allDocSentencesIdToIdx[prevSentId] for prevSentId in presentedSentsSoFar]
        initialSummPhraseIdxs = self._convertInitialSummToPhrases(initialSummSentIdxs)

        # run the KRLMMR model with the docs and initial summary:
        chosen_summ_kps, chosen_summ_scores, initial_summ_kps = \
            decode_single_datum(self.model_dir, self.allDocPhrasesTokensForKRLMMR,
                                initialSummPhraseIdxs, numKeywordsNeeded, cuda=self.is_cuda)

        # if not needed, remove the scores from the list:
        if getScores:
            chosen_summ_kps = list(zip(chosen_summ_kps, chosen_summ_scores))

        return chosen_summ_kps

    def _convertDocsSentsToPhrases(self, docs):
        topic_documents_phrases, topic_sentid_to_phrases_mappings = \
            getPhrasesInDocset(docs, None, max_phrase_len=999, phrasing_method='posregex', make_single_document=False)
        return topic_documents_phrases, topic_sentid_to_phrases_mappings

    def _convertInitialSummToPhrases(self, initial_summary_sent_indices):
        initial_summary_phrase_indices = []
        for sentId in initial_summary_sent_indices:
            initial_summary_phrase_indices.extend(self.topicSentIdToPhrasesMappings[tuple(sentId)])
        return initial_summary_phrase_indices