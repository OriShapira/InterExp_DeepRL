import sys
sys.path.append('Summarization')
from Summarization.decoding import decode_single_datum
from QFSE.SummarizerBase import SummarizerBase
import torch
from nltk.tokenize import word_tokenize
import time
import math
import os
import json

QRLMMR_MODELS_BASE_DIR = 'Summarization/saved_model'

class SummarizerQRLMMR(SummarizerBase):

    def __init__(self, corpus, modelName, initialSummMaxSentLen, queryResponseMaxSentLen, evaluateOnTheFly=False):
        super().__init__(corpus, evaluateOnTheFly=evaluateOnTheFly, filterSentsPerIteration=False)
        self._initDataForRLMMR()
        self._initRLMMRargs(modelName)
        # The list of most highly scored sentences within the RLMMR system, that was returned in the last generation:
        self.previousQRLMMRSentenceIdxs = []
        self.is_cuda = torch.cuda.is_available()
        self.name = 'SummarizerQRLMMR__' + modelName
        self.max_sent_len_initial_summ = min(initialSummMaxSentLen, 50)
        self.max_sent_len_query_response = min(queryResponseMaxSentLen, 50)

    def _initRLMMRargs(self, modelName):
        self.model_name = modelName
        self.model_dir = os.path.join(QRLMMR_MODELS_BASE_DIR, modelName)
        with open(os.path.join(self.model_dir, 'config.json'), 'r') as fIn:
            model_config_json = json.load(fIn)
        self.beta = model_config_json['beta']
        self.query_encode = model_config_json['query_encoding_in_input']
        self.importance_mode = model_config_json['importance_mode'] if 'importance_mode' in model_config_json else 'tfidf'
        self.diversity_mode = model_config_json['diversity_mode'] if 'diversity_mode' in model_config_json else 'tfidf'

    def _initDataForRLMMR(self):
        self.allDocSentencesForQRLMMR = [] # list of list of list of tokens (doc, sent, tok)
        self.allDocSentencesIdToIdx = {} # sentId -> sentIdx within self.allDocSentencesForQRLMMR
        self.allDocSentencesIdxToId = {} # sentIdx within self.allDocSentencesForQRLMMR -> sentId
        self.allDocSentencesIdxToObjQRLMMR = {}  # QRLMMR sentence index -> sent object
        allDocSentencesIdToText = {}  # sentId -> sentence as passed to QRLMMR
        # prepare the sentence info for QRLMMR
        for docIdx, doc in enumerate(self.corpus.documents):
            for sentIdx, sent in enumerate(doc.sentences):
                sentIdxQRLMMR = (docIdx, sentIdx)
                self.allDocSentencesIdToIdx[sent.sentId] = sentIdxQRLMMR
                self.allDocSentencesIdxToId[sentIdxQRLMMR] = sent.sentId
                self.allDocSentencesIdxToObjQRLMMR[sentIdxQRLMMR] = sent
                sentTextForQRLMMR = word_tokenize(sent.text)
                allDocSentencesIdToText[sent.sentId] = sentTextForQRLMMR
            self.allDocSentencesForQRLMMR.append([allDocSentencesIdToText[sent.sentId] for sent in doc.sentences])

    def _getGenericSummaryText(self, desiredWordCount):
        finalSummaryTxtList, finalSummaryIds, numWordsInSummary = \
            self._getQuerySummaryText(query='', numTokensNeeded=desiredWordCount)
        # QRLMMR expects the empty query for the initial summary as well, so we append it to the
        # beginning of self.queries as if the initial summary is for a query:
        self.queries.append(('', None, time.time()))

        if len(finalSummaryTxtList) == 0:
            finalSummaryTxtList = ['NO INFORMATION TO SHOW.']
            finalSummaryIds = []

        return finalSummaryTxtList, finalSummaryIds, numWordsInSummary


    def _getQuerySummaryText(self, query, numSentencesNeeded=-1, numTokensNeeded=-1):

        if self._noMoreSentences():
            return ["NO MORE INFORMATION."], [], 0

        if len(self.previousQRLMMRSentenceIdxs) == 0 and query == '':
            max_sent_len = self.max_sent_len_initial_summ
        else:
            max_sent_len = self.max_sent_len_query_response

        if numSentencesNeeded <= 0 and numTokensNeeded > 0:
            numSentencesNeeded = math.ceil(numTokensNeeded/(max_sent_len/2))

        # run the QRLMMR summarizer:
        initial_summ_sents, expansion_sents, expansion_sents_indices = \
            decode_single_datum(self.model_dir, self.allDocSentencesForQRLMMR,
                                self.previousQRLMMRSentenceIdxs, [[query, numSentencesNeeded]],
                                self.beta, self.query_encode, self.importance_mode, self.diversity_mode,
                                max_sent_len=max_sent_len, cuda=self.is_cuda)

        self.previousQRLMMRSentenceIdxs.extend(expansion_sents_indices)

        # get the sent objects for the decoded summary, but keep only for the summary length needed:
        summaryLength = 0
        sentencesUsing = []
        for sentQRLMMRIdx in expansion_sents_indices:
            sentUsing = self.allDocSentencesIdxToObjQRLMMR[sentQRLMMRIdx]
            sentencesUsing.append(sentUsing)
            self.usedSentences[sentUsing.sentId] = sentUsing
            self.usedSentencesText[sentUsing.textCompressed] = sentUsing.sentId
            summaryLength += len(sentencesUsing[-1])
            if numTokensNeeded > 0 and summaryLength >= numTokensNeeded:
                break

        return [sent.text for sent in sentencesUsing], [sent.sentId for sent in sentencesUsing], summaryLength

    def _noMoreSentences(self):
        return len(self.usedSentences) == len(self.allSentencesForPotentialSummaries)

    def _sentIdsToIdxs(self, sentIdList):
        return [self.allDocSentencesIdToIdx[sId] for sId in sentIdList]

    def forceSetSnapshot(self, snapshotInfo):
        super().forceSetSnapshot(snapshotInfo)
        # In the case of RLMMR, we do need to keep the first empty query of the initial summary,
        # (which was not kept in the base class).
        if len(snapshotInfo) > 0 and snapshotInfo[0][0] == '':
            self.queries = [('', None, time.time())] + self.queries
        self.previousQRLMMRSentenceIdxs = self._sentIdsToIdxs(self.usedSentences)