import matplotlib.pyplot as plt
import logging
import evaluation.RougeEvaluator
import time
import threading
from QFSE.Utilities import isPotentialSentence

class SummarizerBase:

    def __init__(self, corpus, evaluateOnTheFly=False, filterSentsPerIteration=False):
        self.corpus = corpus
        # keep a list of the sentences to potentially use (i.e. filter out unwanted sentences):
        self.allSentencesForPotentialSummaries = [sentence for sentence in corpus.allSentences if
                                                  self._isPotentialSentence(sentence)]
        # keep a copy of the potential summary sentence list, in case it's modified in 'forcePotentialSummarySentences':
        self.allSentencesForPotentialSummariesFull = self.allSentencesForPotentialSummaries[:]
        # the query strings used until now
        self.queries = []
        # The sentences we already used until any given moment. The key is the Sentence ID, and the value is the Sentence
        self.usedSentences = {}
        self.usedSentencesText = {} # lowercase compressed version of the used sentences
        # the summaries returned, kept as lists of Sentence IDs (the first is the initial summary, and the rest are per query):
        self.summaries = [] # list of lists of sentenceIds
        self.summariesRating = [] # list of ratings per summary (0 to 1) -- if -1, then unset
        # keep track of the rouge scores after each operation:
        self.rougeScores = []  # list of tuples (wordLength, rougeScores)
        # should the ROUGE be calculated at each summary generation
        self.evaluateOnTheFly = evaluateOnTheFly
        # should sentences be filtered before extracting srntences for a given query
        # (otherwise this is decided by the basic '_isPotentialSentence' functionality:
        self.filterSentsPerIteration = filterSentsPerIteration
        if filterSentsPerIteration:
            self.initFunctionalityForSentenceFiltering()

        self.haveChanges = False
        self.name = 'SummarizerBase'

    def _isPotentialSentence(self, sentence):
        return isPotentialSentence(sentence)
        ## this should be overridden by the inheriting classes
        #return False

    def summarizeGeneric(self, desiredWordCount):
        summaryTextList, summarySentenceIdsList, summaryLengthInWords = self._getGenericSummaryText(desiredWordCount)
        if len(summarySentenceIdsList) > 0:
            self.summaries.append(summarySentenceIdsList)
            self.summariesRating.append(-1)

        if self.evaluateOnTheFly:
            # run a thread that calculates the ROUGE of the accumulated summary until now:
            threadCalculateRouge = threading.Thread(target=self._keepRougeCurrent, args=())
            threadCalculateRouge.start()
        self.haveChanges = True
        return summaryTextList, summaryLengthInWords

    def _getGenericSummaryText(self, desiredWordCount):
        # this should be overridden by the inheriting classes
        return '', [], 0

    def summarizeByQuery(self, query, numSentencesNeeded, queryType):
        if self.filterSentsPerIteration:
            topSents = self.getTopSentences(query)
            self.forcePotentialSummarySentences(topSents)

        summaryTextList, summarySentenceIdsList, summaryLengthInWords = self._getQuerySummaryText(query, numSentencesNeeded)
        #if len(summarySentenceIdsList) > 0:
        # even if the summary is empty, keep it (otherwise there is a sync bug between the iteration and the summaries):
        self.summaries.append(summarySentenceIdsList)
        self.summariesRating.append(-1)

        self.queries.append((query, queryType, time.time()))

        if self.evaluateOnTheFly:
            # run a thread that calculates the ROUGE of the accumulated summary until now:
            threadCalculateRouge = threading.Thread(target=self._keepRougeCurrent, args=())
            threadCalculateRouge.start()
        self.haveChanges = True
        return summaryTextList, summaryLengthInWords

    def _getQuerySummaryText(self, query, numSentencesNeeded=-1, numTokensNeeded=-1):
        # this should be overridden by the inheriting classes
        return '', [], 0

    def _keepRougeCurrent(self):
        logging.info('Computing ROUGE...')
        allTextSoFar = '\n'.join(sentence.text for sentence in self.usedSentences.values())
        allTextNumWords = sum([len(sentence) for sentence in self.usedSentences.values()])
        results = evaluation.RougeEvaluator.getRougeScores(allTextSoFar, self.corpus.referenceSummariesDirectory)
        self.rougeScores.append((allTextNumWords, results))
        self.haveChanges = True


    def _keepRougeAll(self):
        textSoFar = ''
        numWordsSoFar = 0
        for summaryIdx, summary in enumerate(self.summaries):
            textSoFar += '\n'.join(self.usedSentences[sentId].text for sentId in summary)
            numWordsSoFar += sum([len(self.usedSentences[sentId]) for sentId in summary])
            # only calcualte ROUGE for the accumulated summaries not yet evaluated:
            if len(self.rougeScores) <= summaryIdx:
                results = evaluation.RougeEvaluator.getRougeScores(textSoFar, self.corpus.referenceSummariesDirectory)
                self.rougeScores.append((numWordsSoFar, results))
        self.haveChanges = True


    def plotRougeCurves(self):
        if not self.evaluateOnTheFly:
            self._keepRougeAll()

        fig = plt.figure()
        ax = plt.axes()
        ax.set(ylim=(0, 1),
               xlabel='Word Count', ylabel='ROUGE Score',
               title='Incremental Gain per Operation')

        X = []
        Y_R1_Recall = []
        Y_R2_Recall = []
        Y_RL_Recall = []
        Y_R1_Prec = []
        Y_R2_Prec = []
        Y_RL_Prec = []
        Y_R1_F1 = []
        Y_R2_F1 = []
        Y_RL_F1 = []
        for numWords, results in self.rougeScores:
            X.append(numWords)
            Y_R1_Recall.append(results['R1']['recall'])
            Y_R2_Recall.append(results['R2']['recall'])
            Y_RL_Recall.append(results['RL']['recall'])
            Y_R1_Prec.append(results['R1']['precision'])
            Y_R2_Prec.append(results['R2']['precision'])
            Y_RL_Prec.append(results['RL']['precision'])
            Y_R1_F1.append(results['R1']['f1'])
            Y_R2_F1.append(results['R2']['f1'])
            Y_RL_F1.append(results['RL']['f1'])

        plt.plot(X, Y_R1_Recall, '-b', label='R1_rec')
        plt.plot(X, Y_R2_Recall, '-g', label='R2_rec')
        plt.plot(X, Y_RL_Recall, '-r', label='RL_rec')
        plt.plot(X, Y_R1_Prec, '--b', label='R1_prec')
        plt.plot(X, Y_R2_Prec, '--g', label='R2_prec')
        plt.plot(X, Y_RL_Prec, '--r', label='RL_prec')
        plt.plot(X, Y_R1_F1, ':b', linestyle=':', label='R1_f1')
        plt.plot(X, Y_R2_F1, ':g', linestyle=':', label='R2_f1')
        plt.plot(X, Y_RL_F1, ':r', linestyle=':', label='RL_f1')
        plt.legend()
        plt.grid()

        plt.show()

    def getInfoForJson(self, timePointOfReference, suggestedQueriesObject=None):
        '''
        Gets a list of dictionaries for information about the summaries stored here.
        [
            {   'summary':[<sentenceIds>],
                'query':(<query_text>, <query_type>, <when_query_requested>),
                'rouge':(<total_word_length>, {resultsDict})
            }
        ]
        :param timePointOfReference: the time.time() for which to subtract for the query times.
        :return:
        '''
        ## make sure we have the rouge scores evaluated already:
        #if not self.evaluateOnTheFly:
        #    self._keepRougeAll()

        info = []
        for i in range(len(self.summaries)):
            summInfo = {}
            summInfo['summary'] = self.summaries[i] # list of sentence IDs
            if len(self.summaries) == len(self.queries): # in RLMMR case, the initial empty query is kept in the list
                summInfo['query'] = self.getQueryRepresentation(self.queries[i], timePointOfReference) if i > 0 \
                    else ('', 'initial', 0)  # tuple (<query_text>, <query_type>, <when_query_requested>)
            else:
                summInfo['query'] = self.getQueryRepresentation(self.queries[i - 1], timePointOfReference) if i > 0 \
                    else ('', 'initial', 0)  # tuple (<query_text>, <query_type>, <when_query_requested>)
            if suggestedQueriesObject != None:
                suggQList = suggestedQueriesObject.getSuggestionsList(i)
                summInfo['suggestedQueries'] = suggQList if suggQList != None else []
            summInfo['rating'] = round(self.summariesRating[i], 3)
            summInfo['rouge'] = self.rougeScores[i] if i < len(self.rougeScores) else () # tuple of (<total_word_length>, {resultsDict})
            info.append(summInfo)
        return info

    def getQueryRepresentation(self, queryTuple, timePointOfReference):
        return (queryTuple[0], queryTuple[1], queryTuple[2] - timePointOfReference)

    def setIterationRatings(self, iterationRatingsDict):
        if len(iterationRatingsDict) > 0:
            for iterationIdx, rating in iterationRatingsDict.items():
                self.summariesRating[iterationIdx] = rating
            self.haveChanges = True

    def forceSetSnapshot(self, snapshotInfo):
        '''
        Use this function carefully when requiring to force a list of summaries and queries.
        It will overwrite anything kept thus far (including scores if computed)
        snapshotInfo: a list of (query, [sentId list]) tuples.
        The first in the list will likely be an empty query for the initial summary, but not necessarily
        '''
        # keep the summaries (ids)
        self.summaries = [iterationInfo[1] for iterationInfo in snapshotInfo]

        # set the queries list (like in regular cases, the initial summary empty query is not kept):
        if len(snapshotInfo) > 0 and snapshotInfo[0][0] == '': # first query is empty string (likely initial summary)
            self.queries = [(iterationInfo[0], None, None) for iterationInfo in snapshotInfo[1:]]
        else:
            self.queries = [(iterationInfo[0], None, None) for iterationInfo in snapshotInfo]

        # set the used sentences accordingly
        self.usedSentences = {}
        self.usedSentencesText = {}
        for summ in self.summaries:
            self.usedSentences = {sentId: self.corpus.getSentenceById(sentId) for sentId in summ}
        self.usedSentencesText = {sent.textCompressed: sentId for sentId, sent in self.usedSentences.items()}

        # reset scores and anything else kept
        self.summariesRating = []
        self.rougeScores = []
        self.evaluateOnTheFly = False # no more on-the-fly evaluation if it was turned on
        # reset potential summary sentence list to the full list, in case it was modified
        self.allSentencesForPotentialSummaries = self.allSentencesForPotentialSummariesFull[:]


    def forcePotentialSummarySentences(self, sentIds):
        self.allSentencesForPotentialSummaries = [self.corpus.getSentenceById(sentId) for sentId in sentIds]




    def initFunctionalityForSentenceFiltering(self):
        from QFSE.SummarizerRLMMR import SummarizerRLMMR
        self.rlmmrSummarizer = SummarizerRLMMR(self.corpus, evaluateOnTheFly=False)


    def getTopSentences(self, curQueryStr):
        # set the current snapshot:
        queries = [q[0] for q in self.queries]
        if len(self.queries) == len(self.summaries) - 1:
            queries = [''] + queries
        curSnapshotInfo = [(q, s) for q, s in zip(queries, self.summaries)]

        self.rlmmrSummarizer.forceSetSnapshot(curSnapshotInfo)
        # allow a 2 iteration sentence retrieval, so that there are more top sentence options:
        self.rlmmrSummarizer.summarizeByQuery(curQueryStr, 2, None)
        topSentIds = self.rlmmrSummarizer.lastTopSentences
        return topSentIds