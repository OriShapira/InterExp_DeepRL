from QFSE.Sentence import Sentence
import operator
from rouge import Rouge
from nltk.stem.porter import *
from functools import reduce
from nltk.tokenize import word_tokenize
from QFSE.SummarizerBase import SummarizerBase
from QFSE.Utilities import isPotentialSentence
from QFSE.Utilities import nlp

QUERY_DOC_ALIAS = '_QUERY_'

class SummarizerTextRankPlusLexical(SummarizerBase):

    def __init__(self, corpus, evaluateOnTheFly=False, filterSentsPerIteration=False):
        super().__init__(corpus, evaluateOnTheFly=evaluateOnTheFly, filterSentsPerIteration=filterSentsPerIteration)
        self.lastSentenceIndexFromGeneric = -1
        self.stemmedSentences = {} # sentId -> stemmed sentence text
        #self.rouge = Rouge()
        self.rouge = Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, apply_avg=False, apply_best=False)
        self.stemmer = PorterStemmer()
        self.summarySpacyObject = self._initSummarySpacyObject()
        self.name = 'SummarizerTextRankPlusLexical'

    def _initSummarySpacyObject(self):
        # get the top summary sentences per document (to cut time significantly for the full corpus processing):
        perDocSummTexts = []
        for docIdx, doc in enumerate(self.corpus.documents):
            #docObj = doc.spacyDoc
            docSumm = ''
            #for sent in docObj._.textrank.summary(limit_phrases=20, limit_sentences=3):

            for sentText in doc.topSentencesText:
                if sentText.strip()[-1] != '.':
                    docSumm += sentText.strip() + '. '
                else:
                    docSumm += sentText.strip() + ' '
            perDocSummTexts.append(docSumm)

            #docObj = doc.spacyDoc
            #docSumm = ' '.join([sent.text for sent in docObj._.textrank.summary(limit_phrases=20, limit_sentences=3)])
            #perDocSummTexts.append(docSumm)

        # create a SpaCy object for the concatenated summaries of all the documents:
        return nlp(' '.join(perDocSummTexts))

    def _getGenericSummaryText(self, desiredWordCount):
        # The algorithm here is (executed partly in _initSummarySpacyObject):
        #   Get TextRank summaries of three sentences per document
        #   Get a TextRank summary from the concatenated summaries

        # concatenate sentences until the the word limit is up:
        finalSummaryTxtList, finalSummaryIds, numWordsInSummary = self._getNextGeneralSentences(desiredWordCount, countType='word')

        if len(finalSummaryTxtList) == 0:
            finalSummaryTxtList = ['NO INFORMATION TO SHOW.']
            finalSummaryIds = []

        return finalSummaryTxtList, finalSummaryIds, numWordsInSummary

    def _getNextGeneralSentences(self, desiredCount, countType='word'):
        # desiredCount: how many words or sentences should be returned
        # countType: 'word' for desiredWordCount or 'sentence' for desiredSentenceCount

        if self._noMoreSentences():
            return [], [], 0

        # collect summary sentences until the the word limit is up:
        numWordsInSummary = 0
        finalSummaryTxtList = []
        finalSummaryIds = []
        if countType == 'word':
            numSentenceToGet = int(self.lastSentenceIndexFromGeneric + 1 + ((desiredCount / 10) * 2))
        elif countType == 'sentence':
            numSentenceToGet = int(self.lastSentenceIndexFromGeneric + 1 + (desiredCount * 4))

        numSentencesTaken = 0
        for sentIndex, spacySent in enumerate(
                self.summarySpacyObject._.textrank.summary(limit_phrases=20, limit_sentences=numSentenceToGet)):

            # skip forward to the next sentence to check
            if sentIndex <= self.lastSentenceIndexFromGeneric:
                continue

            # get the sentence ID of this candidate sentence:
            sentId = self.corpus.getSentenceIdByText(spacySent.text)
            # skip this sentence if it wasn't found in the corpus (since this is a SpaCy doc object initialized
            # with potenially slightly modified text, the sentence tokenization may be a bit different):
            if sentId == None:
                continue
            sentence = self.corpus.getSentenceById(sentId)
            # skip non-usable sentences:
            if not self._isPotentialSentence(sentence):
                continue

            # keep this sentence for the summary:
            finalSummaryTxtList.append(sentence.text)
            finalSummaryIds.append(sentId)
            self.usedSentences[sentId] = sentence
            self.usedSentencesText[sentence.textCompressed] = sentId
            numWordsInSummary += len(sentence)
            numSentencesTaken += 1
            # stop conditions:
            if self._noMoreSentences() or \
                    (countType == 'word' and numWordsInSummary >= desiredCount) or \
                    (countType == 'sentence' and numSentencesTaken == desiredCount):
                self.lastSentenceIndexFromGeneric = sentIndex
                break

        return finalSummaryTxtList, finalSummaryIds, numWordsInSummary

    def _getQuerySummaryText(self, query, numSentencesNeeded=-1, numTokensNeeded=-1):
        # The algorithm here is:
        #   stem the query
        #   Get the ROUGE similarity of the query to each of the potential sentences in the corpus
        #   Take the most similar sentences to the query as long as it isn't redundant to the sentences already added
        #       (and not sentences in previous summaries)

        if self._noMoreSentences():
            return ["NO MORE INFORMATION."], [], 0

        if numTokensNeeded <= 0 and numSentencesNeeded > 0:
            numTokensNeeded = numSentencesNeeded*20
        if numSentencesNeeded <= 0:
            numSentencesNeeded = numTokensNeeded / 20

        # an empty query means to just get another "general" sentence:
        if query == '':
            finalSummaryTxtList, finalSummaryIds, numWordsInSummary = self._getNextGeneralSentences(numSentencesNeeded, countType='sentence')
            return finalSummaryTxtList, finalSummaryIds, numWordsInSummary

        # get similarity scores to the rest of the available sentences:
        scoreBySentIdSorted = self._getSimilarityScores(query)

        # append sentences for the response to the query:
        responseSentences = []
        curIdxOfPotentialResults = 0
        while len(responseSentences) < numSentencesNeeded:
            sentId, simScore = scoreBySentIdSorted[curIdxOfPotentialResults]
            curIdxOfPotentialResults += 1
            # if there is no similarity at all (score is 1), then stop looking
            if simScore == 1:
                break
            responseSentences.append(self.corpus.getSentenceById(sentId))
            sentence = self.corpus.getSentenceById(sentId)
            self.usedSentences[sentId] = sentence
            self.usedSentencesText[sentence.textCompressed] = sentId

        responseSentenceTexts = [s.text for s in responseSentences]
        responseSentenceIds = [s.sentId for s in responseSentences]
        summaryLength = sum(len(s) for s in responseSentences)

        return responseSentenceTexts, responseSentenceIds, summaryLength

    def _getSimilarityScores(self, query):
        # gets the similarity scores between the given query and all the relevant available sentences
        # input: query (text string)
        # output: list of tuples (sentId, score) sorted by high to low score

        # get the sentences for which to get similarities (non used potential sentences):
        sentencesToCompareTo = [s for s in self.allSentencesForPotentialSummaries if s.sentId not in self.usedSentences]
        # get different similarity scores:
        lexicalScores = self._getLexicalSimilarityScores(query, sentencesToCompareTo)
        semanticScores = self._getSemanticSimilarityScores(query, sentencesToCompareTo)

        # combine the scores:
        allScores = {} # sentId -> metric -> score
        for sentence in sentencesToCompareTo:
            sId  = sentence.sentId
            allScores[sId] = {}
            for metric in lexicalScores:
                allScores[sId][metric] = lexicalScores[metric][sId]
            for metric in semanticScores:
                allScores[sId][metric] = semanticScores[metric][sId]

        # multiply the measures together (adding 1 to each separate score so that 0 does not cancel everything out)
        # put it in a list of tuples (sentId, score)
        scoreBySentId = [(s.sentId, reduce((lambda x, y: x * y),
                                           [allScores[s.sentId][measure] + 1 for measure in allScores[s.sentId]]))
                          for s in sentencesToCompareTo]

        # sort the tuples by the scores:
        scoreBySentIdSorted = sorted(scoreBySentId, key=lambda x: x[1], reverse=True)

        return scoreBySentIdSorted # sorted list of tuples (sentId, score)

    def _getLexicalSimilarityScores(self, query, sentencesToCompareTo):
        # Gets the ROUGE precision scores between the query and the sentences specified
        # input: query -- the query string given by the user
        #        sentencesToCompareTo -- list of sentence objects
        # output: A dictionary of {metric -> {sentenceId -> score}} where metrics are r1p, r2p and rlp
        #         for ROUGE1-prec, ROUGE2-prec and ROUGEL-prec respectively
        #         The sentence IDs are of those specified in sentencesToCompareTo

        # tokenize sentences:
        qStemmed = " ".join([self.stemmer.stem(token.lower()) for token in word_tokenize(query)])
        for sentence in sentencesToCompareTo:
            if sentence.sentId not in self.stemmedSentences:
                self.stemmedSentences[sentence.sentId] = " ".join([self.stemmer.stem(token.lower()) for token in sentence.tokens])
        stemmedSentencesToCheck = [self.stemmedSentences[s.sentId] for s in sentencesToCompareTo]

        # get ROUGE scores:
        qList = [qStemmed]*len(sentencesToCompareTo)
        scores = self.rouge.get_scores(qList, stemmedSentencesToCheck)
        #scores = self.rouge.get_scores([qStemmed for _ in range(len(sentencesToCompareTo))], stemmedSentencesToCheck)

        # there's a version of Rouge that the returned list of scores is kept differently
        #finalScores = {
        #    'r1p':{s.sentId: round(scores[sentIdx]['rouge-1']['p'], 4) for sentIdx, s in enumerate(sentencesToCompareTo)},
        #    'r2p':{s.sentId: round(scores[sentIdx]['rouge-2']['p'], 4) for sentIdx, s in enumerate(sentencesToCompareTo)},
        #    'rlp':{s.sentId: round(scores[sentIdx]['rouge-l']['p'], 4) for sentIdx, s in enumerate(sentencesToCompareTo)}
        #}
        finalScores = {
            'r1p': {s.sentId: round(scores['rouge-1'][sentIdx]['p'][0], 4) for sentIdx, s in enumerate(sentencesToCompareTo)},
            'r2p': {s.sentId: round(scores['rouge-2'][sentIdx]['p'][0], 4) for sentIdx, s in enumerate(sentencesToCompareTo)},
            'rlp': {s.sentId: round(scores['rouge-l'][sentIdx]['p'][0], 4) for sentIdx, s in enumerate(sentencesToCompareTo)}
        }

        return finalScores

    def _getSemanticSimilarityScores(self, query, sentencesToCompareTo):
        # Gets the semantic similarity scores between the query and the sentences specified
        # input: query -- the query string given by the user
        #        sentencesToCompareTo -- list of sentence objects
        # output: A dictionary of {metric: {sentenceId -> score}} for the representation style defined by the corpus
        #         in self.corpus.representationStyle.
        #         The sentence IDs are of those specified in sentencesToCompareTo

        queryAsSentence = Sentence(QUERY_DOC_ALIAS, len(self.queries), query, self.corpus.representationStyle)
        similaritiesToQuery = {sentence.sentId: sentence.similarity(queryAsSentence) for sentence in sentencesToCompareTo}

        return {self.corpus.representationStyle: similaritiesToQuery}

    #def _isPotentialSentence(self, sentence):
    #    return isPotentialSentence(sentence)

    def _noMoreSentences(self):
        return len(self.usedSentences) == len(self.allSentencesForPotentialSummaries)

    def forceSetSnapshot(self, snapshotInfo):
        super().forceSetSnapshot(snapshotInfo)
        self.lastSentenceIndexFromGeneric = -1
        self.summarySpacyObject = self._initSummarySpacyObject() # seems like this needs to be reset for some reason