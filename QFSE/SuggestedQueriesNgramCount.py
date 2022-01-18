from QFSE.SuggestedQueriesBase import SuggestedQueriesBase
import Levenshtein

class SuggestedQueriesNgramCount(SuggestedQueriesBase):

    def __init__(self, corpus):
        super().__init__(corpus)
        self.name = 'SuggestedQueriesNgramCount'

    def _getTopSuggestions(self, numKeywordsNeeded, presentedSentsSoFar=[], getScores=False):
        # get the next numKeywords keywords (not already sent):
        # wordsToReturn = self.wordCounter.most_common(self._numKeywordsExtracted + numKeywordsNeeded)[
        #                self._numKeywordsExtracted:]
        # wordsToReturn = self._bigramCounter.most_common(self._numKeywordsExtracted + numKeywordsNeeded)[
        #                self._numKeywordsExtracted:]

        potentialNgrams = self.corpus.ngramCounter.most_common()#[extractionStartIndex:]
        curOptionInd = 0
        ngramsToReturn = []
        ngramFullCount = sum(self.corpus.ngramCounter.values())
        while len(ngramsToReturn) < numKeywordsNeeded and curOptionInd < len(potentialNgrams):
            curPotentialNgram, curPotentialNgramCount = potentialNgrams[curOptionInd]#[0]
            if self._isNearDuplicate([g for g, s in ngramsToReturn], curPotentialNgram) < 0:
                ngramScore = float(curPotentialNgramCount) / ngramFullCount
                ngramsToReturn.append((curPotentialNgram, ngramScore))
            curOptionInd += 1

        # if not needed, remove the scores from the list:
        if not getScores:
            ngramsToReturn = [kpTxt for kpTxt, kpScore in ngramsToReturn]

        # return [word for word, frequency in wordsToReturn]
        return ngramsToReturn

    def _isNearDuplicate(self, stringList, newString, distance=2):
        for sInd, s in enumerate(stringList):
            if Levenshtein.distance(s, newString) <= distance:
                return sInd
        return -1