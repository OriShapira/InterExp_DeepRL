

class SuggestedQueriesBase:

    def __init__(self, corpus):
        self.corpus = corpus
        # the number of keywords already returned until now:
        self._keywordsExtractedLists = [] # list of lists if the class regenerates, or list of one list if not
        self.name = 'SuggestedQueriesBase'

    def getTopSuggestions(self, numKeywordsNeeded, presentedSentsSoFar=[], getScores=False):
        """
        Get a list of the top suggested queries.
        :param numKeywordsNeeded: How many to get.
        :param presentedSentsSoFar: Optional: list of sentence IDs of sentences already presented, to consider.
        :param getScores: Optional: should the score of the query be paired with the query text as well.
        :return: List of text phrases, or list of text-score tuples (if getScores==True).
        """
        # prepare an empty list until we get the actual list (possible needed in case the list is requested
        # (in the getSuggestionsList function) in a separate thread before it is ready):
        self._keywordsExtractedLists.append([])
        # get the list of suggested queries:
        suggestions = self._getTopSuggestions(numKeywordsNeeded, presentedSentsSoFar=presentedSentsSoFar,
                                              getScores=getScores)
        # place the actual list now in place of the empty one:
        self._keywordsExtractedLists[-1] = suggestions

        return suggestions

    #def getNextTopSuggestions(self, numKeywordsNeeded, presentedSentsSoFar=[], getScores=False):
    #    """
    #    Get the next list of suggested queries. (Tries reading from the next part of the unseen list)
    #    :param numKeywordsNeeded: How many to get.
    #    :param presentedSentsSoFar: Optional: list of sentence IDs of sentences already presented, to consider.
    #    :param getScores: Optional: should the score of the query be paired with the query text as well.
    #    :return: List of text phrases, or list of text-score tuples (if getScores==True).
    #    """
    #    suggestions = self._getNextTopSuggestions(len(self._keywordsExtracted), numKeywordsNeeded,
    #                                              presentedSentsSoFar=presentedSentsSoFar, getScores=getScores)
    #    self._keywordsExtracted.extend(suggestions)
    #    return suggestions

    #def _getNextTopSuggestions(self, extractionStartIndex, numKeywordsNeeded,
    #                           presentedSentsSoFar=[], getScores=False):
    #    return []

    def _getTopSuggestions(self, numKeywordsNeeded, presentedSentsSoFar=[], getScores=False):
        return []

    def getSuggestionAtIndex(self, index):
        ## if we didn't extract the suggestion at the index requested, get the suggestions up until that index first:
        #if index >= len(self._keywordsExtracted):
        #    self.getNextTopSuggestions(index - len(self._keywordsExtracted) + 1)
        # if we didn't extract the suggestion at the index requested, get the suggestions up until that index first:
        #return self._keywordsExtracted[index]
        if index >= len(self._keywordsExtractedLists[-1]):
            return None
        return self._keywordsExtractedLists[-1][index]

    #def getSuggestionsFromToIndices(self, fromInd, toInd):
    #    # if we didn't extract the suggestion at the toInd requested, get the suggestions up until that index first:
    #    if toInd >= len(self._keywordsExtracted):
    #        self.getNextTopSuggestions(toInd - len(self._keywordsExtracted) + 1)
    #    return self._keywordsExtracted[fromInd : toInd+1]

    def getSuggestionsList(self, listIdx):
        if listIdx >= len(self._keywordsExtractedLists):
            return None
        return self._keywordsExtractedLists[listIdx]