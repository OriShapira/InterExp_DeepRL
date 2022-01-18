#import sys
#sys.path.append('..')

from QFSE.Utilities import loadSpacy, loadBert
from QFSE.Utilities import REPRESENTATION_STYLE_W2V, REPRESENTATION_STYLE_SPACY, REPRESENTATION_STYLE_BERT

# The SpaCy and BERT objects must be loaded before anything else, so that classes using them get the initialized objects.
# The SpaCy and BERT objects are initialized only when needed since these init processes take a long time.
REPRESENTATION_STYLE = REPRESENTATION_STYLE_SPACY #REPRESENTATION_STYLE_W2V REPRESENTATION_STYLE_BERT
loadSpacy()
if REPRESENTATION_STYLE == REPRESENTATION_STYLE_BERT:
    loadBert()

import sys
import time
import os
from QFSE.Corpus import Corpus
from data.Config import CORPORA_LOCATIONS, CORPUS_REFSUMMS_RELATIVE_PATH, CORPUS_QUESTIONNAIRE_RELATIVE_PATH
#from QFSE.SummarizerClustering import SummarizerClustering
#from QFSE.SummarizerTextRankPlusLexical import SummarizerTextRankPlusLexical
#from QFSE.SuggestedQueriesNgramCount import SuggestedQueriesNgramCount
from QFSE.SuggestedQueriesTextRank import SuggestedQueriesTextRank
from QFSE.SummarizerQRLMMR import SummarizerQRLMMR
from QFSE.SuggestedQueriesKRLMMR import SuggestedQueriesKRLMMR

SUMMARIZER_CLASS = SummarizerQRLMMR #SummarizerClustering #SummarizerTextRankPlusLexical
SUGGESTED_QUERIES_CLASS = SuggestedQueriesKRLMMR #SuggestedQueriesNgramCount #SuggestedQueriesTextRank
QRLMMR_MODEL_NAME = 'Apr12_23-23-47' # Apr21_12-00-53
KRLMMR_MODEL_NAME = 'kp_Jul06_20-01-19'
DEFAULT_FIRST_SUMMARY_LENGTH = 75
EVALUATE_ON_THE_FLY = False #True

def main(corpusName, queriesList, SummarizerClass, SuggestedQueriesClass, representationStyle,
         summarizerModelName=None, suggestedQueriesModelName=None,):
    startTime = time.time()
    referenceSummariesFolder = os.path.join(CORPORA_LOCATIONS[corpusName], CORPUS_REFSUMMS_RELATIVE_PATH)
    questionnaireFilepath = os.path.join(CORPORA_LOCATIONS[corpusName], CORPUS_QUESTIONNAIRE_RELATIVE_PATH)
    corpus = Corpus(CORPORA_LOCATIONS[corpusName], referenceSummariesFolder, questionnaireFilepath, representationStyle=representationStyle)
    if SummarizerClass == SummarizerQRLMMR:
        summarizer = SummarizerClass(corpus, summarizerModelName, evaluateOnTheFly=EVALUATE_ON_THE_FLY)
    else:
        summarizer = SummarizerClass(corpus, evaluateOnTheFly=EVALUATE_ON_THE_FLY)
    # initialize suggested queries object:
    if SuggestedQueriesClass == SuggestedQueriesTextRank:
        suggestedQueriesGenerator = SuggestedQueriesClass(corpus, summarizer)
    elif SuggestedQueriesClass == SuggestedQueriesKRLMMR:
        suggestedQueriesGenerator = SuggestedQueriesClass(corpus, suggestedQueriesModelName)
    else:
        suggestedQueriesGenerator = SuggestedQueriesClass(corpus)



    queries = [('', DEFAULT_FIRST_SUMMARY_LENGTH)] + [(query, -1) for query in queriesList]

    queryNum = 0
    while True:
        if queryNum < len(queries):
            queryText, desiredSummLen = queries[queryNum]
        else:
            queryText = input('Enter query or "*exit" or "*plot": ')
            desiredSummLen = -1
            if queryText == '*plot':
                summarizer.plotRougeCurves()
                continue
            elif queryText == '*exit':
                break

        print('--- {} : Query: {} ---'.format(time.time() - startTime, queryText))

        if queryNum == 0:
            summarySentenceList, summaryLength = summarizer.summarizeGeneric(desiredSummLen)
        else:
            summarySentenceList, summaryLength = summarizer.summarizeByQuery(queryText, 2, "free_text")

        print('\n'.join(summarySentenceList))
        print('--- Length: {}'.format(summaryLength))

        # show suggested queries:
        if queryNum == 0 or SUGGESTED_QUERIES_CLASS == SuggestedQueriesKRLMMR:
            suggestions = suggestedQueriesGenerator.getTopSuggestions(
                10, presentedSentsSoFar=list(summarizer.usedSentences.keys()))
            for sugg in suggestions:
                print('\t\t{}'.format(sugg))

        print('--- {} : finish ---\n'.format(time.time() - startTime))

        queryNum += 1


if __name__ == '__main__':
    # set REPRESENTATION_STYLE, SUMMARIZER_CLASS and DEFAULT_FIRST_SUMMARY_LENGTH at the top of this page

    # see corpus names under data.Config.CORPORA_LOCATIONS
    corpusName = sys.argv[1]
    if len(sys.argv) > 2:
        queries = sys.argv[2:]
    else:
        queries = []

    main(corpusName, queries, SUMMARIZER_CLASS, SUGGESTED_QUERIES_CLASS, REPRESENTATION_STYLE,
         summarizerModelName=QRLMMR_MODEL_NAME, suggestedQueriesModelName=KRLMMR_MODEL_NAME)