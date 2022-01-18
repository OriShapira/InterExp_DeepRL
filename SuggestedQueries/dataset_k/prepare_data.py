import sys
import os
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams, trigrams
import string
from collections import Counter
import Levenshtein
from rake_nltk import Rake
from gensim.summarization import keywords


DATA_SETS = {
    'train': {'2007': 'C:/Users/user/Data/DUCTAC/DUC2007/task1.jsonl'},
    'val': {'2005': 'C:/Users/user/Data/DUCTAC/DUC2005/task1.jsonl'},
    'test': {'2006': 'C:/Users/user/Data/DUCTAC/DUC2006/task1.jsonl'}
}
#DATA_SETS = {
#    'train': {
#        '2001': 'C:/Users/user/Data/DUCTAC/DUC2001/task2.train.200.jsonl',
#        '2002': 'C:/Users/user/Data/DUCTAC/DUC2002/task2.200.jsonl',
#        '2003': 'C:/Users/user/Data/DUCTAC/DUC2003/task2.jsonl',
#        '2004': 'C:/Users/user/Data/DUCTAC/DUC2004/task2.jsonl',
#        '2005': 'C:/Users/user/Data/DUCTAC/DUC2005/task1.jsonl',
#        '2008': 'C:/Users/user/Data/DUCTAC/TAC2008/task1.A.jsonl',
#        '2009': 'C:/Users/user/Data/DUCTAC/TAC2009/task1.A.jsonl',
#        '2010': 'C:/Users/user/Data/DUCTAC/TAC2010/task1.A.jsonl'
#    }
#}
OUTPUT_BASE_FOLDER_PATH = 'DUC'


STOP_WORDS = {'a':1, 'able':1, 'about':1, 'above':1, 'abst':1, 'accordance':1, 'according':1, 'accordingly':1,
              'across':1, 'act':1, 'actually':1, 'added':1, 'adj':1, 'adopted':1, 'affected':1, 'affecting':1,
              'affects':1, 'after':1, 'afterwards':1, 'again':1, 'against':1, 'ah':1, 'all':1, 'almost':1, 'alone':1,
              'along':1, 'already':1, 'also':1, 'although':1, 'always':1, 'am':1, 'among':1, 'amongst':1, 'an':1,
              'and':1, 'announce':1, 'another':1, 'any':1, 'anybody':1, 'anyhow':1, 'anymore':1, 'anyone':1,
              'anything':1, 'anyway':1, 'anyways':1, 'anywhere':1, 'apparently':1, 'approximately':1, 'are':1,
              'aren':1, 'arent':1, 'arise':1, 'around':1, 'as':1, 'aside':1, 'ask':1, 'asking':1, 'at':1, 'auth':1,
              'available':1, 'away':1, 'awfully':1, 'b':1, 'back':1, 'be':1, 'became':1, 'because':1, 'become':1,
              'becomes':1, 'becoming':1, 'been':1, 'before':1, 'beforehand':1, 'begin':1, 'beginning':1, 'beginnings':1,
              'begins':1, 'behind':1, 'being':1, 'believe':1, 'below':1, 'beside':1, 'besides':1, 'between':1,
              'beyond':1, 'biol':1, 'both':1, 'brief':1, 'briefly':1, 'but':1, 'by':1, 'c':1, 'ca':1, 'came':1,
              'can':1, 'cannot':1, "can't":1, 'certain':1, 'certainly':1, 'co':1, 'com':1, 'come':1, 'comes':1,
              'contain':1, 'containing':1, 'contains':1, 'could':1, "couldn't":1, 'd':1, 'date':1, 'did':1, "didn't":1,
              'different':1, 'do':1, 'does':1, "doesn't":1, 'doing':1, 'done':1, "don't":1, 'down':1, 'downwards':1,
              'due':1, 'during':1, 'e':1, 'each':1, 'ed':1, 'edu':1, 'effect':1, 'eg':1, 'eight':1, 'eighty':1,
              'either':1, 'else':1, 'elsewhere':1, 'end':1, 'ending':1, 'enough':1, 'especially':1, 'et':1, 'et-al':1,
              'etc':1, 'even':1, 'ever':1, 'every':1, 'everybody':1, 'everyone':1, 'everything':1, 'everywhere':1,
              'ex':1, 'except':1, 'f':1, 'far':1, 'few':1, 'ff':1, 'fifth':1, 'first':1, 'five':1, 'fix':1,
              'followed':1, 'following':1, 'follows':1, 'for':1, 'former':1, 'formerly':1, 'forth':1, 'found':1,
              'four':1, 'from':1, 'further':1, 'furthermore':1, 'g':1, 'gave':1, 'get':1, 'gets':1, 'getting':1,
              'give':1, 'given':1, 'gives':1, 'giving':1, 'go':1, 'goes':1, 'gone':1, 'got':1, 'gotten':1, 'h':1,
              'had':1, 'happens':1, 'hardly':1, 'has':1, "hasn't":1, 'have':1, "haven't":1, 'having':1, 'he':1,
              'hed':1, 'hence':1, 'her':1, 'here':1, 'hereafter':1, 'hereby':1, 'herein':1, 'heres':1, 'hereupon':1,
              'hers':1, 'herself':1, 'hes':1, 'hi':1, 'hid':1, 'him':1, 'himself':1, 'his':1, 'hither':1, 'home':1,
              'how':1, 'howbeit':1, 'however':1, 'hundred':1, 'i':1, 'id':1, 'ie':1, 'if':1, "i'll":1, 'im':1,
              'immediate':1, 'immediately':1, 'importance':1, 'important':1, 'in':1, 'inc':1, 'indeed':1, 'index':1,
              'information':1, 'instead':1, 'into':1, 'invention':1, 'inward':1, 'is':1, "isn't":1, 'it':1, 'itd':1,
              "it'll":1, 'its':1, 'itself':1, "i've":1, 'j':1, 'just':1, 'k':1, 'keep':1, 'keeps':1, 'kept':1,
              'keys':1, 'kg':1, 'km':1, 'know':1, 'known':1, 'knows':1, 'l':1, 'largely':1, 'last':1, 'lately':1,
              'later':1, 'latter':1, 'latterly':1, 'least':1, 'less':1, 'lest':1, 'let':1, 'lets':1, 'like':1,
              'liked':1, 'likely':1, 'line':1, 'little':1, "'ll":1, 'look':1, 'looking':1, 'looks':1, 'ltd':1,
              'm':1, 'made':1, 'mainly':1, 'make':1, 'makes':1, 'many':1, 'may':1, 'maybe':1, 'me':1, 'mean':1,
              'means':1, 'meantime':1, 'meanwhile':1, 'merely':1, 'mg':1, 'might':1, 'million':1, 'miss':1, 'ml':1,
              'more':1, 'moreover':1, 'most':1, 'mostly':1, 'mr':1, 'mrs':1, 'much':1, 'mug':1, 'must':1, 'my':1,
              'myself':1, 'n':1, 'na':1, 'name':1, 'namely':1, 'nay':1, 'nd':1, 'near':1, 'nearly':1, 'necessarily':1,
              'necessary':1, 'need':1, 'needs':1, 'neither':1, 'never':1, 'nevertheless':1, 'new':1, 'next':1, 'nine':1,
              'ninety':1, 'no':1, 'nobody':1, 'non':1, 'none':1, 'nonetheless':1, 'noone':1, 'nor':1, 'normally':1,
              'nos':1, 'not':1, 'noted':1, 'nothing':1, 'now':1, 'nowhere':1, 'o':1, 'obtain':1, 'obtained':1,
              'obviously':1, 'of':1, 'off':1, 'often':1, 'oh':1, 'ok':1, 'okay':1, 'old':1, 'omitted':1, 'on':1,
              'once':1, 'one':1, 'ones':1, 'only':1, 'onto':1, 'or':1, 'ord':1, 'other':1, 'others':1, 'otherwise':1,
              'ought':1, 'our':1, 'ours':1, 'ourselves':1, 'out':1, 'outside':1, 'over':1, 'overall':1, 'owing':1,
              'own':1, 'p':1, 'page':1, 'pages':1, 'part':1, 'particular':1, 'particularly':1, 'past':1, 'per':1,
              'perhaps':1, 'placed':1, 'please':1, 'plus':1, 'poorly':1, 'possible':1, 'possibly':1, 'potentially':1,
              'pp':1, 'predominantly':1, 'present':1, 'previously':1, 'primarily':1, 'probably':1, 'promptly':1,
              'proud':1, 'provides':1, 'put':1, 'q':1, 'que':1, 'quickly':1, 'quite':1, 'qv':1, 'r':1, 'ran':1,
              'rather':1, 'rd':1, 're':1, 'readily':1, 'really':1, 'recent':1, 'recently':1, 'ref':1, 'refs':1,
              'regarding':1, 'regardless':1, 'regards':1, 'related':1, 'relatively':1, 'research':1, 'respectively':1,
              'resulted':1, 'resulting':1, 'results':1, 'right':1, 'run':1, 's':1, 'said':1, 'same':1, 'saw':1,
              'say':1, 'saying':1, 'says':1, 'sec':1, 'section':1, 'see':1, 'seeing':1, 'seem':1, 'seemed':1,
              'seeming':1, 'seems':1, 'seen':1, 'self':1, 'selves':1, 'sent':1, 'seven':1, 'several':1, 'shall':1,
              'she':1, 'shed':1, "she'll":1, 'shes':1, 'should':1, "shouldn't":1, 'show':1, 'showed':1, 'shown':1,
              'showns':1, 'shows':1, 'significant':1, 'significantly':1, 'similar':1, 'similarly':1, 'since':1,
              'six':1, 'slightly':1, 'so':1, 'some':1, 'somebody':1, 'somehow':1, 'someone':1, 'somethan':1,
              'something':1, 'sometime':1, 'sometimes':1, 'somewhat':1, 'somewhere':1, 'soon':1, 'sorry':1,
              'specifically':1, 'specified':1, 'specify':1, 'specifying':1, 'state':1, 'states':1, 'still':1, 'stop':1,
              'strongly':1, 'sub':1, 'substantially':1, 'successfully':1, 'such':1, 'sufficiently':1, 'suggest':1,
              'sup':1, 'sure':1, 't':1, 'take':1, 'taken':1, 'taking':1, 'tell':1, 'tends':1, 'th':1, 'than':1,
              'thank':1, 'thanks':1, 'thanx':1, 'that':1, "that'll":1, 'thats':1, "that've":1, 'the':1, 'their':1,
              'theirs':1, 'them':1, 'themselves':1, 'then':1, 'thence':1, 'there':1, 'thereafter':1, 'thereby':1,
              'thered':1, 'therefore':1, 'therein':1, "there'll":1, 'thereof':1, 'therere':1, 'theres':1, 'thereto':1,
              'thereupon':1, "there've":1, 'these':1, 'they':1, 'theyd':1, "they'll":1, 'theyre':1, "they've":1,
              'this':1, 'those':1, 'thou':1, 'though':1, 'thoughh':1, 'thousand':1, 'throug':1, 'through':1,
              'throughout':1, 'thru':1, 'thus':1, 'til':1, 'tip':1, 'to':1, 'together':1, 'too':1, 'took':1, 'toward':1,
              'towards':1, 'tried':1, 'tries':1, 'truly':1, 'try':1, 'trying':1, 'ts':1, 'twice':1, 'two':1, 'u':1,
              'un':1, 'under':1, 'unfortunately':1, 'unless':1, 'unlike':1, 'unlikely':1, 'until':1, 'unto':1, 'up':1,
              'upon':1, 'ups':1, 'us':1, 'use':1, 'used':1, 'useful':1, 'usefully':1, 'usefulness':1, 'uses':1,
              'using':1, 'usually':1, 'v':1, 'value':1, 'various':1, "'ve":1, 'very':1, 'via':1, 'viz':1, 'vol':1,
              'vols':1, 'vs':1, 'w':1, 'want':1, 'wants':1, 'was':1, "wasn't":1, 'way':1, 'we':1, 'wed':1, 'welcome':1,
              "we'll":1, 'went':1, 'were':1, "weren't":1, "we've":1, 'what':1, 'whatever':1, "what'll":1, 'whats':1,
              'when':1, 'whence':1, 'whenever':1, 'where':1, 'whereafter':1, 'whereas':1, 'whereby':1, 'wherein':1,
              'wheres':1, 'whereupon':1, 'wherever':1, 'whether':1, 'which':1, 'while':1, 'whim':1, 'whither':1,
              'who':1, 'whod':1, 'whoever':1, 'whole':1, "who'll":1, 'whom':1, 'whomever':1, 'whos':1, 'whose':1,
              'why':1, 'widely':1, 'willing':1, 'wish':1, 'with':1, 'within':1, 'without':1, "won't":1, 'words':1,
              'world':1, 'would':1, "wouldn't":1, 'www':1, 'x':1, 'y':1, 'yes':1, 'yet':1, 'you':1, 'youd':1,
              "you'll":1, 'your':1, 'youre':1, 'yours':1, 'yourself':1, 'yourselves':1, "you've":1, 'z':1, 'zero':1,
              '`':1, '``':1, "'":1, '(':1, ')':1, ',':1, '_':1, ';':1, ':':1, '~':1, '-':1, '--':1, '$':1, '^':1, '*':1,
              "'s":1, "'t":1, "'m":1, 'doesn':1, 'don':1, 'hasn':1, 'haven':1, 'isn':1, 'wasn':1,
              'won':1, 'weren':1, 'wouldn':1, 'didn':1, 'shouldn':1, 'couldn':1, '':1}


def getDatasetData(dsPath):
    datasetData = []
    with open(dsPath, 'r') as fIn:
        for line in fIn:
            lineTxt = line.strip()
            docSetInfo = json.loads(lineTxt)
            docSetId = docSetInfo['instance_id'] if 'instance_id' in docSetInfo else docSetInfo['id']
            datasetData.append({'docSetId': docSetId, 'docs': [], 'refs': []})
            docsInfo = docSetInfo['documents'] if 'documents' in docSetInfo else docSetInfo['documents_A']
            for docInfo in docsInfo:
                if isinstance(docInfo['text'][0], list) and len(docInfo['text']) == 1:
                    docInfoText = docInfo['text'][0]
                else:
                    docInfoText = docInfo['text']
                sentences = [' '.join(word_tokenize(s if isinstance(s, str) else s[0])) for s in docInfoText]
                datasetData[-1]['docs'].append(sentences)
            refsInfo = docSetInfo['summaries'] if 'summaries' in docSetInfo else docSetInfo['summaries_A']
            for refInfo in refsInfo:
                if isinstance(refInfo, list):
                    refInfo = refInfo[0]
                #print(refInfo)
                sentences = [' '.join(word_tokenize(s)) for s in refInfo['text']]
                datasetData[-1]['refs'].append(sentences)

    return datasetData

def loadData(datasetsPaths):
    dsData = {}
    for split, sourceDataJsonlPaths in datasetsPaths.items():
        print('\tLoading split ' + split)
        dsData[split] = {}
        for dataName, dataJsonlPath in sourceDataJsonlPaths.items():
            print('\t\tLoading ds ' + dataName)
            # get the data:
            dsData[split][dataName] = getDatasetData(dataJsonlPath)
    return dsData


def outputTopics(dsData, split, outputBaseFolderPath, dsName):
    outputFolderPath = os.path.join(outputBaseFolderPath, split, 'topics')
    if not os.path.exists(outputFolderPath):
        os.makedirs(outputFolderPath)
    for docSetIdx, docSetInfo in enumerate(dsData):
        id = '{}_{}'.format(dsName, docSetInfo['docSetId'])
        dataDict = {}
        dataDict['id'] = id
        dataDict['documents'] = docSetInfo['docs']
        dataDict['reference_summaries'] = docSetInfo['refs']
        # output the data to the fastAbsRl format:
        outputJsonPath = os.path.join(outputFolderPath, '{}.json'.format(id))
        with open(outputJsonPath, 'w') as fOut:
            json.dump(dataDict, fOut, indent=4)

def outputSamples(keyphrasesData, split, outputBaseFolderPath):
    outputFolderPath = os.path.join(outputBaseFolderPath, split, 'samples')
    if not os.path.exists(outputFolderPath):
        os.makedirs(outputFolderPath)
    for topicId, topicKpData in keyphrasesData.items():
        # if there are queries:
        if len(topicKpData) > 0:
            for methodName, kpData in topicKpData.items():
                dataDict = {}
                dataDict['id'] = '{}_{}'.format(topicId, methodName)
                dataDict['topic_id'] = topicId
                dataDict['method'] = methodName
                dataDict['initial_summary_sentence_ids'] = []
                dataDict['queries'] = [[kp, 1] for kp in kpData]
                # output the data to the QRLMMR format:
                outputJsonPath = os.path.join(outputFolderPath, '{}.json'.format(dataDict['id']))
                with open(outputJsonPath, 'w') as fOut:
                    json.dump(dataDict, fOut, indent=4)
        else: # no keyphrases, so set as generic summary:
            dataDict = {}
            dataDict['id'] = '{}_generic'.format(topicId)
            dataDict['topic_id'] = topicId
            dataDict['method'] = 'generic'
            dataDict['initial_summary_sentence_ids'] = []
            dataDict['queries'] = [['', 10]]
            # output the data to the QRLMMR format:
            outputJsonPath = os.path.join(outputFolderPath, '{}.json'.format(dataDict['id']))
            with open(outputJsonPath, 'w') as fOut:
                json.dump(dataDict, fOut, indent=4)


def topNgrams(docs, numKeywordsNeeded=20):
    stop_words = set(stopwords.words('english'))
    # count bigrams and trigrams:
    allTokensLower = [token.lower().strip() for doc in docs for sent in doc for token in word_tokenize(sent)]
    allWords = [token for token in allTokensLower if
                token not in STOP_WORDS and (token == '&' or token not in string.punctuation)]
    #wordCounter = Counter(allWords)
    allBigrams = ['{} {}'.format(word1, word2) for word1, word2 in bigrams(allWords) if
                  word1 != '&' and word2 != '&']
    allTrigrams = ['{} {} {}'.format(word1, word2, word3) for word1, word2, word3 in trigrams(allWords) if
                   word1 != '&' and word3 != '&']
    bigramCounter = Counter(allBigrams)
    trigramCounter = Counter(allTrigrams)
    ngramCounter = bigramCounter | trigramCounter
    # remove bigrams within trigrams:
    for bigram, bCount in bigramCounter.items():
        if bCount > 2:
            for trigram, tCount in trigramCounter.items():
                if tCount > 2:
                    if bCount == tCount and bigram in trigram:
                        del ngramCounter[bigram]

    def _isNearDuplicate(stringList, newString, distance=2):
        for sInd, s in enumerate(stringList):
            if Levenshtein.distance(s, newString) <= distance:
                return sInd
        return -1

    # get the top non-duplicate bi/tri-grams:
    potentialWords = ngramCounter.most_common()
    curOptionInd = 0
    wordsToReturn = []
    while len(wordsToReturn) < numKeywordsNeeded and curOptionInd < len(potentialWords):
        curPotentialWord = potentialWords[curOptionInd][0]
        if _isNearDuplicate(wordsToReturn, curPotentialWord) < 0:
            wordsToReturn.append(curPotentialWord)
        curOptionInd += 1

    return wordsToReturn

def rakeKeyphrases(docs, numKeywordsNeeded=20):
    #fullDocsText = ' '.join(word_tokenize(sent) for doc in docs for sent in doc)
    fullText = ' '.join([token for doc in docs for sent in doc for token in word_tokenize(sent)
                          if token not in ['-', '--', '``', "''", "'", '[', ']']])
    r = Rake()
    r.extract_keywords_from_text(fullText)
    return r.get_ranked_phrases()[:numKeywordsNeeded]

def getTextRanksKeyphrases(docs, numKeywordsNeeded=20):
    fullDocsText = ' '.join(sent for doc in docs for sent in doc)
    #fullText = ' '.join([token for doc in docs for sent in doc for token in word_tokenize(sent)
    #                     if token not in ['-', '--', '``', "''", "'", '[', ']']])
    kps = keywords(fullDocsText, words=numKeywordsNeeded, lemmatize=True, split=True)#.split('\n')[:numKeywordsNeeded*3]
    ## remove almost duplicate keyphrases:
    #kpsIdxIgnore = {}
    #for kpIdx, kp in enumerate(kps):
    #    for i in range(kpIdx+1, len(kps)):
    #        if i not in kpsIdxIgnore: # this index was not yet filtered
    #            if abs(len(kp) - len(kps[i])) < 2: # length difference of 1 or less
    #                if kp in kps[i]:
    #                    kpsIdxIgnore[i] = True
    #                elif kps[i] in kp:
    #                    kpsIdxIgnore[kpIdx] = True
    #topKpsNoRepeat = []
    #for idx in range(len(kps)):
    #    if idx not in kpsIdxIgnore:
    #        topKpsNoRepeat.append(kps[idx])
    #    if len(topKpsNoRepeat) == numKeywordsNeeded:
    #        break

    #return topKpsNoRepeat
    return kps

def getKeyphrases(dsData, dataName, numKeywordsNeeded=20):
    # top-frequent n-grams in the documents
    # top-frequent n-grams in the reference summaries
    # rake key-phrases in the documents
    # rake key-phrases in the reference summaries
    # TextRank key-phrases in the documents
    # TextRank key-phrases in the reference summaries
    keyphrases_all = {}
    for docSetIdx, docSetInfo in enumerate(dsData):
        topicId = '{}_{}'.format(dataName, docSetInfo['docSetId'])
        docs = docSetInfo['docs']
        refSumms = docSetInfo['refs']

        keyphrases_all[topicId] = {}

        if numKeywordsNeeded > 0:
            # get TextRank based keyphrases:
            keyphrases_all[topicId]['textrank_docs'] = getTextRanksKeyphrases(docs, numKeywordsNeeded=20)
            keyphrases_all[topicId]['textrank_refs'] = getTextRanksKeyphrases(refSumms, numKeywordsNeeded=20)

            # get n-gram based keyphrases:
            keyphrases_all[topicId]['ngram_docs'] = topNgrams(docs, numKeywordsNeeded=20)
            keyphrases_all[topicId]['ngram_refs'] = topNgrams(refSumms, numKeywordsNeeded=20)

            # get RAKE based keyphrases:
            keyphrases_all[topicId]['rake_docs'] = rakeKeyphrases(docs, numKeywordsNeeded=20)
            keyphrases_all[topicId]['rake_refs'] = rakeKeyphrases(refSumms, numKeywordsNeeded=20)



    return keyphrases_all





if __name__ == '__main__':
    if len(sys.argv) == 1:
        needOutputTopics = True
        needOutputSamples = True
    elif len(sys.argv) == 2 and sys.argv[1] == '--topics':
        needOutputTopics = True
        needOutputSamples = False
    elif len(sys.argv) == 2 and sys.argv[1] == '--samples':
        needOutputTopics = False
        needOutputSamples = True
    elif len(sys.argv) == 2:
        print('Usage: --topics or --samples')
        sys.exit()

    print('Loading data...')
    dsDataAll = loadData(DATA_SETS)
    
    if needOutputTopics:
        print('Outputting topic data in needed format...')
        for split, splitDsData in dsDataAll.items():
            print('\tOn split ' + split)
            for dataName, dsData in splitDsData.items():
                print('\tOn DS ' + dataName)
                # output to QRLMMR JSON format files in the topics folder:
                outputTopics(dsData, split, OUTPUT_BASE_FOLDER_PATH, dataName)
    if needOutputSamples:
        print('Outputting sample data in needed format...')
        for split, splitDsData in dsDataAll.items():
            print('\tOn split ' + split)
            for dataName, dsData in splitDsData.items():
                print('\tOn DS ' + dataName)
                # get the top keyphrases, to be used as queries lists
                if split == 'train':
                    keyphrases_all = getKeyphrases(dsData, dataName, numKeywordsNeeded=0)
                    #keyphrases_all = getKeyphrases(dsData, dataName, numKeywordsNeeded=20)
                else: # in val and test splits, just use generic summaries (no keywords needed for queries)
                    keyphrases_all = getKeyphrases(dsData, dataName, numKeywordsNeeded=0)
                # output the queries into sample JSON files in the samples folder:
                outputSamples(keyphrases_all, split, OUTPUT_BASE_FOLDER_PATH)

    '''
    - 'topics' folder contains a json file per topic with keys:
        - id: the name of the topic (the file name without the json extension)
        - documents: list of lists of strings (list of sentences per document)
        - reference_summaries: list of list of strings (list of sentences per reference summary)
    - 'samples' folder contains a json file per sample with keys:
        - id: the name of the sample (the file name without the json extension), e.g. 'topicId_method'
        - topic_id: the topic ID for which this sample refers to
        - method: method name used to create the queries. If 'generic', then there are no queries.
        - initial_summary_sentence_ids: list of pair-lists for [doc_index, sent_index] within the documents in the topic
        - queries: a list of pair-lists for [query_str, num_sentences]
            - string representing the query to run
            - an integer for how many sentences should be fetched for the query
    '''