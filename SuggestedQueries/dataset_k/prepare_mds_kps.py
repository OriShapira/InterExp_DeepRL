import json
import os
import shutil
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from numpy import mean

stemmer = PorterStemmer()

DUC2001_JSONL_PATH_TEMPLATE = 'dataset_k/KP_test/task2.test.{}.jsonl'
SUMM_LENS = ['50', '100', '200', '400'] # for the DUC2001_JSONL_PATH_TEMPLATE format
DUC2001_KP_SINGLEDOC_JSON_PATH = 'dataset_k/KP_test/test.reader.json'


def checkFilesCovered(kpDatasetFilenames, ducFilenamesPerTopic, ducAllFilenames):
    # Check that all files are covered between the KP dataset and the official DUC dataset
    missing = False
    for topicId in ducFilenamesPerTopic:
        for fn in ducFilenamesPerTopic[topicId]:
            if fn not in kpDatasetFilenames:
                print(f'Missing from DUC topic {topicId}: {fn}')
                missing = True

    for dsFn in kpDatasetFilenames:
        if dsFn not in ducAllFilenames:
            print(f'Missing from KP DS: {dsFn}')
            missing = True
    
    return not missing


def loadData(kp_duc01_json_path, duc01_jsonl_path_template, duc01_jsonl_summLens):
    
    def loadSummariesFromJsonl(jsonl_path):
        topicsSumms = {}
        with open(jsonl_path, 'r') as fIn:
            for line in fIn:
                topicDatum = json.loads(line.strip())
                topicId = topicDatum['instance_id']
                summaries = [' '.join(summData['text']) for summData in topicDatum['summaries']]
                topicsSumms[topicId] = summaries
        return topicsSumms
    
    # load the KP data from the dataset json:
    with open(kp_duc01_json_path, 'r') as kpDsIn:
        kpJsonData = json.load(kpDsIn)
        kpData = {}
        kpDataStemmed = {}
        for docId in kpJsonData:
            kpData[docId] = [kpL[0] for kpL in kpJsonData[docId]]
            kpDataStemmed[docId] = [' '.join(stemmer.stem(w) for w in word_tokenize(kp)) for kp in kpData[docId]]
            
    # load the DUC 2001 data from the sacreRouge jsonl file:
    
    # get the summaries (at different lengths) for each of the topics in DUC2001:
    allTopicsSumms = {} # topicId -> summLen -> list of summaries (summary is one string)
    for summLen in duc01_jsonl_summLens:
        topicsSumms = loadSummariesFromJsonl(DUC2001_JSONL_PATH_TEMPLATE.format(summLen))
        for topicId, topicSumms in topicsSumms.items():
            if topicId not in allTopicsSumms:
                allTopicsSumms[topicId] = {}
            allTopicsSumms[topicId][summLen] = topicSumms
    
    # load the topic documents for each of the topics in DUC2001:
    allTopicFilenames = {}
    ducFilenamesPerTopic = {}
    ducData = {}
    duc01_jsonl_path = DUC2001_JSONL_PATH_TEMPLATE.format(duc01_jsonl_summLens[0]) # (same docs in all jsonl files)
    with open(duc01_jsonl_path, 'r') as fIn:
        for line in fIn:
            topicDatum = json.loads(line.strip())
            topicId = topicDatum['instance_id']
            ducFilenamesPerTopic[topicId] = []
            ducData[topicId] = {'documents': {}, 'summaries': allTopicsSumms[topicId]}
            for docDatum in topicDatum['documents']:
                docId = docDatum['filename']
                docText = ' '.join(docDatum['text'])
                ducData[topicId]['documents'][docId] = {'text': docText}
                allTopicFilenames[docId] = topicId
                ducFilenamesPerTopic[topicId].append(docId)
           
    # make sure all files are covered:
    allFilesCovered = checkFilesCovered(kpData.keys(), ducFilenamesPerTopic, allTopicFilenames)
    if not allFilesCovered:
        return None
        
    # if all files covered, put the keyphrases in the full duc data per document:
    for docId in kpData:
        docTopicId = allTopicFilenames[docId]
        ducData[docTopicId]['documents'][docId]['keyphrases'] = kpData[docId]
        ducData[docTopicId]['documents'][docId]['keyphrases_stemmed'] = kpDataStemmed[docId]
    
    # returned dict is of structure:
    # {<topicId>: {
    #   'documents': {<docId>: {'text': <str>, 'keyphrases': [<str>], 'keyphrases_stemmed': [<str>]}},
    #   'summaries': {'50': [<str>], '100': [<str>], '200': [<str>], '400': [<str>]}
    # }}
    return ducData
        

def removeNearDuplicateKPs(kps, kpsStemmed):
    #kpsT = [k.split() for k in kps]
    #kpsStemmedT = [k.split() for k in kpsStemmed]
    removalIndices = []
    for i in range(len(kps)):
        for j in range(i + 1, len(kps)):
            if kpsStemmed[i] in kpsStemmed[j]:
                removalIndices.append(i)
                break
    kpsCleaned = [kps[i] for i in range(len(kps)) if i not in removalIndices]
    kpsStemmedCleaned = [kpsStemmed[i] for i in range(len(kpsStemmed)) if i not in removalIndices]
    return kpsCleaned, kpsStemmedCleaned
        


def combineKPsPerTopic(data):
    kpsPerTopic = {} # {<topicId> -> [(<kp>, <kpScore>)]}
    for topicId in data:
        
        ## KP scoring by how many times tokens appear in all KPs of all documents:
        #topicKpTokensCounter = Counter()
        #for docInfo in data[topicId]['documents'].values():
        #    allDocKpTokens = (' '.join(docInfo['keyphrases_stemmed'])).split()
        #    topicKpTokensCounter.update(allDocKpTokens)
        #    #topicKpCounter.update(docInfo['keyphrases_stemmed'])
        #topicKpTokenCountSum = sum(topicKpTokensCounter.values())
        #topicTokenScoresKP = {t: float(topicKpTokensCounter[t])/topicKpTokenCountSum for t in topicKpTokensCounter}
        #topicKpScores = {}
        #topicKPs = []
        #topicKPsStemmed = []
        #for docInfo in data[topicId]['documents'].values():
        #    for kp, kpStemmed in zip(docInfo['keyphrases'], docInfo['keyphrases_stemmed']):
        #        kpScore = mean([topicTokenScoresKP[t] for t in kpStemmed.split()])
        #        topicKpScores[kp] = kpScore
        #        if kp not in topicKPs:
        #            topicKPs.append(kp)
        #            topicKPsStemmed.append(kpStemmed)
        
        ## KP scoring by how many times tokens appear in all reference summaries:
        #topicSummTokensCounter = Counter()
        #for summLen in ['400']: #data[topicId]['summaries']:
        #    for summStr in data[topicId]['summaries'][summLen]:
        #        summaryTokens = [stemmer.stem(t) for t in word_tokenize(summStr)]
        #        topicSummTokensCounter.update(summaryTokens)
        #topicSummsTokenCountSum = sum(topicSummTokensCounter.values())
        #topicTokenScoresSumms = {t: float(topicSummTokensCounter[t])/topicSummsTokenCountSum for t in topicSummTokensCounter}
        #topicKpScores = {}
        #topicKPs = []
        #topicKPsStemmed = []
        #for docInfo in data[topicId]['documents'].values():
        #    for kp, kpStemmed in zip(docInfo['keyphrases'], docInfo['keyphrases_stemmed']):
        #        kpScore = mean([topicTokenScoresSumms[t] if t in topicTokenScoresSumms else 0. for t in kpStemmed.split()])
        #        topicKpScores[kp] = kpScore
        #        if kp not in topicKPs:
        #           topicKPs.append(kp)
        #           topicKPsStemmed.append(kpStemmed)
        
        
        # KP scoring by how many reference summaries tokens appear in:
        topicSummTokensCounter = Counter() # how many ref summs is it in
        for summStr in data[topicId]['summaries']['400']:
            summaryTokens = list(set([stemmer.stem(t) for t in word_tokenize(summStr.lower())])) # just to see if a token is there or not
            topicSummTokensCounter.update(summaryTokens)
            #print(topicSummTokensCounter)
        topicSummsCount = len(data[topicId]['summaries']['400'])
        topicTokenScoresSumms = {t: float(topicSummTokensCounter[t])/topicSummsCount for t in topicSummTokensCounter}
        #print(topicTokenScoresSumms)
        topicKpScores = {}
        topicKPs = []
        topicKPsStemmed = []
        for docInfo in data[topicId]['documents'].values():
            for kp, kpStemmed in zip(docInfo['keyphrases'], docInfo['keyphrases_stemmed']):
                kpScore = mean([topicTokenScoresSumms[t] if t in topicTokenScoresSumms else 0. for t in kpStemmed.split()])
                topicKpScores[kp] = kpScore
                if kp not in topicKPs:
                    topicKPs.append(kp)
                    topicKPsStemmed.append(kpStemmed)
                
        # get rid of near duplicate or contained KPs:
        topicKPsCleaned, topicKPsStemmedCleaned = removeNearDuplicateKPs(topicKPs, topicKPsStemmed)
        
        #print('------------------------------')
        #print(set(topicKPs) ^ set(topicKPsCleaned)) # the removed KPs
        #print(f'------------------ {topicId}')
        #print({k: v for k, v in sorted(topicKpScores.items(), key=lambda item: item[1], reverse=True) if k in topicKPsCleaned})
        
        # the topic's sorted list of (kp, score) tuples for the KPs left in topicKPsCleaned, whose score is above 0:
        kpsPerTopic[topicId] = [(kpTxt, kpScore) for kpTxt, kpScore in
                                sorted(topicKpScores.items(), key=lambda item: item[1], reverse=True)
                                if kpTxt in topicKPsCleaned and kpScore > 0]
    
    return kpsPerTopic
        
    

def getDUC01KeyphraseDataset():
    data = loadData(DUC2001_KP_SINGLEDOC_JSON_PATH, DUC2001_JSONL_PATH_TEMPLATE, SUMM_LENS)
    if data:
        #print(json.dumps(data, indent=2))
        kpsPerTopic = combineKPsPerTopic(data)
        docsPerTopic = {topicId: [data[topicId]['documents'][docId]['text'] for docId in data[topicId]['documents']] for topicId in data}
        refsPerTopic = {topicId: data[topicId]['summaries']['400'] for topicId in data}
        returnedDataPerTopic = {}
        for topicId in docsPerTopic:
            returnedDataPerTopic[topicId] = {
                'documents': docsPerTopic[topicId],
                'references': refsPerTopic[topicId],
                'keyphrases': kpsPerTopic[topicId]}
    else:
        returnedDataPerTopic = None
        print('Error loading data. Exiting')
    
    return returnedDataPerTopic


def prepare_data_in_datadir(base_test_data_dir, split_name):
    # create the temp folder for the test data in the expected structure in data_manage:
    test_basepath = os.path.join(base_test_data_dir, split_name)
    if os.path.exists(test_basepath):
        shutil.rmtree(test_basepath)
    os.makedirs(test_basepath)
    topics_folderpath = os.path.join(test_basepath, 'topics')
    samples_folderpath = os.path.join(test_basepath, 'samples')
    os.makedirs(topics_folderpath)
    os.makedirs(samples_folderpath)

    # get the KP data (documents and KPs per topic):
    dataPerTopic = getDUC01KeyphraseDataset()

    for topicId in dataPerTopic:
        topicName = f'2001_{topicId}'
        docs_sents = [[' '.join(word_tokenize(s)) for s in sent_tokenize(docTxt)] for docTxt in dataPerTopic[topicId]['documents']]
        refs_sents = [[' '.join(word_tokenize(s)) for s in sent_tokenize(docTxt)] for docTxt in dataPerTopic[topicId]['references']]
        doc_kps = dataPerTopic[topicId]['keyphrases']
        topicDict = {'id': topicName, 'documents': docs_sents, 'reference_summaries': refs_sents, 'reference_kps': doc_kps}
        with open(os.path.join(topics_folderpath, topicName + '.json'), 'w') as fOut:
            json.dump(topicDict, fOut, indent=4)

        sample_id = f'{topicName}_none'
        sample_dict = {'id': sample_id, 'topic_id': topicName, 'method': 'none',
                       'initial_summary_sentence_ids': [], "queries": []}
        with open(os.path.join(samples_folderpath, sample_id + '.json'), 'w') as fOut:
            json.dump(sample_dict, fOut, indent=4)

    return test_basepath


if __name__ == '__main__':
    getDUC01KeyphraseDataset()