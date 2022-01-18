#####################################################
# For testing a trained model on sessions already collected
# or on session simulations. I.e., gets a snapshot and a query,
# and outputs the next expansion. See bottom of file
# for explanation on how to run the script.
#####################################################

import sys
#sys.path.append('../../../')
import os
import json
from nltk import word_tokenize
import tempfile
from decoding import RLExtractor
from dataset.utils import prepare_data_for_model_input
import torch
from evaluate import getRougeScores
from metric import compute_similarity_query_to_text
from cytoolz import concat
import argparse
from pyrouge import Rouge155
from utils import str2bool

def loadCorpus(data_dir):
    documentsDict = {} # topicId -> [list of list of strings] (sentences in documents)
    refSummsDict = {} # topicId -> [list of list of strings] (sentences in references)
    for topicFilename in os.listdir(data_dir):
        if topicFilename.endswith('.json'):
            topicId = topicFilename[:-5]  # remove the '.json' from the end
            with open(os.path.join(data_dir, topicFilename)) as fIn:
                topicData = json.load(fIn)
            documentsDict[topicId] = topicData['documents']
            refSummsDict[topicId] = topicData['reference_summaries']

    return documentsDict, refSummsDict


def normalizeSentence(sentStr):
    #s = sentStr.replace('`', "'").replace('& AMP ;', '&').replace('.', ' ')
    return sentStr.lower().replace('`', '').replace("'", '').replace('"', '').replace(',', '').replace('& amp ;', '&').replace('.', '').replace(' ', '').replace('-', '').replace('&', '').replace('hesaid', '')
    #return ' '.join(word_tokenize(s)).lower()


def normalizeSentenceDict(docsDict):
    normalizedSentenceDict = {}
    for topicId in docsDict:
        normalizedSentenceDict[topicId] = []
        for docSents in docsDict[topicId]:
            normalizedSentenceDict[topicId].append([normalizeSentence(sentStr) for sentStr in docSents])
    return normalizedSentenceDict

def almostSameStr(str1, str2):
    if len(str1) == len(str2) == 0:
        return True
    diffRatio = abs(len(str1) - len(str2)) / max(len(str1), len(str2))
    diffRatioSmallEnough = (diffRatio <= 0.9) # at least 25% overlap
    minLenLargeEnough = (min(len(str1), len(str2)) > 20) # shorter string is atleast 20 chars long
    isOneInTheOther = (str1 in str2 or str2 in str1)
    return minLenLargeEnough and isOneInTheOther and diffRatioSmallEnough


def loadRealSessionsJson(inputFilepath, sentencesDictToUseNormalized, sentencesDictToUse):

    # function to find the sentence given in the list of sentences given:
    def findSentence(docsList, sentenceStr):
        sentToFind = normalizeSentence(sentenceStr)
        for docIdx, docSents in enumerate(docsList):
            for sentIdx, sent in enumerate(docSents):
                sentOption = sent
                if almostSameStr(sentToFind, sentOption):
                    return (docIdx, sentIdx)
        print(f'ERROR: Sentence not found: "{sentToFind}"')
        return -1

    def matchTopicId(topicIdForSession, topicIds):
        for topicId in topicIds:
            if topicIdForSession in topicId:
                return topicId
        raise Exception("Couldn't find topicId in list of possible topicIds: " + topicIdForSession)

    allSessionsDict = {} # topicId -> [[(query, [])]]
    topicIdsPossible = [topicId.lower() for topicId in sentencesDictToUseNormalized.keys()]
    sentenceIdsUsed = []
    with open(inputFilepath, 'r') as fIn:
        allSessionsInfo = json.load(fIn)
        sentencesDict = allSessionsInfo['sentences'] # id -> str
        for sessionInfo in allSessionsInfo['info']:
            topicIdForSession = sessionInfo['topicId'].lower()
            algoType = sessionInfo['summType']
            topicId = matchTopicId(topicIdForSession, topicIdsPossible)
            if topicId not in allSessionsDict:
                allSessionsDict[topicId] = []
            allSessionsDict[topicId].append({'algo_type': algoType, 'iter_info': []})
            for snapshotIdx, snapshotInfo in enumerate(sessionInfo['summaries']):
                queryStr = snapshotInfo['query'][0]
                summarySentStrsOrig = [sentencesDict[sentId] for sentId in snapshotInfo['summary']]
                summarySentIndices = [findSentence(sentencesDictToUseNormalized[topicId], sentStr) for sentStr in summarySentStrsOrig]
                #summarySentStrsToUse = [sentencesDictToUse[topicId][sentIdx] for sentIdx in summarySentIndices]
                #snapshotLength = snapshotInfo['rouge'][0]
                #snapshotScores = snapshotInfo['rouge'][1]
                allSessionsDict[topicId][-1]['iter_info'].append({
                    'summary_result': summarySentIndices, # list of (doc_idx, sent_idx)
                    'query_info': [queryStr, 2] if snapshotIdx > 0 else [queryStr, 5] # list of [query, num_sentences]
                })
                # keep track of the sentence ids used:
                sentenceIdsUsed.extend(summarySentIndices)

    return allSessionsDict, set(sentenceIdsUsed)


def outputDeltaRougeScores(infoList1, infoList2):
    for idx in range(min(len(infoList1), len(infoList2))):
        #for idx, (info1, info2) in enumerate(zip(infoList1, infoList2)):
        print(
            '{}\t{}\t{:.4f}\t{}\t{:.4f}'.format(idx, infoList1[idx]['length'], infoList1[idx]['scores']['R1']['recall'],
                                                infoList2[idx]['length'], infoList2[idx]['scores']['R1']['recall']))
    print()
    for idx in range(min(len(infoList1), len(infoList2))):
        #for idx, (info1, info2) in enumerate(zip(infoList1, infoList2)):
        print(f'--- {idx} "{infoList1[idx]["query"]}" ---')
        print(infoList1[idx]['summary'])
        print(f'--- "{infoList2[idx]["query"]}"')
        print(infoList2[idx]['summary'])


def computeDeltaScores(allSessionsGeneratedDict):
    # computes the delta ROUGE-recall scores and delta lengths between each iteration and the previous original iteration
    # adds the information into the given dictionary

    def getDeltaScores(scoresDictHigh, scoresDictLow):
        deltaDict = {}
        for metric in scoresDictHigh:
            deltaDict[metric] = round(scoresDictHigh[metric]['recall'] - scoresDictLow[metric]['recall'], 4)
        return deltaDict

    for topicId in allSessionsGeneratedDict:
        for sessionInfo in allSessionsGeneratedDict[topicId]:
            for iterIdx, iterInfo in enumerate(sessionInfo['RLMMR']):
                if iterIdx > 0:
                    sessionInfo['RLMMR'][iterIdx]['delta'] = \
                        getDeltaScores(sessionInfo['RLMMR'][iterIdx]['scores'], sessionInfo['original'][iterIdx - 1]['scores'])
                    sessionInfo['RLMMR'][iterIdx]['delta_length'] = \
                        sessionInfo['RLMMR'][iterIdx]['length'] - sessionInfo['original'][iterIdx - 1]['length']
                    sessionInfo['original'][iterIdx]['delta'] = \
                        getDeltaScores(sessionInfo['original'][iterIdx]['scores'], sessionInfo['original'][iterIdx - 1]['scores'])
                    sessionInfo['original'][iterIdx]['delta_length'] = \
                        sessionInfo['original'][iterIdx]['length'] - sessionInfo['original'][iterIdx - 1]['length']



def evaluate(data_dir, sessions_dir, model_dir, beta, query_encode, cuda,
             importance_mode, diversity_mode, output_json_path, max_sent_len=999, only_generic=False, reevaluate=True):

    # if this model was already evaluated, raise an exception:
    if not reevaluate and os.path.exists(output_json_path) and os.path.exists(output_json_path[:-5] + '_generic.json'):
        raise Exception('Already processed ' + output_json_path)

    # load data:
    documentsDict, refSummsDict = loadCorpus(data_dir) # the document sets + ref summs
    documentsDictNormalized = normalizeSentenceDict(documentsDict) # the document sets' sentences for easy/fast processing
    allSessionsDict, sentIdsUsed = loadRealSessionsJson(sessions_dir, documentsDictNormalized, documentsDict) # the QFSE sessions

    # for each snapshot in all the QFSE sessions, get the expansions for the different algorithms:
    allSessionsGeneratedDict = {} # topicId -> [list of {algoName -> sessionResults} per session]
    genericSummariesGeneratedDict = {} # topicId -> {'summary': text, 'scores_info': [{'len':len, 'scores': scoresDict}]}
    output_json_path_genericSumms = output_json_path[:-5] + '_generic.json' # for the generic summaries output json
    for topicId in allSessionsDict:
        #if topicId in ['2006_d0603', '2006_d0608', '2006_d0624', '2006_d0617']: # FOR DEBUG
        #    continue

        topic_docs_sents_tokens = [[word_tokenize(sent) for sent in doc] for doc in documentsDict[topicId]]

        #topic_docs = [' '.join(list(concat(doc_sents))) for doc_sents in topic_docs_sents_tokens]  # one text per document (used as the base data)
        topic_docs = [' '.join(doc_sents) for doc_sents in documentsDict[topicId]]  # one text per document (used as the base data)
        agent = RLExtractor(model_dir, topic_docs, beta, query_encode,
                            importance_mode, diversity_mode,
                            cuda=(torch.cuda.is_available() and cuda))

        allSessionsGeneratedDict[topicId] = [] # prepare for the list of sessions in the current topic
        with tempfile.TemporaryDirectory() as refSummsForRougeTmpDirpath:
            # generate SEE versions (for ROUGE) of the reference summary for reuse on all sessions of this topic:
            with tempfile.TemporaryDirectory() as refSummsTextTmpDirname:
                # write the reference summaries to a temporary folder for reuse in the current topic:
                for refSummInd, refSumm in enumerate(refSummsDict[topicId]):
                    with open(os.path.join(refSummsTextTmpDirname, f'{refSummInd}.txt'), 'w', encoding='utf-8') as fRefOut:
                        fRefOut.write(' '.join(refSumm))
                Rouge155.convert_summaries_to_rouge_format(refSummsTextTmpDirname, refSummsForRougeTmpDirpath)

            # process all the sessions of this topic:
            if not only_generic:
                for sessIdx, sessInfo in enumerate(allSessionsDict[topicId]):
                    allSessionsGeneratedDict[topicId].append({}) # for each algorithm, keep the session information
                    aggregatedOriginalSummaryIndices = [] # aggregate the snapshots' sentence indices over all iterations
                    aggregatedOriginalSummaryText = ''  # aggregate the snapshots' text over all iterations
                    sessionInfoRLMMR = []
                    sessionInfoOrig = []
                    for iterationIdx, iterationInfo in enumerate(sessInfo['iter_info']):

                        # prepare the sentences and indices for the agent:
                        topic_docs_filtered, topic_docs_filtered_flat, resulting_summ_indices_adjusted, queriesList, sent_idx_mapper = \
                            prepare_data_for_model_input(topic_docs_sents_tokens,
                                                         iterationInfo['summary_result'],
                                                         [iterationInfo['query_info']],
                                                         get_index_mapper=True,
                                                         do_not_filter_sents=sentIdsUsed)
                        queriesStrList = [qStr for qStr, qGroupIdx in queriesList]

                        chosen_sent_indices = agent(topic_docs_filtered, #datum_name=topicId,
                                                    initial_summ=aggregatedOriginalSummaryIndices,
                                                    queries_list=queriesStrList, max_sent_len=max_sent_len)

                        # combine the chosen sentences into one text
                        chosenSentences = [topic_docs_filtered_flat[idx.item()] for idx in chosen_sent_indices]
                        if iterationIdx > 0: # not the initial summary
                            expansionText = ' '.join([' '.join(sentTokens) for sentTokens in chosenSentences])
                        else: # the initial summary needs to be a little over at least 75 tokens
                            expansionText = ''
                            tokenLen = 0
                            chosenIdx = 0
                            while tokenLen <= 75:
                                #chosenSentTokens = topic_docs_filtered_flat[chosen_sent_indices[chosenIdx].item()]
                                chosenSentTokens = chosenSentences[chosenIdx]
                                expansionText += ' ' + ' '.join(chosenSentTokens)
                                tokenLen += len(chosenSentTokens)
                                chosenIdx += 1

                        # compute the similarity between the query and each chosen sentence:
                        if iterationIdx > 0:
                            queryTokens = word_tokenize(queriesStrList[0])
                            querySimScoresRLMMR = [compute_similarity_query_to_text(queryTokens, sentTokens)
                                                   for sentTokens in chosenSentences]
                            originalExtensionSentences = [topic_docs_filtered_flat[idx] for idx in resulting_summ_indices_adjusted]
                            querySimScoresOriginal = [compute_similarity_query_to_text(queryTokens, sentTokens)
                                                      for sentTokens in originalExtensionSentences]
                        else:
                            querySimScoresRLMMR = []
                            querySimScoresOriginal = []

                        # concatenate the expansion to the last snapshot and compute ROUGE:
                        summToEvaluateRLMMR = aggregatedOriginalSummaryText + ' ' + expansionText
                        rougeScoresRLMMR = getRougeScores(systemSummaryText=summToEvaluateRLMMR,
                                                          referenceSummariesFolderpath=refSummsForRougeTmpDirpath)
                        sessionInfoRLMMR.append(
                            {'length': len(word_tokenize(summToEvaluateRLMMR)), 'scores': rougeScoresRLMMR,
                             'scores_query': querySimScoresRLMMR, 'summary': summToEvaluateRLMMR, 'query': queriesStrList[0]})

                        # aggregate the last summary expansion form the *original* iteration:
                        aggregatedOriginalSummaryIndices.extend(resulting_summ_indices_adjusted)
                        aggregatedOriginalSummaryText += ' ' + ' '.join([' '.join(topic_docs_filtered_flat[idx])
                                                                         for idx in resulting_summ_indices_adjusted])
                        rougeScoresOrig = getRougeScores(systemSummaryText=aggregatedOriginalSummaryText,
                                                         referenceSummariesFolderpath=refSummsForRougeTmpDirpath)
                        sessionInfoOrig.append(
                            {'length': len(word_tokenize(aggregatedOriginalSummaryText)), 'scores': rougeScoresOrig,
                             'scores_query': querySimScoresOriginal, 'summary': aggregatedOriginalSummaryText,
                             'query': queriesStrList[0]})


                    ## output the scores and summaries of the sessions and of QRLMMR:
                    #outputDeltaRougeScores(sessionInfoRLMMR, sessionInfoOrig) #sessInfo)

                    # keep the scores:
                    allSessionsGeneratedDict[topicId][-1]['RLMMR'] = sessionInfoRLMMR
                    allSessionsGeneratedDict[topicId][-1]['original'] = sessionInfoOrig
                    allSessionsGeneratedDict[topicId][-1]['original_algo'] = sessInfo['algo_type']
                    computeDeltaScores(allSessionsGeneratedDict)

                    # output allSessionsGeneratedDict to json
                    with open(output_json_path, 'w') as fOut:
                        json.dump(allSessionsGeneratedDict, fOut, indent=2)

            # get a generic summary from the agent for this topic, and compute ROUGE at 150, 250 and 350 words:
            # prepare the sentences and indices for the agent:
            topic_docs_filtered, topic_docs_filtered_flat, resulting_summ_indices_adjusted, queriesList, sent_idx_mapper = \
                prepare_data_for_model_input(topic_docs_sents_tokens, [], [['', 12]],
                                             get_index_mapper=True, do_not_filter_sents=sentIdsUsed)
            queriesStrList = [qStr for qStr, qGroupIdx in queriesList]
            # decode sentences from RLMMR model:
            chosen_sent_indices = agent(topic_docs_filtered, initial_summ=[], queries_list=queriesStrList, max_sent_len=max_sent_len)
            chosenSentences = [topic_docs_filtered_flat[idx.item()] for idx in chosen_sent_indices]
            genericSummText = ' '.join([' '.join(sentTokens) for sentTokens in chosenSentences])
            # get scores at different lengths:
            summary_len = sum(len(s) for s in chosenSentences)
            summary_num_sents = len(chosenSentences)
            genericSummariesGeneratedDict[topicId] = {'summary': genericSummText,
                                                      'summary_len': summary_len,
                                                      'summary_num_sents': summary_num_sents,
                                                      'avg_sent_len': float(summary_len) / summary_num_sents,
                                                      'scores_info': []}
            for summLen in [75, 150, 250, 350]:
                rougeScoresGeneric = getRougeScores(systemSummaryText=genericSummText,
                                                    referenceSummariesFolderpath=refSummsForRougeTmpDirpath,
                                                    limitLengthWords=summLen)
                genericSummariesGeneratedDict[topicId]['scores_info'].append({'len': summLen, 'scores': rougeScoresGeneric})
            # output genericSummariesGeneratedDict to json
            with open(output_json_path_genericSumms, 'w') as fOut:
                json.dump(genericSummariesGeneratedDict, fOut, indent=2)

if __name__ == '__main__':
    # Notice that this script runs on a list of parameter configurations,
    # and not through the parameter passed in.

    parser = argparse.ArgumentParser(description='run decoding of the session data collected in the InterExp paper')
    parser.add_argument('--data_dir', action='store', default='dataset/DUC/test/topics',
                        help='directory of the topic data')
    parser.add_argument('--sessions_path', action='store', default='../Crowdsourcing/RealSessions/BaselineOriginal/results.json',
                        help='directory of the topic data')
    parser.add_argument('--only_generic', type=str2bool, default=False, help='only prepare the generic summary output CSV')
    parser.add_argument('--output_json_path', required=False, default='./temp', help='path to output')
    parser.add_argument('--model_dir', default='saved_model/trained/Mar23_00-26-01', help='root of the full model')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='weight for the query in the MMR score (1=noWeightForQuery, 0=fullWeightForQuery)')
    parser.add_argument('--query_encoding_in_input', type=str2bool, default=True,
                        help='should a query be encoded into the input')
    parser.add_argument('--importance_mode', action='store', default='tfidf',
                        help='MMR importance function (tfidf or w2v)')
    parser.add_argument('--diversity_mode', action='store', default='tfidf',
                        help='MMR diversity function (tfidf or w2v)')
    parser.add_argument('--no_cuda', action='store_true', help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    #evaluate(args.data_dir, args.sessions_path, args.model_dir, args.beta, args.query_encoding_in_input, args.cuda,
    #         args.importance_mode, args.diversity_mode, args.output_json_path, only_generic=args.only_generic)

    # example configurations to test:
    CONFIGS = [
        {'model_dir': 'Apr22_00-26-13', 'output_json_path': 'Apr22_00-26-13/eval_sessions_30_30.json',
         'query_encoding_in_input': False, 'beta': 0.5, 'only_generic': False, 'max_sent_len': 30},
        {'model_dir': 'Apr21_12-00-53', 'output_json_path': 'Apr21_12-00-53/eval_sessions_30_30_noQmmrInference.json',
         'query_encoding_in_input': False, 'beta': 1.0, 'only_generic': False, 'max_sent_len': 30},
    ]
    
    for c in CONFIGS:
        print(c)
        args.model_dir = 'saved_model/' + c['model_dir']
        args.output_json_path = 'saved_model/' + c['output_json_path']
        args.query_encoding_in_input = c['query_encoding_in_input']
        args.beta = c['beta']
        args.only_generic = c['only_generic']
        args.importance_mode = 'tfidf'
        args.diversity_mode = 'tfidf'
        max_sent_len = c['max_sent_len']

        try:
            evaluate(args.data_dir, args.sessions_path, args.model_dir, args.beta, args.query_encoding_in_input, args.cuda,
                     args.importance_mode, args.diversity_mode, args.output_json_path, max_sent_len=max_sent_len,
                     only_generic=args.only_generic, reevaluate=False)
        except Exception as ex:
            print('ERROR: could not process ' + c['model_dir'])
            print('error is: ' + str(ex))




    # input: a simulation file or real sessions file
    #   e.g.    ../Crowdsourcing/RealSessions/BaselineOriginal/results.json
    # procedure: for each session, read snapshot-by-snapshot and get the expansion for the next query. Then compare
    #   the delta ROUGE of the expansion. Keep one copy where the summary is aggregated, and one where the
    #   current snapshot is taken from the session file.
    # output: A json file with the outputs over all sessions of all topics.