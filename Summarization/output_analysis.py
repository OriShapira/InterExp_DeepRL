import sys
import numpy as np
import json
import os
from collections import defaultdict
from sklearn.metrics import auc


# We create here a CSV for the results in the eval_sessions and eval_sessions_generic json files specified.
# These are simulated sessions run on the the DUC 2006 collected sessions from the NAACL paper.
# At each iteration, the previous summary-so-far is fed to the model, and the model outputs an expansion with the next query.
# We therefore cannot compute AUC of the ROUGE recall curve here, since each iteration is a test on its own.
# We look at the average delta-ROUGE over all iterations and all sessions and all topics for the model.
# We also look at the query-to-expansion similarity on average over all iterations.


def avgScoresInitial(scoresDictList):
    metrics = list(scoresDictList[0].keys())
    scoreTypes = list(scoresDictList[0][metrics[0]].keys())
    avgScores = {metric : {scoreType : np.mean([s[metric][scoreType] for s in scoresDictList])
                           for scoreType in scoreTypes} for metric in metrics}
    return avgScores


def normalizeScoresInitial(scoresDictList, lenDictList):
    metrics = list(scoresDictList[0].keys())
    scoreTypes = list(scoresDictList[0][metrics[0]].keys())
    avgScores = {metric: {scoreType: np.mean([s[metric][scoreType] / l for s, l in zip(scoresDictList, lenDictList)])
                          for scoreType in scoreTypes} for metric in metrics}
    return avgScores


def avgScoresExpansion(scoresDictList):
    metrics = list(scoresDictList[0].keys())
    avgScores = {metric : np.mean([s[metric] for s in scoresDictList]) for metric in metrics}
    return avgScores


def normalizeScoresExpansion(scoresDictList, lenDictList):
    metrics = list(scoresDictList[0].keys())
    avgScores = {metric : np.mean([s[metric] / l for s, l in zip(scoresDictList, lenDictList)]) for metric in metrics}
    return avgScores


def outputOverallComparison(allSessionsGeneratedDict, genericScoresDict, outputCsvPath):
    # computes overall stats on the delta scores and lengths
    # adds the information to the given dictionary as a new 'overall' entry

    # get the numbers per session:
    initialSummScore = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) # algo -> algoType -> topicId -> list of scores_dict
    initialSummLen = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) # algo -> algoType -> topicId -> list of token_length
    expansionDeltaScore = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # algo -> algoType -> topicId -> list of avg_scores_dict (over iterations)
    expansionLen = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # algo -> algoType -> topicId -> list of token_length
    expansionQueryScore = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # algo -> algoType -> topicId -> list of avg_query_score (over iterations)
    #aucScore = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) # algo -> algoType -> topicId -> list of list of aucScores_dict (each session has a list of accumulating AUC scores)
    #aucLowerLimit, aucUpperLimit = getAucLimits(allSessionsGeneratedDict) # get the AUC limits on which to compute overall AUC scores:
    for topicId in allSessionsGeneratedDict:
        for sessionInfo in allSessionsGeneratedDict[topicId]:
            algo = sessionInfo['original_algo']
            #for iterIdx, iterInfo in enumerate(sessionInfo['RLMMR']):
            #    if iterIdx == 0:
            #        initialSummScore[algo]['RLMMR'][topicId].append(sessionInfo['RLMMR'][iterIdx]['scores'])
            #        initialSummScore[algo]['original'][topicId].append(sessionInfo['original'][iterIdx]['scores'])
            #        initialSummLen[algo]['RLMMR'][topicId].append(sessionInfo['RLMMR'][iterIdx]['length'])
            #        initialSummLen[algo]['original'][topicId].append(sessionInfo['original'][iterIdx]['length'])
            #    else:
            #        expansionDeltaScore[algo]['RLMMR'][topicId].append(sessionInfo['RLMMR'][iterIdx]['delta'])
            #        expansionDeltaScore[algo]['original'][topicId].append(sessionInfo['original'][iterIdx]['delta'])
            #        expansionLen[algo]['RLMMR'][topicId].append(sessionInfo['RLMMR'][iterIdx]['delta_length'])
            #        expansionLen[algo]['original'][topicId].append(sessionInfo['original'][iterIdx]['delta_length'])
            initialSummScore[algo]['RLMMR'][topicId].append(sessionInfo['RLMMR'][0]['scores'])
            initialSummScore[algo]['original'][topicId].append(sessionInfo['original'][0]['scores'])
            initialSummLen[algo]['RLMMR'][topicId].append(sessionInfo['RLMMR'][0]['length'])
            initialSummLen[algo]['original'][topicId].append(sessionInfo['original'][0]['length'])
            # keep an averaged score dict over the iterations of the current session:
            expansionDeltaScore[algo]['RLMMR'][topicId].append(avgScoresExpansion([iterInfo['delta'] for iterInfo in sessionInfo['RLMMR'][1:]]))
            expansionDeltaScore[algo]['original'][topicId].append(avgScoresExpansion([iterInfo['delta'] for iterInfo in sessionInfo['original'][1:]]))
            expansionLen[algo]['RLMMR'][topicId].append(np.mean([iterInfo['delta_length'] for iterInfo in sessionInfo['RLMMR'][1:]]))
            expansionLen[algo]['original'][topicId].append(np.mean([iterInfo['delta_length'] for iterInfo in sessionInfo['original'][1:]]))
            expansionQueryScore[algo]['RLMMR'][topicId].append(np.mean([np.mean(iterInfo['scores_query']) for iterInfo in sessionInfo['RLMMR'][1:]]))
            expansionQueryScore[algo]['original'][topicId].append(np.mean([np.mean(iterInfo['scores_query']) for iterInfo in sessionInfo['original'][1:]]))
            #aucScore[algo]['RLMMR'][topicId].append(computeAUCInterpolated(sessionInfo['RLMMR'], aucLowerLimit, aucUpperLimit))
            #aucScore[algo]['original'][topicId].append(computeAUCInterpolated(sessionInfo['original'], aucLowerLimit, aucUpperLimit))

    # get the numbers per topic:
    initialSummScoreAvg = defaultdict(lambda: defaultdict(lambda: defaultdict()))  # algo -> algoType -> topicId -> avg_scores_dict
    initialSummLenAvg = defaultdict(lambda: defaultdict(lambda: defaultdict()))  # algo -> algoType -> topicId -> avg_token_length
    initialSummScoreAvgNorm = defaultdict(lambda: defaultdict(lambda: defaultdict()))  # algo -> algoType -> topicId -> avg_scores_dict_normalized
    expansionDeltaScoreAvg = defaultdict(lambda: defaultdict(lambda: defaultdict()))  # algo -> algoType -> topicId -> avg_scores_dict
    expansionLenAvg = defaultdict(lambda: defaultdict(lambda: defaultdict()))  # algo -> algoType -> topicId -> avg_token_length
    expansionQueryScoreAvg = defaultdict(lambda: defaultdict(lambda: defaultdict()))  # algo -> algoType -> topicId -> avg_score
    expansionDeltaScoreAvgNorm = defaultdict(lambda: defaultdict(lambda: defaultdict()))  # algo -> algoType -> topicId -> avg_scores_dict_normalized
    #aucScoreAvg = defaultdict(lambda: defaultdict(lambda: defaultdict()))  # algo -> algoType -> topicId -> avg_auc_scores_dict
    for algo in initialSummScore:
        for algoType in initialSummScore[algo]:
            for topicId in initialSummScore[algo][algoType]:
                initialSummScoreAvg[algo][algoType][topicId] = avgScoresInitial(initialSummScore[algo][algoType][topicId])
                initialSummLenAvg[algo][algoType][topicId] = np.mean(initialSummLen[algo][algoType][topicId])
                initialSummScoreAvgNorm[algo][algoType][topicId] = normalizeScoresInitial(initialSummScore[algo][algoType][topicId], initialSummLen[algo][algoType][topicId])
                expansionDeltaScoreAvg[algo][algoType][topicId] = avgScoresExpansion(expansionDeltaScore[algo][algoType][topicId])
                expansionLenAvg[algo][algoType][topicId] = np.mean(expansionLen[algo][algoType][topicId])
                expansionQueryScoreAvg[algo][algoType][topicId] = np.mean(expansionQueryScore[algo][algoType][topicId])
                expansionDeltaScoreAvgNorm[algo][algoType][topicId] = normalizeScoresExpansion(expansionDeltaScore[algo][algoType][topicId], expansionLen[algo][algoType][topicId])
                ## for AUC, the session-level scores (in aucScore) already did the interpolation and raw AUC number computation,
                ## therefore we just need to average the numbers now since they are all in the same limits:
                #aucScoreAvg[algo][algoType][topicId] = np.mean(aucScore[algo][algoType][topicId])

    # get the overall numbers:
    initialSummOverallScore = defaultdict(lambda: defaultdict())  # algo -> algoType -> scores_dict
    initialSummOverallLen = defaultdict(lambda: defaultdict())  # algo -> algoType -> token_length
    initialSummOverallScoreNorm = defaultdict(lambda: defaultdict())  # algo -> algoType -> scores_dict_normalized
    expansionOverallDeltaScore = defaultdict(lambda: defaultdict())  # algo -> algoType -> scores_dict
    expansionOverallLen = defaultdict(lambda: defaultdict())  # algo -> algoType -> token_length
    expansionOverallQueryScore = defaultdict(lambda: defaultdict())  # algo -> algoType -> score
    expansionOverallDeltaScoreNorm = defaultdict(lambda: defaultdict()) # algo -> algoType -> scores_dict_normalized
    #aucOverallScore = defaultdict(lambda: defaultdict())  # algo -> algoType -> auc_scores_dict
    for algo in initialSummScoreAvg:
        for algoType in initialSummScoreAvg[algo]:
            initialSummOverallScore[algo][algoType] = avgScoresInitial(list(initialSummScoreAvg[algo][algoType].values()))
            initialSummOverallLen[algo][algoType] = np.mean(list(initialSummLenAvg[algo][algoType].values()))
            initialSummOverallScoreNorm[algo][algoType] = normalizeScoresInitial(list(initialSummScoreAvg[algo][algoType].values()), list(initialSummLenAvg[algo][algoType].values()))
            expansionOverallDeltaScore[algo][algoType] = avgScoresExpansion(list(expansionDeltaScoreAvg[algo][algoType].values()))
            expansionOverallLen[algo][algoType] = np.mean(list(expansionLenAvg[algo][algoType].values()))
            expansionOverallQueryScore[algo][algoType] = np.mean(list(expansionQueryScoreAvg[algo][algoType].values()))
            expansionOverallDeltaScoreNorm[algo][algoType] = normalizeScoresExpansion(list(expansionDeltaScoreAvg[algo][algoType].values()), list(expansionLenAvg[algo][algoType].values()))
            #aucOverallScore[algo][algoType] = np.mean(list(aucScoreAvg[algo][algoType].values()))


    # output the data to the csv file:
    topicIds = sorted(list(allSessionsGeneratedDict.keys()))
    algos = sorted(list(initialSummOverallScore.keys()))
    metrics = ['R1', 'R2', 'RL', 'RSU4']
    outputStr = ''
    for metric in metrics:
        outputStr += f'Initial Summary {metric}\ntopicId,,'
        for algo in algos:
            outputStr += f'{algo}_{metric}f,RLMMR_{metric}f,{algo}_len,RLMMR_len,{algo}_{metric}f_norm,RLMMR_{metric}f_norm,,'
        outputStr = outputStr[:-2] + '\n'
        for topicId in topicIds:
            outputStr += f'{topicId},,'
            for algo in algos:
                outputStr += f'{initialSummScoreAvg[algo]["original"][topicId][metric]["f1"]},' \
                             f'{initialSummScoreAvg[algo]["RLMMR"][topicId][metric]["f1"]},' \
                             f'{initialSummLenAvg[algo]["original"][topicId]},' \
                             f'{initialSummLenAvg[algo]["RLMMR"][topicId]},' \
                             f'{initialSummScoreAvgNorm[algo]["original"][topicId][metric]["f1"]},' \
                             f'{initialSummScoreAvgNorm[algo]["RLMMR"][topicId][metric]["f1"]},,'
            outputStr = outputStr[:-2] + '\n'
        outputStr += f'overall,,'
        for algo in algos:
            outputStr += f'{initialSummOverallScore[algo]["original"][metric]["f1"]},' \
                         f'{initialSummOverallScore[algo]["RLMMR"][metric]["f1"]},' \
                         f'{initialSummOverallLen[algo]["original"]},' \
                         f'{initialSummOverallLen[algo]["RLMMR"]},' \
                         f'{initialSummOverallScoreNorm[algo]["original"][metric]["f1"]},' \
                         f'{initialSummOverallScoreNorm[algo]["RLMMR"][metric]["f1"]},,'
        outputStr = outputStr[:-2] + '\n\n'

        outputStr += f'Expansion {metric}\ntopicId,,'
        for algo in algos:
            outputStr += f'{algo}_d{metric}r,RLMMR_d{metric}r,{algo}_len,RLMMR_len,{algo}_d{metric}r_norm,RLMMR_d{metric}r_norm,{algo}_qScore,RLMMR_qScore,,'
        outputStr = outputStr[:-2] + '\n'
        for topicId in topicIds:
            outputStr += f'{topicId},,'
            for algo in algos:
                outputStr += f'{expansionDeltaScoreAvg[algo]["original"][topicId][metric]},' \
                             f'{expansionDeltaScoreAvg[algo]["RLMMR"][topicId][metric]},' \
                             f'{expansionLenAvg[algo]["original"][topicId]},' \
                             f'{expansionLenAvg[algo]["RLMMR"][topicId]},' \
                             f'{expansionDeltaScoreAvgNorm[algo]["original"][topicId][metric]},' \
                             f'{expansionDeltaScoreAvgNorm[algo]["RLMMR"][topicId][metric]},' \
                             f'{expansionQueryScoreAvg[algo]["original"][topicId]},' \
                             f'{expansionQueryScoreAvg[algo]["RLMMR"][topicId]},,'
            outputStr = outputStr[:-2] + '\n'
        outputStr += f'overall,,'
        for algo in algos:
            outputStr += f'{expansionOverallDeltaScore[algo]["original"][metric]},' \
                         f'{expansionOverallDeltaScore[algo]["RLMMR"][metric]},' \
                         f'{expansionOverallLen[algo]["original"]},' \
                         f'{expansionOverallLen[algo]["RLMMR"]},' \
                         f'{expansionOverallDeltaScoreNorm[algo]["original"][metric]},' \
                         f'{expansionOverallDeltaScoreNorm[algo]["RLMMR"][metric]},' \
                         f'{expansionOverallQueryScore[algo]["original"]},' \
                         f'{expansionOverallQueryScore[algo]["RLMMR"]},,'
        outputStr = outputStr[:-2] + '\n\n'
        
        #outputStr += f'AUC {metric},{aucLowerLimit},{aucUpperLimit}\ntopicId,,'
        #for algo in algos:
        #    outputStr += f'{algo}_AUC_{metric}r,RLMMR_AUC_{metric}r,,'
        #outputStr = outputStr[:-2] + '\n'
        #for topicId in topicIds:
        #    outputStr += f'{topicId},,'
        #    for algo in algos:
        #        outputStr += f'{aucScoreAvg[algo]["original"][topicId][metric]},' \
        #                     f'{aucScoreAvg[algo]["RLMMR"][topicId][metric]},,'
        #    outputStr = outputStr[:-2] + '\n'
        #outputStr += f'overall,,'
        #for algo in algos:
        #    outputStr += f'{aucOverallScore[algo]["original"][metric]},' \
        #                 f'{aucOverallScore[algo]["RLMMR"][metric]},,'
        #outputStr = outputStr[:-2] + '\n\n'



    if genericScoresDict != None:
        outputStr += f'Generic Summaries Scores\n'
        # read in the generic summaries scores:
        genericSummInfo = defaultdict(lambda: defaultdict(lambda: dict))  # topicId -> summLen -> scores_dict
        for topicId in genericScoresDict:
            for scoreInfo in genericScoresDict[topicId]['scores_info']:
                summLen = scoreInfo['len']
                genericSummInfo[topicId][summLen] = scoreInfo['scores']
        # write out the column headers for the generic scores:
        summLens = sorted(list(genericSummInfo[topicId]))
        outputStr += f'topicId,,'
        for summLen in summLens:
            outputStr += f'{summLen}_R1f,{summLen}_R1r,{summLen}_R2f,{summLen}_R2r,{summLen}_RLf,{summLen}_RLr,{summLen}_RSU4f,{summLen}_RSU4r,,'
        outputStr = outputStr[:-2] + '\n'
        # write out the scores per topic:
        for topicId in topicIds:
            outputStr += f'{topicId},,'
            for summLen in summLens:
                outputStr += f'{genericSummInfo[topicId][summLen]["R1"]["f1"]},' \
                             f'{genericSummInfo[topicId][summLen]["R1"]["recall"]},' \
                             f'{genericSummInfo[topicId][summLen]["R2"]["f1"]},' \
                             f'{genericSummInfo[topicId][summLen]["R2"]["recall"]},' \
                             f'{genericSummInfo[topicId][summLen]["RL"]["f1"]},' \
                             f'{genericSummInfo[topicId][summLen]["RL"]["recall"]},'\
                             f'{genericSummInfo[topicId][summLen]["RSU4"]["f1"]},' \
                             f'{genericSummInfo[topicId][summLen]["RSU4"]["recall"]},,'
            outputStr = outputStr[:-2] + '\n'
        # write out the average generic scores over all topics:
        outputStr += 'overall,,'
        for summLen in summLens:
            overallScoresDict = avgScoresInitial([genericSummInfo[topicId][summLen] for topicId in genericSummInfo])
            outputStr += f'{overallScoresDict ["R1"]["f1"]},' \
                         f'{overallScoresDict["R1"]["recall"]},' \
                         f'{overallScoresDict["R2"]["f1"]},' \
                         f'{overallScoresDict["R2"]["recall"]},' \
                         f'{overallScoresDict["RL"]["f1"]},' \
                         f'{overallScoresDict["RL"]["recall"]},' \
                         f'{overallScoresDict["RSU4"]["f1"]},' \
                         f'{overallScoresDict["RSU4"]["recall"]},,'
        outputStr = outputStr[:-2] + '\n\n'

    # write all the output to a csv file:
    with open(outputCsvPath, 'w') as fOut:
        fOut.write(outputStr)


'''
def getAucLimits(allSessionsGeneratedDict): #allAucScores):
    # allSessionsGeneratedDict is the original JSON which we are analyzing:
    # {topicId -> [{algoType -> [{"length" key, ...} per iteration]} per session]
    globalMinLimit = -1
    globalMaxLimit = 99999
    for topicId in allSessionsGeneratedDict:
        for sessionIdx, sessionInfo in enumerate(allSessionsGeneratedDict[topicId]):
            for algoType in ['RLMMR', 'original']:
                intialSumLen = sessionInfo[algoType][0]['length']
                finalSumLen = sessionInfo[algoType][-1]['length']
                globalMinLimit = max(globalMinLimit, intialSumLen)
                globalMaxLimit = min(globalMaxLimit, finalSumLen)
    return globalMinLimit, globalMaxLimit

def computeAUCInterpolated(sessionScores, aucLowerLimit, aucUpperLimit):
    # sessionScores: [{'scores':{'R1': {'recall':<>,'precision':<>,'f1':<>}, 'R2': {...}, 'RL': {...}, 'RSU4': {...}}, 'length':<int>} per iteration]
    # Get the AUC score of the session given within the length limits specified, per metric
    
    def getInterpolatedYval(xList, yList, idxStart, neededXval):
        # gets the y value at the neededXval, between idxStart and idxStart+1:
        return np.interp(neededXval, xList[idxStart:idxStart+2], yList[idxStart:idxStart+2])
    
    metrics = ['R1', 'R2', 'RL', 'RSU4']
    xList = [iterInfo['length'] for iterInfo in sessionScores]
    yListPerMetric = {metric: [iterInfo['scores'][metric]['recall'] for iterInfo in sessionScores] for metric in metrics}
    
    # create new lists of X (length) and Y (ROUGE score), only within the lower and upper X limits (aucLowerLimit, aucUpperLimit)
    # interpolate the Y values at the bounds when necessary:
    newXList = []
    newYListPerMetric = {metric: [] for metric in metrics}
    for xIdx, x in enumerate(xList):
        # if the current x value is the one right before the aucLowerLimit, then get the interpolated y value:
        if x < aucLowerLimit and xList[xIdx + 1] > aucLowerLimit:
            newXList.append(aucLowerLimit)
            for metric in metrics:
                newYListPerMetric[metric].append(getInterpolatedYval(xList, yListPerMetric[metric], xIdx, aucLowerLimit))
        # if the current x value is the one right after the aucUpperLimit, then get the interpolated y value:
        elif x > aucUpperLimit and xList[xIdx - 1] < aucUpperLimit:
            newXList.append(aucUpperLimit)
            for metric in metrics:
                newYListPerMetric[metric].append(getInterpolatedYval(xList, yListPerMetric[metric], xIdx - 1, aucUpperLimit))
        # if the current x value is the exact aucLowerLimit or aucUpperLimit or between the two, keep the original values as is:
        elif x >= aucLowerLimit and x <= aucUpperLimit:
            newXList.append(x)
            for metric in metrics:
                newYListPerMetric[metric].append(yListPerMetric[metric][xIdx])
                
    print(aucLowerLimit, aucUpperLimit)
    print(xList)
    print(newXList)
    
    # now just compute the AUC on the scores with limit interpolations:
    aucScores = {}
    for metric in metrics:
        aucScores[metric] = auc(newXList, newYListPerMetric[metric])
    
    return aucScores
'''
        


if __name__ == '__main__':
    # expected argument: path to the JSON as output by test_session_data.py
    if len(sys.argv) > 1:
        inputScoresFilePath = sys.argv[1]
        PATH_LIST = [inputScoresFilePath]
    else:
        # example list of JSONs to analyze:
        PATH_LIST = [
            'saved_model/Apr12_23-25-48/eval_sessions_30_30.json',
            'saved_model/Apr22_00-26-13/eval_sessions_30_30.json'
        ]

    for path in PATH_LIST:
        print(path)
        if not os.path.exists(path):
            print('\tSkipping because cannot find the file.')
            continue
        else:
            print('\tProcessing...')
        with open(path, 'r') as fIn:
            allSessionsGeneratedDict = json.load(fIn)

        genericScoresFilePath = path[:-5] + '_generic.json'
        if os.path.exists(genericScoresFilePath):
            with open(genericScoresFilePath, 'r') as fIn:
                genericScoresDict = json.load(fIn)
        else:
            genericScoresDict = None

        outputCsvPath = path[:-4] + 'csv'
        outputOverallComparison(allSessionsGeneratedDict, genericScoresDict, outputCsvPath)