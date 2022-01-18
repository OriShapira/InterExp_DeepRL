""" evaluation scripts"""
from os.path import join
import logging
import tempfile
import subprocess as sp
from pyrouge import Rouge155
from pyrouge.utils import log
from collections import defaultdict
from Config import ROUGE_PATH
import os


def calc_official_rouge(ref_summs, system_summaries_dir, dataset_name, summ_len_limit=250, reset_refsumms=False):
    """
    :param ref_summs: dictionary of topic_id -> list_of_strings where the list_of_strings is the reference summaries of the topic
    :param system_summaries_dir: The path to the dir of the system summaries (each system summary should be in a file
    called "<topicId>.dec").
    :param dataset_name: the name of this dataset, used for cacheing for faster processing
    :param summ_len_limit: The ROUGE truncation point (in token length).
    :return: Scores dictionary with keys [R1, R2, RL, RSU4] and subkeys
    [recall, recall_lower, recall_upper, f1, f1_lower, f1_upper, precision, precision_lower, precision_upper]
    """
    print(f'calc_official_rouge: dataset={dataset_name}')
    if reset_refsumms:
        global _ref_summs_rouge_format_dirpath
        _ref_summs_rouge_format_dirpath = defaultdict(lambda: {})
    output = eval_rouge(ref_summs, system_summaries_dir, dataset_name, summ_len_limit)
    # print(output)
    R = parse_rouge_output(output)
    print(R, '\n')
    return R

def parse_rouge_output(output):
    def parse_rouge_line(line):
        parts = line.split()
        score, lower, upper = float(parts[3]), float(parts[5]), float(parts[7].split(')')[0])
        return score, lower, upper

    for line in output.split('\n'):
        if line.startswith('1 ROUGE-1 Average_F'):
            r1f, r1flow, r1fupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-1 Average_R'):
            r1r, r1rlow, r1rupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-1 Average_P'):
            r1p, r1plow, r1pupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-2 Average_F'):
            r2f, r2flow, r2fupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-2 Average_R'):
            r2r, r2rlow, r2rupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-2 Average_P'):
            r2p, r2plow, r2pupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-L Average_F'):
            rlf, rlflow, rlfupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-L Average_R'):
            rlr, rlrlow, rlrupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-L Average_P'):
            rlp, rlplow, rlpupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-SU4 Average_F'):
            rsu4f, rsu4flow, rsu4fupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-SU4 Average_R'):
            rsu4r, rsu4rlow, rsu4rupp = parse_rouge_line(line)
        elif line.startswith('1 ROUGE-SU4 Average_P'):
            rsu4p, rsu4plow, rsu4pupp = parse_rouge_line(line)
    R = {'R1': {'recall': r1r, 'recall_lower': r1rlow, 'recall_upper': r1rupp, 'f1': r1f, 'f1_lower': r1flow, 'f1_upper': r1fupp, 'precision': r1p, 'precision_lower': r1plow, 'precision_upper': r1pupp},
         'R2': {'recall': r2r, 'recall_lower': r2rlow, 'recall_upper': r2rupp, 'f1': r2f, 'f1_lower': r2flow, 'f1_upper': r2fupp, 'precision': r2p, 'precision_lower': r2plow, 'precision_upper': r2pupp},
         'RL': {'recall': rlr, 'recall_lower': rlrlow, 'recall_upper': rlrupp, 'f1': rlf, 'f1_lower': rlflow, 'f1_upper': rlfupp, 'precision': rlp, 'precision_lower': rlplow, 'precision_upper': rlpupp},
         'RSU4': {'recall': rsu4r, 'recall_lower': rsu4rlow, 'recall_upper': rsu4rupp, 'f1': rsu4f, 'f1_lower': rsu4flow, 'f1_upper': rsu4fupp, 'precision': rsu4p, 'precision_lower': rsu4plow, 'precision_upper': rsu4pupp}}
    return R

_ref_summs_rouge_format_dirpath = defaultdict(lambda: {})
def eval_rouge(ref_summs, system_summaries_dir, dataset_name, summ_len_limit, cmd=None, system_id=1):
    """
    Evaluate with the original Perl implementation.
    :param ref_summs: dictionary of topic_id -> list_of_strings where the list_of_strings is the reference summaries of the topic
    :param system_summaries_dir: system_summaries_dir: The path to the dir of the system summaries (each system summary should be in a file
    called "<topicId>.dec").
    :param dataset_name: the name of this dataset, used for cacheing for faster processing
    :param summ_len_limit: The ROUGE truncation point (in token length).
    :param cmd: The ROUGE command, None if to use the default.
    '-c 95 -2 4 -U -r 1000 -n 2 -l {summ_len_limit} -m'
    :param system_id: An ID to give the system. Not really important.
    :return: A string with the results from the Perl script.
    """
    # silence pyrouge logging
    log.get_global_console_logger().setLevel(logging.WARNING)

    if dataset_name not in _ref_summs_rouge_format_dirpath:
        with tempfile.TemporaryDirectory() as refSummsTextTmpDirname:
            # output the reference summaries to a temp dir:
            for topicId, refSumms in ref_summs.items():
                for refSummInd, refSumm in enumerate(refSumms):
                    refFilepath = join(refSummsTextTmpDirname, f'{topicId}.{chr(refSummInd+65)}.ref')
                    with open(refFilepath, 'w', encoding='utf-8') as fRefOut:
                        fRefOut.write(refSumm)
            # convert the reference summaries to the ROUGE (SEE) format in a temporary dir (to be kept for reuse):
            refSummsForRougeTmpDirpath = tempfile.mkdtemp(prefix='QRLMMR_refsumms_')
            Rouge155.convert_summaries_to_rouge_format(refSummsTextTmpDirname, refSummsForRougeTmpDirpath)
            _ref_summs_rouge_format_dirpath[dataset_name] = refSummsForRougeTmpDirpath


    sys_summ_pattern = r'(.*).dec'
    ref_summ_pattern = '#ID#.[A-Z].ref'

    # put the system summaries in a temp dir for ROUGE:
    with tempfile.TemporaryDirectory(prefix='QRLMMR_syssumms_') as sysSummsTmpDirname:
        Rouge155.convert_summaries_to_rouge_format(system_summaries_dir, sysSummsTmpDirname)
        Rouge155.write_config_static(
            sysSummsTmpDirname, sys_summ_pattern,
            _ref_summs_rouge_format_dirpath[dataset_name], ref_summ_pattern,
            join(sysSummsTmpDirname, 'settings.xml'), system_id
        )
        base_cmd = cmd if cmd != None else f'-c 95 -2 4 -U -r 1000 -n 2 -l {summ_len_limit} -m'
        cmd = ('perl ' + join(ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(join(ROUGE_PATH, 'data'))
               + base_cmd
               + ' -a {}'.format(join(sysSummsTmpDirname, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output

'''
EXAMPLE OUTPUT from ROUGE command:
---------------------------------------------
1 ROUGE-1 Average_R: 0.33603 (95%-conf.int. 0.32136 - 0.35077)
1 ROUGE-1 Average_P: 0.33046 (95%-conf.int. 0.31588 - 0.34507)
1 ROUGE-1 Average_F: 0.33313 (95%-conf.int. 0.31864 - 0.34755)
---------------------------------------------
1 ROUGE-2 Average_R: 0.05456 (95%-conf.int. 0.04747 - 0.06215)
1 ROUGE-2 Average_P: 0.05366 (95%-conf.int. 0.04678 - 0.06125)
1 ROUGE-2 Average_F: 0.05409 (95%-conf.int. 0.04708 - 0.06168)
---------------------------------------------
1 ROUGE-L Average_R: 0.14935 (95%-conf.int. 0.14278 - 0.15575)
1 ROUGE-L Average_P: 0.14684 (95%-conf.int. 0.14023 - 0.15306)
1 ROUGE-L Average_F: 0.14805 (95%-conf.int. 0.14160 - 0.15435)
---------------------------------------------
1 ROUGE-S4 Average_R: 0.06091 (95%-conf.int. 0.05482 - 0.06737)
1 ROUGE-S4 Average_P: 0.05988 (95%-conf.int. 0.05393 - 0.06630)
1 ROUGE-S4 Average_F: 0.06038 (95%-conf.int. 0.05431 - 0.06688)
---------------------------------------------
1 ROUGE-SU4 Average_R: 0.10702 (95%-conf.int. 0.09973 - 0.11413)
1 ROUGE-SU4 Average_P: 0.10522 (95%-conf.int. 0.09811 - 0.11238)
1 ROUGE-SU4 Average_F: 0.10608 (95%-conf.int. 0.09878 - 0.11314)
'''


def getRougeScores(systemSummaryText, referenceSummariesFolderpath, limitLengthWords=-1,
                   referenceSummariesNamePattern='.*'):
    """
    Evaluate with the original Perl implementation.
    :param ref_summs: dictionary of topic_id -> list_of_strings where the list_of_strings is the reference summaries of the topic
    :param system_summaries_dir: system_summaries_dir: The path to the dir of the system summaries (each system summary should be in a file
    called "<topicId>.dec").
    :param dataset_name: the name of this dataset, used for cacheing for faster processing
    :param summ_len_limit: The ROUGE truncation point (in token length).
    :param cmd: The ROUGE command, None if to use the default.
    '-c 95 -2 4 -U -r 1000 -n 2 -l {summ_len_limit} -m'
    :param system_id: An ID to give the system. Not really important.
    :return: A string with the results from the Perl script.
    """
    # silence pyrouge logging
    log.get_global_console_logger().setLevel(logging.WARNING)

    system_summaries_dir = tempfile.mkdtemp()
    if not os.path.exists(system_summaries_dir):
        os.makedirs(system_summaries_dir)
    sysSummFilename = "systemSummary.txt"
    tempFilePath = os.path.join(system_summaries_dir, sysSummFilename)
    with open(tempFilePath, 'w') as tmp:
        # do stuff with temp file
        tmp.write(systemSummaryText)

    sys_summ_pattern = '(.*)'
    ref_summ_pattern = referenceSummariesNamePattern

    # put the system summaries in a temp dir for ROUGE:
    with tempfile.TemporaryDirectory(prefix='QRLMMR_syssumms_') as sysSummsTmpDirname:
        Rouge155.convert_summaries_to_rouge_format(system_summaries_dir, sysSummsTmpDirname)
        Rouge155.write_config_static(
            sysSummsTmpDirname, sys_summ_pattern,
            referenceSummariesFolderpath, ref_summ_pattern,
            join(sysSummsTmpDirname, 'settings.xml'), 1
        )
        if limitLengthWords > 0:
            base_cmd = f'-c 95 -2 4 -U -r 1000 -n 2 -l {limitLengthWords} -m'
        else:
            base_cmd = f'-c 95 -2 4 -U -r 1000 -n 2 -m'
        cmd = ('perl ' + join(ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(join(ROUGE_PATH, 'data'))
               + base_cmd
               + ' -a {}'.format(join(sysSummsTmpDirname, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)

    rouge_scores = parse_rouge_output(output)
    return rouge_scores