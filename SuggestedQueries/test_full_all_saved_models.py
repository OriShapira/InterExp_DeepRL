import os
import sys
import json
from datetime import datetime
import re

def get_best_checkpoint_number(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward/score"""
    # get the step number of the checkpoint with the best score
    ckpts = os.listdir(os.path.join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    return ckpts[0].split('-')[-1] # e.g. ckpt-10.714387-1800 -> return 1800

def getSingleTestJsonOfModel(model_dir_path):
    # get the test results filepath of the best checkpoint, in case it has both test and test_kp results
    # otherwise return empty string
    testJsonPath = ''
    # find the best checkpoint step number:
    best_ckpt_num = get_best_checkpoint_number(model_dir_path, reverse=True)
    # find the test results json for that step number:
    foundTestJson = False
    for test_json_filename in os.listdir(model_dir_path):
        if f'test_scores_{best_ckpt_num}' in test_json_filename:
            foundTestJson = True
            break
    # check if the test results json has both test and test_kp keys, in which case the model was fully tested:
    if foundTestJson:
        test_json_filepath = os.path.join(model_dir_path, test_json_filename)
        with open(test_json_filepath, 'r') as fIn:
            test_json = json.load(fIn)
            if 'test' in test_json and 'test_kp' in test_json:
                testJsonPath = test_json_filepath
    return testJsonPath

def isModelTestedAlready(model_dir_path):
    wasModelTested = False
    model_name = os.path.basename(os.path.normpath(model_dir_path))
    # in case the model was tested after training (with the runAllTests, and the results are in a test_results folder:
    if os.path.exists(os.path.join(model_dir_path, 'test_results', f'scores_{model_name}_test.json')) and \
        os.path.exists(os.path.join(model_dir_path, 'test_results', f'scores_{model_name}_test_kp.json')):
        wasModelTested = True
    # in case the model was tested during training, and the results are in the test json under the main model folder:
    elif getSingleTestJsonOfModel(model_dir_path) != '':
        wasModelTested = True
    return wasModelTested

def getModelInfo(json_path):
    with open(json_path, 'r') as fIn:
        model_json = json.load(fIn)
    return model_json

def getModelResultsInfo(model_dir_path):
    model_name = os.path.basename(os.path.normpath(model_dir_path))
    results = {}
    # if the results were made with this script, then they are in the test_results folder:
    if os.path.exists(os.path.join(model_dir_path, 'test_results', f'scores_{model_name}_test.json')) and \
            os.path.exists(os.path.join(model_dir_path, 'test_results', f'scores_{model_name}_test_kp.json')):
        results['test'] = getModelInfo(os.path.join(model_dir_path, 'test_results', f'scores_{model_name}_test.json'))
        results['test_kp'] = getModelInfo(os.path.join(model_dir_path, 'test_results', f'scores_{model_name}_test_kp.json'))
    else:
        # otherwise the results were made during training, in which case we read them from the single test
        # json of the best checkpoint:
        fullTestResultsJsonPath = getSingleTestJsonOfModel(model_dir_path)
        fullTestResultsJson = getModelInfo(fullTestResultsJsonPath)
        # replace the key name with that of this script:
        for t in ['test', 'test_kp']:
            if 'summ_len_full' in fullTestResultsJson[t]:
                fullTestResultsJson[t]['avg_num_tokens_in_summ'] = fullTestResultsJson[t]['summ_len_full']
                del fullTestResultsJson[t]['summ_len_full']
            if 'num_KPs_full' in fullTestResultsJson[t]:
                fullTestResultsJson[t]['avg_num_kps_in_summ'] = fullTestResultsJson[t]['num_KPs_full']
                del fullTestResultsJson[t]['num_KPs_full']
        results.update(fullTestResultsJson) # has both test and test_kp items already
    results['model_config'] = getModelInfo(os.path.join(model_dir_path, 'config.json'))
    return results

def output_all_results(all_model_results, output_csv_filepath):
    test_fields = list(list(all_model_results.values())[0]['test'].keys())
    test_kp_fields = list(list(all_model_results.values())[0]['test_kp'].keys())
    config_fields = ['phrasing_method', 'input_mode', 'max_num_steps', 'max_phrase_len', 'reward_func',
                     'weight_initial', 'weight_preceding', 'stem', 'remove_stop', 'method_mode', 'importance_mode',
                     'diversity_mode', 'beta']
    csv_lines = ['model_name,' + ','.join(config_fields) + ',test,' + ','.join(test_fields) + ',test_kp,' + ','.join(test_kp_fields)]
    for model_name, model_results in all_model_results.items():
        line = f'{model_name},' + \
               ','.join([f"{model_results['model_config'][f]}" for f in config_fields]) + \
               ',,' + \
               ','.join([f"{model_results['test'][f]:.4f}" if f in model_results['test'] else "" for f in test_fields]) + \
               ',,' + \
               ','.join([f"{model_results['test_kp'][f]:.4f}" if f in model_results['test_kp'] else "" for f in test_kp_fields])
        csv_lines.append(line)
    with open(output_csv_filepath, 'w') as fOut:
        fOut.write('\n'.join(csv_lines))


def runAllTests(saved_models_path, output_filepath, recompute=False, splits_to_compute=['test']):
    all_model_results = {}
    for model_dir in os.listdir(saved_models_path):
        model_dir_path = os.path.join(saved_models_path, model_dir)
        if os.path.isdir(model_dir_path) and model_dir.startswith('kp_'):
            print(f'--- {model_dir} -------------')
            if not recompute and isModelTestedAlready(model_dir_path):
                print('\tAlready computed.')
            else:
                # run the test with the same configuration as it was trained:
                model_config = getModelInfo(os.path.join(model_dir_path, 'config.json'))
                method_mode = model_config['method_mode']
                phrasing_method = model_config['phrasing_method']
                input_mode = model_config['input_mode'] if model_config['input_mode'] != '' else '""'
                importance_mode = model_config['importance_mode']
                diversity_mode = model_config['diversity_mode']
                beta = model_config['beta']
                for split in splits_to_compute:
                    cmd = 'python test_full.py ' + \
                          f'--save_dir {model_dir_path}/test_results ' + \
                          f'--model_dir {model_dir_path} --{split} --num_steps 20 ' + \
                          f'--method_mode {method_mode} --phrasing_method {phrasing_method} ' + \
                          f'--input_mode {input_mode} --importance_mode {importance_mode} ' + \
                          f'--diversity_mode {diversity_mode} --beta {beta}'
                    print('------- ' + cmd)
                    os.system(cmd)
            if 'test' in splits_to_compute:
                all_model_results[model_dir] = getModelResultsInfo(model_dir_path)
        if len(all_model_results) > 0:
            output_all_results(all_model_results, output_filepath)

if __name__ == '__main__':
    # send in the arg '--recompute' to test a model even if it was already tested
    saved_models_path = 'saved_model'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    output_filepath = f'saved_model/results_kp_{current_time}.csv'
    splits_to_compute = ['test'] #'val', 'train']
    recompute = '--recompute' in sys.argv
    runAllTests(saved_models_path, output_filepath, recompute=recompute, splits_to_compute=splits_to_compute)