""" run decoding of RLMMR_K """
import argparse
import json
import os
from datetime import timedelta
from time import time
import torch

from train_full_rl import build_batchers
from decoding import RLExtractor, decode_full_dataset
from dataset_k.prepare_mds_kps import prepare_data_in_datadir

def test_all(data_dir, save_path, model_dir, test_set_type, batch_size, beta, cuda, importance_mode, diversity_mode,
             method_mode, phrasing_method, input_mode, num_steps, max_phrase_len):

    start = time()

    def run_decode(split_name, split_loader):
        full_dataset_scores = decode_full_dataset(agent, split_loader, save_path, split_name, num_steps, recompute_if_exists=False)

        message = f'finished in {timedelta(seconds=int(time() - start))}! '
        message += ', '.join([f'{metric}: {full_dataset_scores[metric]:.4f}' for metric in full_dataset_scores])
        message += f'avg # tokens noInitial: {full_dataset_scores["kps_tokens_len_mean_noInitial"]:.2f}, '
        message += f'avg # KPs withInitial: {full_dataset_scores["num_kps_mean_full"]:.2f}'
        #message += f'avg # tokens: {avg_num_tokens_in_summ:.2f}, '
        #message += f'avg # KPs: {avg_num_kps_in_summ:.2f}'
        print(message)
        # print(f'finished in {timedelta(seconds=int(time() - start))}! '
        #      f'AVG_AUC_NORM_OVERLAP: {full_dataset_scores["auc_norm_overlap"]:.4f}, '
        #      f'AVG_AUC_NORM_TFIDF: {full_dataset_scores["auc_norm_tfidf"]:.4f}, '
        #      f'MEAN_KP_SCORE_OVERLAP: {full_dataset_scores["mean_kp_score_overlap"]:.4f}, '
        #      f'MEAN_KP_SCORE_TFIDF: {full_dataset_scores["mean_kp_score_tfidf"]:.4f}, '
        #      f'avg # tokens: {avg_num_tokens_in_summ:.2f}, '
        #      f'avg # KPs: {avg_num_kps_in_summ:.2f}')

        #full_dataset_scores['avg_num_tokens_in_summ'] = avg_num_tokens_in_summ
        #full_dataset_scores['avg_num_kps_in_summ'] = avg_num_kps_in_summ

        model_name = os.path.basename(os.path.normpath(model_dir))
        fname = os.path.join(save_path, f'scores_{model_name}_{split_name}.json')
        with open(fname, 'w') as fOut:
            json.dump(full_dataset_scores, fOut, indent=4)


    # load the agent:
    agent = RLExtractor(model_dir, data_dir, beta, method_mode, importance_mode, diversity_mode,
                        cuda=(torch.cuda.is_available() and cuda))

    # prepare the batchers:
    train_batcher, val_batcher, test_batcher, test_kp_batcher = \
        build_batchers(batch_size, data_dir, input_mode, phrasing_method, num_steps, max_phrase_len)

    # run the decoding according to the test requested:
    if test_set_type == 'val':
        run_decode('val', val_batcher)
    elif test_set_type == 'test':
        run_decode('test_kp', test_kp_batcher)
        run_decode('test', test_batcher)
    elif test_set_type == 'train':
        run_decode('train', train_batcher)
    else:
        raise Exception('Provide test or val split type.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='run decoding of the full model (RL)')
    parser.add_argument('--data_dir', action='store', default='dataset_k/DUC',
                        help='directory of the database for the dynamic test type')
    parser.add_argument('--save_dir', required=True, help='path to store/eval')
    parser.add_argument('--method_mode', action='store', default='soft-attn',
                        help='mode for use of MMR: soft-attn provides MMR scors as soft attention for sentences, and no-mmr does not')
    parser.add_argument('--model_dir', help='root of the full model')
    parser.add_argument('--phrasing_method', action='store', default='posregex',
                        help='How to prepare keyphrases: nounchunks, posregex')
    parser.add_argument('--input_mode', action='store', default='',
                        help='Any manipulation on the keyphrases: <>, <singledoc>')  # , <noduplicates>, <noduplicates_singledoc>')
    parser.add_argument('--num_steps', type=int, action='store', default=20,
                        help='How many keyphrases should be extracted during train/val/test.')
    parser.add_argument('--max_phrase_len', type=int, action='store', default=999,
                        help='The max phrase token-length to use.')
    # decode options
    parser.add_argument('--importance_mode', action='store', default='tfidf',
                        help='MMR importance function (tfidf)')
    parser.add_argument('--diversity_mode', action='store', default='tfidf',
                        help='MMR diversity function (tfidf)')
    parser.add_argument('--beta', type=float, action='store', default=0.6,  # 5e-5
                        help='the weight of importance (vs diversity) in the MMR component')
    parser.add_argument('--batch_size', type=int, action='store', default=8,
                        help='batch size of faster decoding')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')
    data.add_argument('--train', action='store_true', help='use train set')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    if args.test:
        data_split = 'test'
    elif args.val:
        data_split = 'val'
    else:
        data_split = 'train'

    # Create a multi-doc keyphrase list set with the DUC 2001 KP dataset:
    prepare_data_in_datadir(args.data_dir, 'test_kp')

    test_all(args.data_dir, args.save_dir, args.model_dir, data_split,
             args.batch_size, args.beta, args.cuda, args.importance_mode, args.diversity_mode,
             method_mode=args.method_mode,
             phrasing_method=args.phrasing_method, input_mode=args.input_mode,
             num_steps=args.num_steps, max_phrase_len=args.max_phrase_len)
