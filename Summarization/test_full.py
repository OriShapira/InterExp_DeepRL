######################################################
# Test the model on the full test or validation set.
######################################################
import argparse
import json
import os
from datetime import timedelta
from time import time
import torch

from train_full_rl import build_batchers
from decoding import RLExtractor, decode_full_dataset


def test_all(data_dir, save_path, model_dir, split, batch_size, beta, cuda,
             query_encode, importance_mode, diversity_mode,
             testing_summary_len=250, max_num_steps=10, reset_rouge_evaluator=False,
             max_sent_len=999):

    start = time()

    agent = RLExtractor(model_dir, data_dir, beta, query_encode,
                        importance_mode, diversity_mode,
                        cuda=(torch.cuda.is_available() and cuda))
    _, val_batcher, test_batcher = build_batchers(batch_size, data_dir)
    if split == 'val':
        loader = val_batcher
    elif split == 'test':
        loader = test_batcher
    else:
        raise Exception('Provide test or val split.')

    full_dataset_scores, avg_num_sents_in_summ, avg_num_tokens_in_summ = \
        decode_full_dataset(agent, loader, save_path, split, max_num_steps=max_num_steps,
                            testing_summary_len=testing_summary_len,
                            reset_rouge_evaluator=reset_rouge_evaluator, max_sent_len=max_sent_len)

    print(f'finished in {timedelta(seconds=int(time() - start))}! '
          f'AVG_AUC_NORM: {full_dataset_scores["auc_norm"]:.4f}, '
          f'R1_F1: {full_dataset_scores["R1"]["f1"]:.4f}, '
          f'R1_Recall: {full_dataset_scores["R1"]["recall"]:.4f}, '
          f'avg # tokens: {avg_num_tokens_in_summ:.2f}, '
          f'avg # sents: {avg_num_sents_in_summ:.2f}')

    full_dataset_scores['summ_len_full'] = avg_num_tokens_in_summ
    full_dataset_scores['num_sents_full'] = avg_num_sents_in_summ

    fname = os.path.join(save_path, f'scores.json')
    with open(fname, 'w') as fOut:
        json.dump(full_dataset_scores, fOut, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='run decoding of the full model (RL)')
    parser.add_argument('--data_dir', action='store', default='dataset/DUC',
                        help='directory of the database')
    parser.add_argument('--save_dir', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')
    parser.add_argument('--testing_summary_len', type=int, action='store', default=250,
                        help='the token-length of the summary to output during validation and testing')
    parser.add_argument('--max_num_steps', type=int, action='store', default=10,
                        help='how many sentences should be extracted during train/val/test out of the specified samples')
    # decode options
    parser.add_argument('--beta', type=float, default=0.5,
                        help='1-weight for the query in the MMR score (1=noWeightForQuery, 0=fullWeightForQuery)')
    parser.add_argument('--importance_mode', action='store', default='tfidf',
                        help='MMR importance function (tfidf or w2v)')
    parser.add_argument('--diversity_mode', action='store', default='tfidf',
                        help='MMR diversity function (tfidf or w2v)')
    parser.add_argument('--batch_size', type=int, action='store', default=8,
                        help='batch size of faster decoding')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'

    test_all(args.data_dir, args.save_dir, args.model_dir, data_split,
             args.batch_size, args.beta, args.cuda, args.importance_mode, args.diversity_mode,
             testing_summary_len=args.testing_summary_len, max_num_steps=args.max_num_steps)
