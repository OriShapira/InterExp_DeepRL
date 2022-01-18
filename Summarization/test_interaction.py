################################################
# CLI application for using a trained model.
################################################

import argparse
import torch
import json
import os
from decoding import decode_single_datum
from dataset.batching import _tokenize_sentence_list
from utils import str2bool

class SingleFileDataset:
    def __init__(self, jsonFilePath):
        with open(jsonFilePath) as f:
            self.js = json.load(f)

    def getData(self):
        docs_sents = self.js['documents']
        sample_id = self.js['id'] if 'id' in self.js else ''
        refsumms_sents = self.js['reference_summaries'] if 'reference_summaries' in self.js else []
        return sample_id, docs_sents, refsumms_sents



def main(input_path, model_dir, beta, query_encode, importance_mode, diversity_mode, max_sent_len, cuda):
    # load the data:
    data = SingleFileDataset(input_path)
    sample_id, docs_sents, refsumms_sents = data.getData()
    docs_sents = [_tokenize_sentence_list(docs) for docs in docs_sents]
    initial_summ_inds = []

    while True:
        queryText = input('Enter "query ||| #sents ||| max_sent_len" or "*exit": ')
        if queryText == '*exit':
            break
        else:
            queryParts = queryText.split('|||')
            #if len(queryParts) == 1 or len(queryParts) > 2:
            if len(queryParts) not in [2, 3]:
                print('Wrong input: Please enter "query ||| #sents ||| max_sent_len" with or without max_sent_len.')
                continue
            else:
                if len(queryParts) == 2:
                    queryStr, numSents = queryParts
                    maxSentLen = max_sent_len
                else:
                    queryStr, numSents, maxSentLen = queryParts
                queryStr = queryStr.strip()
                try:
                    numSents = int(numSents)
                    maxSentLen = int(maxSentLen)
                except ValueError:
                    print('Wrong input: Please enter "query ||| num_sentences".')
                    continue

                initial_summ_sents, expansion_sents, expansion_sents_indices_orig = \
                    decode_single_datum(model_dir, docs_sents, initial_summ_inds, [[queryStr, numSents]],
                                        beta, query_encode, importance_mode, diversity_mode, maxSentLen, cuda=cuda)

                print()
                print('\n'.join([' '.join(sent) for sent in expansion_sents]))
                print()

                initial_summ_inds.extend(expansion_sents_indices_orig)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='interact with a model')
    parser.add_argument('--input_path', required=True,
                        help='filepath of input json with document set and optional reference summary')
    parser.add_argument('--model_dir', required=True, help='root path of the full model')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='1-weight for the query in the MMR score (1=noWeightForQuery, 0=fullWeightForQuery)')
    parser.add_argument('--query_encoding_in_input', type=str2bool, default=True,
                        help='should a query be encoded into the input')
    parser.add_argument('--importance_mode', action='store', default='tfidf',
                        help='MMR importance function (tfidf or w2v)')
    parser.add_argument('--diversity_mode', action='store', default='tfidf',
                        help='MMR diversity function (tfidf or w2v)')
    parser.add_argument('--max_sent_len', type=int, action='store', default=9999,
                        help='what is the max sentence token-length to use')
    parser.add_argument('--no_cuda', action='store_true', help='disable GPU training')
    args = parser.parse_args()
    cuda = torch.cuda.is_available() and not args.no_cuda
    main(args.input_path, args.model_dir, args.beta, args.query_encoding_in_input,
         args.importance_mode, args.diversity_mode, args.max_sent_len, cuda)

    # Example input_path: dataset/DUC/test/topics/2006_d0601.json
    # Keep the arguments consistent with the arguments of the model when it was trained.