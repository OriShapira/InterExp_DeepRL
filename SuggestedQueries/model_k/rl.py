import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np
from nltk.tokenize import word_tokenize

from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

from .rnn import MultiLayerLSTMCells
from .extract import LSTMPointerNet
from dataset_k.data_manage import get_dataset_texts
from metric_k import _spacy_nlp, _stemmer, STOP_TOKENS

INI = 1e-2

class ActorExtractor(nn.Module):
    """ to be used as Actor (predicts an output)"""

    def __init__(self, ptr_net, data_source, beta=0.5,
                 method_mode='soft-attn', importance_mode='tfidf', diversity_mode='tfidf'):
        """
        The class for RL-based sentence extraction.
        :param ptr_net: the base extractor on which to rely (of type LSTMPointerNet), should be pre-trained
        :param data_source: the base path of the database (see dataset.data_manage for structure info)
               or a topic's list of texts (where each text is a "document").
        :param beta: the hyper-parameter for the MMR component
        :param method_mode: what method to use for summarizing ('soft-attn')
        :param importance_mode: the method for importance scoring in MMR (only tfidf supported)
        :param diversity_mode: the method for diversity scoring in MMR (only tfidf supported)
        """
        assert isinstance(ptr_net, LSTMPointerNet)
        assert method_mode in ['soft-attn', 'no-mmr']
        super().__init__()

        # for the LSTM of the output summary
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters for the input sentences
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone()) # W3
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone()) # W4
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone()) # v2

        # hop parameters for the input sentences
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone()) # W1
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone()) # W2
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone()) # v1
        self._n_hop = ptr_net._n_hop # 1 for two-hop

        self.beta = beta  # sentence_importance_score = beta * doc_importance_score + (1 - beta) * query_similarity
        self.method_mode = method_mode
        self.importance_mode = importance_mode
        self.diversity_mode = diversity_mode

        # scorers dict (keys are 'train', 'val', 'test'):
        self.score_agents = create_score_agent_dict(data_source,
                                                    importance_mode=self.importance_mode,
                                                    diversity_mode=self.diversity_mode)

        # FF for the MMR scores - to_score means the output is a single number:
        self.mlp = MLP(in_dim=1, to_score=True)

        print(f'beta={self.beta}, method_mode={self.method_mode}')

    def forward(self, attn_mem, num_outputs, kp_list=None, datum_name='', split='',
                initial_summ=None):
        """attn_mem: Tensor of size [num_kps, input_dim] -- the encoded kps
           kp_list: the list of textual keyphrases
           split: the set on which to work (train, val, test)"""

        def filter_by_cond(score_per_kp, ignoreInitialSummary=False):
            # do not allow to reuse kps already chosen for the summary:
            for o in output_indices_obj:
                score_per_kp[0, o.item()] = -1e18
            # in case there's an initial summary, don't allow those sentences either:
            if not ignoreInitialSummary:
                for i in initial_summ:
                    score_per_kp[0, i] = -1e18
            return score_per_kp

        output_indices_obj = []  # list of selected sentences for the output summary
        output_scores = [] # list of scores for the above selected sentences (output_indices_obj)
        dists_kp_indices = []  # list of distributions of kp indices at each output (list of lists)

        # initializing the LSTM for the "current output summary":
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))

        # if there is an initial summary, prepare the "current summary LSTM" accordingly:
        if initial_summ != None:
            for kpIdx in initial_summ:
                extracted_kps_lstm_h, extracted_sents_lstm_c = self._lstm_cell(lstm_in, lstm_states)
                lstm_in = attn_mem[kpIdx].unsqueeze(0)
                lstm_states = (extracted_kps_lstm_h, extracted_sents_lstm_c)
        else:
            initial_summ = []

        # get num_outputs keyphrases:
        for _ in range(num_outputs):
            # the current current-summary LSTM state (z_t in paper):
            extracted_kps_lstm_h, extracted_kps_lstm_c = self._lstm_cell(lstm_in, lstm_states)
            last_extracted_kp_lstm_h = extracted_kps_lstm_h[:, -1, :]  # 1 x 256
            cur_output_summ_indices = initial_summ + [o.item() for o in output_indices_obj] # the indices of the sentences in the summary so far

            # for the soft-attention mode, first get the A' matrix (sentence representations), based on MMR scores:
            if self.method_mode == 'soft-attn':
                # get the basic MMR scores and make it a tensor:
                mmr_scores = [self.score_agents[split].calc_score(kp_idx, cur_output_summ_indices,
                                                                  kp_list, datum_name, beta=self.beta)
                              for kp_idx in range(len(kp_list))]
                mmr_scores = torch.Tensor(mmr_scores).to(attn_mem.device).view(-1, 1)
                # run FF on MMR scores:
                mmr_scores = self.mlp(mmr_scores).view(1, -1) # FF(m)
                # softmax on the scores - this is mu^t from the paper:
                mmr_scores = F.softmax(mmr_scores * 2, dim=-1) # mu = softmax(FF(m)*2)
                #mmr_scores = F.softmax(mmr_scores, dim=-1)  # mu = softmax(FF(m))
                # multiply with the sentence attention vectors (so that A turns into A' in paper)
                kp_encodings_mmr = attn_mem * mmr_scores.t() # A' = A*mu
                attn_feat = torch.mm(kp_encodings_mmr, self._attn_wm) # A'*W3
                hop_feat = torch.mm(kp_encodings_mmr, self._hop_wm) # # A'*W1
            elif self.method_mode == 'no-mmr':
                attn_feat = torch.mm(attn_mem, self._attn_wm)  # A*W3
                hop_feat = torch.mm(attn_mem, self._hop_wm)  # # A*W1
            else:
                raise Exception('Error: ActorExtractor method_mode not supported: ' + str(self.method_mode))

            # get the glimpse value (g_t in the paper) based on A' and z_t
            # this essentially gives importance to the sentences according to the overall text (A) and the
            # summary generated until now (z_t) - this is two-hop (self._n_hop is 1):
            for _ in range(self._n_hop):
                # g_t = softmax(v1*tanh(W1*A' + W2*z_t)) * W1*A'
                #   z_t  = last_extracted_sent_lstm_h
                #   W1*A'= hop_feat
                #   v1   = self._hop_v
                #   W2   = self._hop_wq
                last_extracted_kp_lstm_h = ActorExtractor.attention(hop_feat, last_extracted_kp_lstm_h,
                                                                    self._hop_v, self._hop_wq)

            # get the scores for the sentences (p^t in the paper) based on g_t and A':
            # p^t = v2*tanh(W3*A' + W4*g_t)
            #   g_t  = last_extracted_sent_lstm_h
            #   W3*A = attn_feat
            #   v2   = self._attn_v
            #   W4   = self._attn_wq
            score_per_kp = ActorExtractor.attention_score(attn_feat, last_extracted_kp_lstm_h,
                                                          self._attn_v, self._attn_wq)

            # make sure there is a score for each sentence:
            assert len(kp_list) == score_per_kp.shape[1]

            # do not re-use kps already selected:
            score_per_kp = filter_by_cond(score_per_kp, ignoreInitialSummary=False)

            # if training, sample the next sentence for the output summary according to the score distribution:
            if self.training:
                probs = F.softmax(score_per_kp, dim=-1) # softmax on the scores
                distribution_categorical = torch.distributions.Categorical(probs) # distribution of indices by probabilities
                dists_kp_indices.append(distribution_categorical)
                out_idx_obj = distribution_categorical.sample() # get a sentence index according to the probability distribution
            # if at inference time, just take the top scoring sentence for the output summary:
            else:
                out_idx_obj = score_per_kp.max(dim=1, keepdim=True)[1] # get the max scoring kp
            output_indices_obj.append(out_idx_obj)
            sumScores = sum([s.item() for s in score_per_kp[0] if s.item() >= 0])
            output_scores.append(score_per_kp[0][out_idx_obj.item()].item() / sumScores if sumScores > 0 else 0.)

            lstm_in = attn_mem[out_idx_obj.item()].unsqueeze(0) # push in the chosen kp to the summ-so-far LSTM
            lstm_states = (extracted_kps_lstm_h, extracted_kps_lstm_c)

        # multiply each score by 1/rank+1, where rank is it's position in the list:
        epsilon = min(output_scores) / 100
        for i, s in enumerate(output_scores):
            # if the new score is greater than the last score, then set the score as epsilon less than the last:
            output_scores[i] = s / (i + 2) if i == 0 else min(s / (i + 2), output_scores[i - 1] - epsilon)

        if dists_kp_indices:
            # return distributions only when not empty (training)
            return output_indices_obj, output_scores, dists_kp_indices
        else:
            # in any other inference time mode, just return the list of output indices objects and corresponding scores
            return output_indices_obj, output_scores


    @staticmethod
    def attention_score(a_vec, q_vec, v_params, w_params):
        """ unnormalized attention score"""
        sum_ = a_vec + torch.mm(q_vec, w_params)  # s = W*A' + W2*z_t
        score = torch.mm(torch.tanh(sum_), v_params.unsqueeze(1)).t()  # v1*tanh(W*A' + W2*z_t)
        return score


    @staticmethod
    def attention(a_vec, q_vec, v_params, w_params):
        """ attention context vector"""
        # alpha^t = softmax(v1*tanh(W*A + W2*z_t))
        score = F.softmax(ActorExtractor.attention_score(a_vec, q_vec, v_params, w_params), dim=-1)
        output = torch.mm(score, a_vec)  # g_t = alpha^t * W*A'
        return output


class CriticScorer(nn.Module):
    """ to be used as Critic (predicts a scalar baseline reward)"""

    def __init__(self, ptr_net):
        # ptr_net is a model.extract.LSTMPointerNet (the extractor doc-level sentence encoder)
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone()) # W3
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone()) # W4
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone()) # v2

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone()) # W1
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone()) # W2
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone()) # v1
        self._n_hop = ptr_net._n_hop

        # regression layer
        self._score_linear = nn.Linear(self._lstm_cell.input_size, 1)

    def forward(self, attn_mem, num_outputs, initial_summ=None):
        """atten_mem: Tensor of size [num_sents, input_dim] -- the encoded sentences"""
        #attn_feat = torch.mm(attn_mem, self._attn_wm) # W1*A
        #hop_feat = torch.mm(attn_mem, self._hop_wm) # W3*A
        scores = [] # list of scores for the chosen sentences (one score per sentence)
        # initializing the LSTM for the "current output summary":
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))

        # if there is an initial summary, prepare the "current summary LSTM" accordingly:
        if initial_summ != None:
            for kpIdx in initial_summ:
                extracted_kps_lstm_h, extracted_kps_lstm_c = self._lstm_cell(lstm_in, lstm_states)
                lstm_in = attn_mem[kpIdx].unsqueeze(0)
                lstm_states = (extracted_kps_lstm_h, extracted_kps_lstm_c)

        for _ in range(num_outputs):
            extracted_kps_lstm_h, extracted_kps_lstm_c = self._lstm_cell(lstm_in, lstm_states)
            last_extracted_kp_lstm_h = extracted_kps_lstm_h[:, -1, :]  # 1 x 256

            attn_feat = torch.mm(attn_mem, self._attn_wm)  # A*W3
            hop_feat = torch.mm(attn_mem, self._hop_wm)  # A*W1

            for _ in range(self._n_hop):
                # the two-hop attention mechanism for the glimpse operation to get g_t
                last_extracted_kp_lstm_h = CriticScorer.attention(hop_feat, hop_feat, last_extracted_kp_lstm_h,
                                                                  self._hop_v, self._hop_wq)

            # get the extraction probability estimate to get p_j^t - or actually the next e_t output sentence:
            output = CriticScorer.attention(attn_mem, attn_feat, last_extracted_kp_lstm_h, self._attn_v, self._attn_wq)

            score = self._score_linear(output) # get a FF score for the output
            scores.append(score) # keep the score of the current chosen sentence
            lstm_in = output # feed the chosen sentence sentence to the LSTM
            lstm_states = (extracted_kps_lstm_h, extracted_kps_lstm_c)
        return scores

    @staticmethod
    def attention(attention, attention_feat, query, v, w):
        """ attention context vector"""
        sum_ = attention_feat + torch.mm(query, w) # s = W*AF + W2*z_t
        score = F.softmax(torch.mm(torch.tanh(sum_), v.unsqueeze(1)).t(), dim=-1) # alpha^t = softmax(v1*tanh(W*AF + W2*z_t))
        output = torch.mm(score, attention) # g_t = alpha^t * W*A
        return output


def create_score_agent_dict(data_source, importance_mode='tfidf', diversity_mode='tfidf'):
    """
    The source can be from a list of texts (each text is a "document"), or from a database directory.
    If from directory, assuming train, val, test folders under it (see dataset.data_manage for database structure).
    :param data_source: dataset_basepath or list of document texts
    :param importance_mode:
    :param diversity_mode:
    :return: Dictionary of ScoreAgents. If from database dir, then the keys are the splits (train, val, test).
    If from list of docs, then the key is an empty string.
    """
    # if the source is from a list of texts:
    if isinstance(data_source, list):
        print('[ScoreAgent.py create_score_agent_dict()] Data from text list')
        sc = {'': ScoreAgent(data_source, importance_mode, diversity_mode)}
    # if the source is from a database directory:
    else:
        print('[ScoreAgent.py create_score_agent_dict()] Data:', data_source)
        sc = {'train': ScoreAgent(os.path.join(data_source, 'train'), importance_mode, diversity_mode),
              'val': ScoreAgent(os.path.join(data_source, 'val'), importance_mode, diversity_mode),
              'test': ScoreAgent(os.path.join(data_source, 'test'), importance_mode, diversity_mode),
              'test_kp': ScoreAgent(os.path.join(data_source, 'test_kp'), importance_mode, diversity_mode)}

    return sc


class ScoreAgent:
    # Class for computing scores (importance, diversity and query-similarity).
    # Currently supports only tfidf scores.

    def __init__(self, data_source, importance_mode='tfidf', diversity_mode='tfidf', stem=False, remove_stop=False):
        """
        The source can be from a list of texts (each text is a "document"), or from a dataset directory.
        If from directory, assuming structure as in dataset.data_manage.
        :param data_source: dataset_path or list of document texts
        :param importance_mode:
        :param diversity_mode:
        """
        if isinstance(data_source, list): # from list of texts (text is a "document")
            print(f'[ScoreAgent __init__] From given documents importance_mode={importance_mode} diversity_mode={diversity_mode}')
            docs_all_dict = {'': data_source}
        else: # from dataset directory
            print(f'[ScoreAgent __init__] Dataset={data_source} importance_mode={importance_mode} diversity_mode={diversity_mode}')
            docs_all_dict = get_dataset_texts(data_source)

        def tokenize(text):
            tokens = word_tokenize(text)
            if stem:
                tokens = [_stemmer.stem(item) for item in tokens]
            return tokens

        stop_list = STOP_TOKENS if remove_stop else None

        if diversity_mode == 'tfidf' or importance_mode == 'tfidf':
            self.tfidf_per_topic = {}
            for topicId in docs_all_dict:
                tfidf = TfidfVectorizer(max_features=50000, tokenizer=tokenize, stop_words=stop_list)
                #tfidf = TfidfVectorizer(max_features=50000)
                tfidf_transform = tfidf.fit_transform(docs_all_dict[topicId])
                self.tfidf_per_topic[topicId] = tfidf
        #if diversity_mode == 'w2v' or importance_mode == 'w2v':
        #    self.w2v_representation_cache = {}

        self.diversity_mode = diversity_mode  # similarity mode for diversity e.g. tfidf
        self.importance_mode = importance_mode  # similarity mode for importance and queries e.g. tfidf
        # cache for the importance, diversity and query scores:
        self.i_scores = defaultdict(dict) # the importance scores
        self.d_scores = defaultdict(dict) # the diversity scores

    def calc_importance(self, sent_idx, sent_list, datum_name):
        # similarity between the given sentence and the whole sentence set
        if datum_name == None or self.importance_mode not in self.i_scores[datum_name]:
            if self.importance_mode == 'tfidf':
                doc_vec = self.tfidf_per_topic[datum_name].transform(sent_list)
                doc_vec_all = self.tfidf_per_topic[datum_name].transform([' '.join(sent_list)])
                scores = cosine_similarity(doc_vec, doc_vec_all).squeeze()
            #elif self.importance_mode == 'w2v':
            #    for sent in sent_list:
            #        if sent not in self.w2v_representation_cache:
            #            self.w2v_representation_cache[sent] = _spacy_nlp(sent)
            #    full_text = ' '.join(sent_list)
            #    if full_text not in self.w2v_representation_cache:
            #        self.w2v_representation_cache[full_text] = _spacy_nlp(full_text)
            #    scores = [self.w2v_representation_cache[full_text].similarity(self.w2v_representation_cache[sent])
            #              for sent in sent_list]
            else:
                raise NotImplementedError
            self.i_scores[datum_name][self.importance_mode] = scores
        return self.i_scores[datum_name][self.importance_mode][sent_idx]

    def calc_diversity(self, sent_idx, sent_list, cur_output_summ_indices, datum_name):
        # Diversity between the requested sentence and the currently used sentences
        # this is actually the negative similarity score to the most similar sentence in the used sentences.
        # The max score is 0.
        if len(cur_output_summ_indices) == 0:
            return 0
        if datum_name == None or self.diversity_mode not in self.d_scores[datum_name]:
            if self.diversity_mode == 'tfidf':
                doc_vec = self.tfidf_per_topic[datum_name].transform(sent_list)
                scores = cosine_similarity(doc_vec, doc_vec).squeeze() # similarity between two sentences (for all pairs)
            #elif self.diversity_mode == 'w2v':
            #    for sent in sent_list:
            #        if sent not in self.w2v_representation_cache:
            #            self.w2v_representation_cache[sent] = _spacy_nlp(sent)
            #    scores = [[self.w2v_representation_cache[sent1].similarity(self.w2v_representation_cache[sent2])
            #              for sent1 in sent_list] for sent2 in sent_list]
            else:
                raise NotImplementedError
            self.d_scores[datum_name][self.diversity_mode] = scores

        # get the score between the requested sentence and the sentence most similar within the currently used
        # sentences, and return negative (the more similar, the lower the score)
        return -max([self.d_scores[datum_name][self.diversity_mode][sent_idx][i] for i in cur_output_summ_indices])


    def calc_score(self, sent_idx, cur_output_summ_indices, sent_list, datum_name, beta=0.5):
        # sentence was already selected
        if sent_idx in cur_output_summ_indices:
            return -1

        # get importance and diversity scores:
        I = self.calc_importance(sent_idx, sent_list, datum_name)
        D = self.calc_diversity(sent_idx, sent_list, cur_output_summ_indices, datum_name)

        # I importance score is positive cosine similarity and the D is a negative cosine similarity score
        # so we're lowering the score with D:
        return beta * I + (1 - beta) * D


class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=80, to_score=True):
        super(MLP, self).__init__()
        self.to_score = to_score
        # self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        if self.to_score:
            self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.to_score:
            x = self.fc2(x)
        return x


class ActorCritic(nn.Module):
    """ shared encoder between actor/critic"""

    def __init__(self, sent_encoder, art_encoder,
                 extractor, art_batcher, data_source, beta,
                 method_mode, importance_mode, diversity_mode):
        super().__init__()
        self._sentence_encoder = sent_encoder
        self._article_encoder = art_encoder
        self._extractor = ActorExtractor(extractor, data_source, beta=beta, method_mode=method_mode,
                                         importance_mode=importance_mode, diversity_mode=diversity_mode)
        self._scorer = CriticScorer(extractor)
        self._batcher = art_batcher


    def forward(self, raw_article_kps_list, num_outputs, datum_name='', split='', initial_summ=None):
        # encode doc set
        encoded_kps_out_list = []
        for raw_article_kps in raw_article_kps_list:
            article_kps = self._batcher(raw_article_kps)
            encoded_kps = self._sentence_encoder(article_kps)
            encoded_kps_out = self._article_encoder(encoded_kps.unsqueeze(0)).squeeze(0)
            encoded_kps_out_list.append(encoded_kps_out)
        all_kp_list = [' '.join(kp) for raw_article_kps in raw_article_kps_list for kp in raw_article_kps]
        all_encoded_kps_within_article = torch.cat(encoded_kps_out_list, dim=0)

        # get the output indices:
        extractor_output = self._extractor(all_encoded_kps_within_article, num_outputs,
                                           kp_list=all_kp_list,
                                           datum_name=datum_name, split=split,
                                           initial_summ=initial_summ)
        # extractor_output == (output_indices_obj, output_scores, dists_kp_indices) in train
        # extractor_output == (output_indices_obj, output_scores) in val/test

        # In training, output_indices_obj_actor is a pair of lists:
        #   - a list of the output index object (obj.item() is the index)
        #   - a list of distribution_categorical objects (indices of sentences distributed by their probabilities)
        # At inference time, it's just the first list, and possibly the topKInfo when self.method_mode == 'soft-attn-plusGetTopK'
        if self.training:
            #num_outputs = len(output_indices_obj_actor[0])
            # the critic's guess's scores for the same number of sentences:
            scores_per_sentence_critic = self._scorer(all_encoded_kps_within_article, initial_summ=initial_summ,
                                                      num_outputs=num_outputs)
            return extractor_output, scores_per_sentence_critic
        else:
            return extractor_output
