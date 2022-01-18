
def should_filter(tokens, out=False, filter_no_period=True): #, keep_hash=False):
    # tokens: list of strings (tokens)
    # out: should the decision of filtering be printed out
    contains_quotation_marks = len(tokens) >= 3 and \
                               (tokens[0] == "``" or tokens[0] == "''" or tokens[0] == '"') and \
                               ("``" in tokens[2:] or "''" in tokens[2:] or '"' in tokens[2:])
    doesnt_end_with_period = len(tokens) > 0 and tokens[-1] != "."
    # contains_says = "says" in tokens or "said" in tokens
    decision = contains_quotation_marks or (filter_no_period and doesnt_end_with_period)
    #if keep_hash and tokens[0] == '###':
    #    decision = False
    if out and decision:
        print("Skipping quote: ", ' '.join(tokens))
    return decision

def prepare_data_for_model_input(topic_docs, initial_summ_indices, queries_info, get_index_mapper=False,
                                 ignore_queries=False, max_num_steps=-1, filter_sents=True, do_not_filter_sents=None):
    """
    Prepares the topic documents for the model, and adjusts the initial summary indices to be with respect
    to the flattened sentences list returned.
    :param topic_docs: list of list of lists (documents, sentences, tokens)
    :param initial_summ_indices: list of (docIdx, sentIdx) pairs
    :param queries_info: list of pair-lists with [query, num_sentences]
    :param get_index_mapper: should the sent_idx_mapper be returned (mapping original sentIdx to the flat sentIdx)
    :param ignore_queries: should the queries be ignored, just making a list of empty queries
    :param max_num_steps: the max number of queries to put in the query list (if 0 or less, then all queries are used)
    :param do_not_filter_sents: a list/set of sentence id pairs not to filter out
    :return: topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted, queries_list, <sent_idx_mapper>
    """
    if do_not_filter_sents == None:
        do_not_filter_sents = []

    topic_docs_filtered = [] # the same as topic_docs but filtered of certain sentences
    topic_docs_filtered_flat = [] # the same as topic_docs_filtered but flattened
    sent_idx_mapper = {} # keeps (docIdx, sentIdx) -> idx_in_topic_docs_filtered_flat
    cur_idx = 0
    for docIdx, doc_sents in enumerate(topic_docs):
        doc_sents_filtered = []
        for sentIdx, sent in enumerate(doc_sents):
            if not filter_sents or (docIdx, sentIdx) in do_not_filter_sents or not should_filter(sent):
                doc_sents_filtered.append(sent)
                topic_docs_filtered_flat.append(sent)
                sent_idx_mapper[(docIdx, sentIdx)] = cur_idx
                cur_idx += 1
        if len(doc_sents_filtered) > 0:
            topic_docs_filtered.append(doc_sents_filtered)

    # get the adjusted initial summary indices to be with respect to the topic_docs_filtered_flat list:
    initial_summ_indices_adjusted = [sent_idx_mapper[(doc_idx, sent_idx)]
                                     for doc_idx, sent_idx in initial_summ_indices]

    # create a list of queries, repeating each query according to the number of sentences needed per query:
    queries_list = [] # list of tuples (queryStr, queryGroupIdx)
    for queryGroupIdx, (queryStr, numSentencesNeeded) in enumerate(queries_info):
        if not ignore_queries:
            queries_list.extend([(queryStr, queryGroupIdx)] * numSentencesNeeded)
        else:
            queries_list.extend([('', 1)] * numSentencesNeeded)
    if max_num_steps > 0:
        queries_list = queries_list[:max_num_steps]

    if get_index_mapper:
        return topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted, queries_list, sent_idx_mapper
    else:
        return topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted, queries_list