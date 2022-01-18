

def should_filter(tokens, out=False, filter_no_period=True): #, keep_hash=False):
    # tokens: list of strings (tokens)
    # out: should the decision of filtering be printed out
    '''
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
    '''
    return False


def prepare_data_for_model_input(topic_docs, initial_summ_indices, get_index_mapper=False,
                                 filter_sents=True, do_not_filter_sents=None):
    """
    Prepares the topic documents for the model, and adjusts the initial summary indices to be with respect
    to the flattened sentences list returned.
    :param topic_docs: list of list of lists (documents, keyphrases, tokens)
    :param initial_summ_indices: list of (docIdx, kpIdx) pairs
    :param get_index_mapper: should the sent_idx_mapper be returned (mapping original sentIdx to the flat sentIdx)
    :param do_not_filter_sents: a list/set of sentence id pairs not to filter out
    :return: topic_docs_filtered, topic_docs_filtered_flat, <sent_idx_mapper>
    """
    if do_not_filter_sents == None:
        do_not_filter_sents = []

    topic_docs_filtered = [] # the same as topic_docs but filtered of certain sentences
    topic_docs_filtered_flat = [] # the same as topic_docs_filtered but flattened
    sent_idx_mapper = {} # keeps (docIdx, sentIdx) -> idx_in_topic_docs_filtered_flat
    cur_idx = 0
    for docIdx, doc_kps in enumerate(topic_docs):
        doc_kps_filtered = []
        for kpIdx, kp in enumerate(doc_kps):
            if not filter_sents or (docIdx, kpIdx) in do_not_filter_sents or not should_filter(kp):
                doc_kps_filtered.append(kp)
                topic_docs_filtered_flat.append(kp)
                sent_idx_mapper[(docIdx, kpIdx)] = cur_idx
                cur_idx += 1
        if len(doc_kps_filtered) > 0:
            topic_docs_filtered.append(doc_kps_filtered)

    # get the adjusted initial summary indices to be with respect to the topic_docs_filtered_flat list:
    initial_summ_indices_adjusted = [sent_idx_mapper[(doc_idx, kp_idx)]
                                     for (doc_idx, kp_idx) in initial_summ_indices]

    if get_index_mapper:
        return topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted, sent_idx_mapper
    else:
        return topic_docs_filtered, topic_docs_filtered_flat, initial_summ_indices_adjusted