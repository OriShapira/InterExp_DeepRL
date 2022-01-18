import json
import os
import spacy
from cytoolz import concat
import itertools, nltk, string

class DatasetManager():
    """
    Enables reading the dataset, which looks like this:
    In the base folder there are three folders: 'train', 'val' and 'test'
    Each such folder should have two folders: 'topics', 'samples'
    - The 'topics' folder contains a json file per topic with keys:
        - id: the name of the topic (the file name without the json extension)
        - documents: list of lists of strings (list of sentences per document)
        - reference_summaries: list of list of strings (list of sentences per reference summary)
    - The 'samples' folder contains a json file per sample with keys:
        - id: the name of the sample (the file name without the json extension), e.g. 'topicId_method'
        - topic_id: the topic ID for which this sample refers to
        - method: method name used to create the queries. If 'generic', then there are no queries.
        - initial_summary_sentence_ids: list of pair-lists for [doc_index, sent_index] within the documents in the topic
        - queries: a list of pair-lists for [query_str, num_sentences]
            - string representing the query to run
            - an integer for how many sentences should be fetched for the query
    The samples are the data on which we train/val/test, and the topics are the documents and
    references for the samples (so that they are not repeated in the sample files).
    """

    INPUT_MODE_PHRASES = 10
    INPUT_MODE_PHRASES_NODUPLICATES = 11
    INPUT_MODE_PHRASES_SINGLEDOC = 12
    INPUT_MODE_PHRASES_SINGLEDOC_NODUPLICATES = 13


    def __init__(self, split: str, base_path: str, input_mode: int, phrasing_method: str,
                 num_output_phrases=0, max_phrase_len=4) -> None:
        """
        Initialize the data manager with which to iterate through the data.
        :param split:
        :param base_path:
        :param input_mode: INPUT_MODE_SENTENCES or INPUT_MODE_PHRASES - are summaries or keyphrases being generated
        :param phrasing_method: method of preparing the candidate keyphrases
        :param num_output_phrases: relevant for keyphrase generation only - how many should be generated
        :param max_phrase_len: relevant for keyphrase generation only - the maximum phrase length
        """
        assert split in ['train', 'val', 'test', 'test_kp']
        assert input_mode in [
            DatasetManager.INPUT_MODE_PHRASES, DatasetManager.INPUT_MODE_PHRASES_SINGLEDOC]
        # TODO: if need NODUPLICATES, need to implement the functionality in getPhrasesInDocset function
            #DatasetManager.INPUT_MODE_PHRASES_NODUPLICATES,
            #DatasetManager.INPUT_MODE_PHRASES_SINGLEDOC_NODUPLICATES]
        assert phrasing_method in ['nounchunks', 'posregex'] #'unsupervised'

        # We have a folder with the topics and a folder with the samples
        self._data_path_base = os.path.join(base_path, split)
        self._data_topics_path = os.path.join(self._data_path_base, 'topics')
        self._data_samples_path = os.path.join(self._data_path_base, 'samples')
        # keep the alphanumerically ordered list of samples for when the DataLoader calls __getitem__:
        self._data_samples_names = sorted(
            [name for name in os.listdir(self._data_samples_path) if name.endswith('.json')])
        self._data_count = len(self._data_samples_names)
        # read in the sample data:
        self._js_data_samples = {}
        for sampleFilename in self._data_samples_names:
            with open(os.path.join(self._data_samples_path, sampleFilename)) as fIn:
                self._js_data_samples[sampleFilename] = json.load(fIn)
        # read in the topic data:
        self._js_data_topics = {}
        for topicFilename in os.listdir(self._data_topics_path):
            if topicFilename.endswith('.json'):
                topicId = topicFilename[:-5]  # remove the '.json' from the end
                with open(os.path.join(self._data_topics_path, topicFilename)) as fIn:
                    self._js_data_topics[topicId] = json.load(fIn)

        # set input mode:
        self._input_mode = input_mode
        self._phrasing_method = phrasing_method
        # set the num phrases needed, and the SpaCy model:
        self._num_phrases_needed = num_output_phrases
        self._max_phrase_len = max_phrase_len
        self._nlp = spacy.load('en_core_web_md') if phrasing_method == 'nounchunks' else None
        # keep a cache for the phrases, so they are created once per docset:
        self._phrases_cache = {} # topicId -> docset phrases (list of lists)
        self._phrases_mapping_cache = {} # topicId -> dict of (docId, sentIdx) -> [phraseIdxs]


    def __len__(self) -> int:
        return self._data_count

    def __getitem__(self, item_idx: int):
        js_data_sample = self._js_data_samples[self._data_samples_names[item_idx]]
        sample_id = js_data_sample['id']
        topic_id = js_data_sample['topic_id']
        initial_summary_sent_indices = js_data_sample['initial_summary_sentence_ids']
        #initial_text_phrases = getPhrasesInDocset([initial_text_sentences], self._nlp,
        #                                          max_phrase_len=self._max_phrase_len,
        #                                          phrasing_method=self._phrasing_method)[0]

        js_data_topic = self._js_data_topics[topic_id]
        refsumms_sents = js_data_topic['reference_summaries']
        reference_kps = js_data_topic['reference_kps'] if 'reference_kps' in js_data_topic else []
        # the input is the document phrases (list of list strings - list of phrases per document):
        make_single_document = self._input_mode in [DatasetManager.INPUT_MODE_PHRASES_SINGLEDOC,
                                                    DatasetManager.INPUT_MODE_PHRASES_SINGLEDOC_NODUPLICATES]
        remove_duplicates = self._input_mode in [DatasetManager.INPUT_MODE_PHRASES_NODUPLICATES,
                                                 DatasetManager.INPUT_MODE_PHRASES_SINGLEDOC_NODUPLICATES]
        if topic_id not in self._phrases_cache:
            topic_documents_phrases, topic_sentid_to_phrases_mappings = \
                getPhrasesInDocset(js_data_topic['documents'], self._nlp,
                                   max_phrase_len=self._max_phrase_len, phrasing_method=self._phrasing_method,
                                   make_single_document=make_single_document) #, remove_duplicates=remove_duplicates)
            self._phrases_cache[topic_id] = topic_documents_phrases
            self._phrases_mapping_cache[topic_id] = topic_sentid_to_phrases_mappings

        docs_phrases = self._phrases_cache[topic_id]
        initial_summary_phrase_indices = []
        for sentId in initial_summary_sent_indices:
            initial_summary_phrase_indices.extend(self._phrases_mapping_cache[topic_id][tuple(sentId)])

        return sample_id, topic_id, docs_phrases, refsumms_sents, initial_summary_phrase_indices, reference_kps


def get_dataset_texts(dataset_path):
    # Gets the texts from the datasource specified
    ##   - if full dataset, one text for each topic
    #   - if full dataset, a dict of of lists of texts, topicId -> list of docs
    #   - if single topic json file, a list with one text per document in the topic
    if os.path.isdir(dataset_path) and 'topics' in os.listdir(dataset_path):
        return _get_all_topics_text(dataset_path)
    elif dataset_path.endswith('.json'):
        return _get_docs_text_in_topic_file(dataset_path)
    else:
        raise('Error: get_dataset_texts dataset path cannot be processed.')

def _get_all_topics_text(base_path):
    ## Gets a list of texts, one for each topic in the data folder specified.
    # Gets a dict of lists of texts, topicId -> list of docs - in the data folder specified.
    allTexts = {}
    topicsDirPath = os.path.join(base_path, 'topics')
    for topicFilename in os.listdir(topicsDirPath):
        if topicFilename.endswith('.json'):
            topicFilepath = os.path.join(topicsDirPath, topicFilename)
            topicDocsTexts, topicId = _get_docs_text_in_topic_file(topicFilepath, get_topic_id=True) # text per doc in topic
            allTexts[topicId] = topicDocsTexts
            #allTexts.append(' '.join(topicDocsTexts)) # a topic should have all docs in one text
    return allTexts

def _get_docs_text_in_topic_file(topic_json_path, get_topic_id=False):
    # Gets a list of texts, one for each document in the given topic json.
    docsTexts = []
    topicId = ''
    if topic_json_path.endswith('.json'):
        with open(topic_json_path) as fIn:
            topicData = json.load(fIn)
        docsTexts = [' '.join(docSents) for docSents in topicData['documents']]
        topicId = topicData['id']
    if get_topic_id:
        return docsTexts, topicId
    else:
        return docsTexts


def list_duplicate_remover(l): # remove duplicates and keep order in a list
    seen = set()
    seen_add = seen.add
    return [x for x in l if not (x in seen or seen_add(x))]

def getPhrasesInDocset(documents, nlp, max_phrase_len=4, phrasing_method='nounchunks', make_single_document=False):
                       #remove_duplicates=False):
    new_documents = [] # each phrase in a document will act as a "sentence" in the document
    all_sent_to_phrase_mappings = {} # (docIdx, sentIdx) -> [list of phraseIdxs]
    for docIdx, docSents in enumerate(documents):
        #docText = ' '.join(docSents)
        if phrasing_method == 'posregex':
            #document_phrases = extract_phrases_posregex(docText, max_phrase_len)
            document_phrases, sent_to_phrases_mapping = extract_phrases_posregex(docIdx, docSents, max_phrase_len)
        elif phrasing_method == 'nounchunks':
            document_phrases, sent_to_phrases_mapping = extract_phrases_nounchunks(docIdx, docSents, nlp, max_phrase_len)
        else:
            raise Exception(f'Phrasing method not supported: {phrasing_method}')

        #if len(document_phrases) > 0:
        new_documents.append(document_phrases)
        all_sent_to_phrase_mappings.update(sent_to_phrases_mapping)

        # if needed, concatenate all documents to a single long document (and make the mapping with one document):
        # TODO: not sure this next part is working due to indices
        if make_single_document:
            new_documents = [list(concat(new_documents))]
            single_mapping = {}
            curSentIdx = 0
            curPhraseIdx = 0
            for (docIdx, sentIdx), phraseIdxList in all_sent_to_phrase_mappings.items():
                single_mapping[(0, curSentIdx)] = list(range(curPhraseIdx, curPhraseIdx + len(phraseIdxList)))
                curSentIdx += 1
                curPhraseIdx += len(phraseIdxList)
            new_documents_sent_to_phrase_mappings = [single_mapping]

        ## if needed, remove duplicates from each document (separately):
        #if remove_duplicates:
        #    new_documents = [list_duplicate_remover(doc) for docIdx, doc in enumerate(new_documents)]
        #    # TODO: need to update mappings per sentence with the removed duplicates

    return new_documents, all_sent_to_phrase_mappings


punct = set(string.punctuation)
stop_words = set(nltk.corpus.stopwords.words('english'))
def extract_phrases_posregex(listId, sentList, max_phrase_len, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    """
    Gets chunks of phrases, created with POS groupings.
    :param listId: An ID for the given sentence list, used as an ID for the mapping
    :param sentList: List of list of tokens (list of sentences)
    :param max_phrase_len: maximum candidate phrase length allowed
    :param grammar: a regex for finding POS chunks
    :return: one list of phrases for all sentences together, and a mapping of (listId, sentIdx) -> [list of phraseIdx]
    """
    # idea from: https://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/
    # tokenize, POS-tag, and chunk using regular expressions
    # get the POS tags of all tokens for each sentence:
    pos_tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in sentList)
    # prepare a chunker according to the grammar regex
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    # mark the chunks in each sentence:
    chunks_per_sent = {sentIdx: nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                       for sentIdx, tagged_sent in enumerate(pos_tagged_sents)}
    # leave only the relevant chunks in each sentence:
    candidates_per_sent = {sentIdx: [[word for word, pos, chunk in group] for key, group in
                                     itertools.groupby(sent_chunks, lambda chunk: chunk[2] != 'O') if key]
                           for sentIdx, sent_chunks in chunks_per_sent.items()}
    # convert each chunk's list of tokens to full phrases (leaving only the ones within the length constraint):
    candidate_phrases_per_sent = {
        sentIdx: [' '.join(phraseWords) for phraseWords in sent_candidates if len(phraseWords) <= max_phrase_len]
        for sentIdx, sent_candidates in candidates_per_sent.items()}
    # remove phrases that are only stop words or punctuation:
    candidates_phrases_per_sent_final = {sentIdx: [cand for cand in sent_candidatePhrases
                                                  if cand not in stop_words and not all(char in punct for char in cand)]
                                        for sentIdx, sent_candidatePhrases in candidate_phrases_per_sent.items()}
    # map each sentence to the indices of its candidate keyphrases:
    sent_to_phrases_mapping = {}
    lastPhraseIdx = 0
    document_phrases = []
    for sentIdx, sentPhrases in candidates_phrases_per_sent_final.items():
        numPhrasesInSent = len(sentPhrases)
        sent_to_phrases_mapping[(listId, sentIdx)] = [(listId, phraseId) for phraseId in range(lastPhraseIdx, lastPhraseIdx + numPhrasesInSent)]
        #sent_to_phrases_mapping[(listId, sentIdx)] = [(listId, phraseIdx) for phraseIdx in
        #                                               range(lastPhraseIdx, lastPhraseIdx + numPhrasesInSent)]
        lastPhraseIdx += numPhrasesInSent
        document_phrases.extend(sentPhrases)

    return document_phrases, sent_to_phrases_mapping

    ## tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    #all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
    #                                                for tagged_sent in tagged_sents))
    ## join constituent chunk words into a single chunked phrase
    #candidates = [[word for word, pos, chunk in group]#.lower()
    #              for key, group in itertools.groupby(all_chunks, lambda chunk: chunk[2] != 'O') if key]
    #candidates = [' '.join(phraseWords) for phraseWords in candidates if len(phraseWords) <= max_phrase_len]
    ## remove phrases that are only stop words or punctuation:
    #document_phrases = [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
    #return document_phrases

def extract_phrases_nounchunks(listId, sentList, nlp, max_phrase_len):
    """
    Gets chunks of phrases, created with SpaCy noun chunks and named entites.
    :param listId: An ID for the given sentence list, used as an ID for the mapping
    :param sentList: List of list of tokens (list of sentences)
    :param nlp: the SpaCy model object to use for getting the chunks
    :param max_phrase_len: maximum candidate phrase length allowed
    :return: one list of phrases for all sentences together, and a mapping of (listId, sentIdx) -> [list of phraseIdx]
    """
    prefix_pos_filter = ['PUNCT', 'DET']  # punctuation, determiners
    suffix_pos_filter = ['PUNCT', 'PART', 'AUX']  # punctuation, 's, 've/'re/'d
    def spacy_span_clean(spacySpan):
        while len(spacySpan) > 0 and spacySpan[0].pos_ in prefix_pos_filter:
            spacySpan = spacySpan[1:]
        while len(spacySpan) > 0 and spacySpan[-1].pos_ in suffix_pos_filter:
            spacySpan = spacySpan[:-1]
        return spacySpan

    def filter_span(span):
        isEmpty = len(span) == 0
        isOneWordPronoun = len(span) == 1 and span[0].pos_ == 'PRON' # the phrase is one word and is a pronoun
        isLong = len(span) > max_phrase_len # do not surpass the max phrase length
        return isEmpty or isOneWordPronoun or isLong
        #return all([t.pos_ == 'PRON' for t in span]) # all tokens are pronouns

    document_phrases = []
    sent_to_phrases_mapping = {}
    lastPhraseIdx = 0
    for sentIdx, sentTxt in enumerate(sentList):
        sent = nlp(sentTxt)
        # add the named entities and noun chunks of the sentence (cleaning the span if necessary):
        sentence_phrases_spacy = [spacy_span_clean(ent) for ent in sent.ents]
        sentence_phrases_spacy.extend([spacy_span_clean(chunk) for chunk in sent.noun_chunks])
        # convert from spacy spans to strings (ignoring filtered ones):
        sentence_phrases_text = [p.text for p in sentence_phrases_spacy if not filter_span(p)]
        # remove duplicates within the sentence:
        sentence_phrases_text = list_duplicate_remover(sentence_phrases_text)
        # map sentIdx to list of phraseIdx:
        numPhrasesInSent = len(sentence_phrases_text)
        #sent_to_phrases_mapping[(listId, sentIdx)] = list(range(lastPhraseIdx, lastPhraseIdx + numPhrasesInSent))
        sent_to_phrases_mapping[(listId, sentIdx)] = [(listId, phraseId) for phraseId in range(lastPhraseIdx, lastPhraseIdx + numPhrasesInSent)]
        #sent_to_phrases_mapping[(listId, sentIdx)] = [(listId, phraseIdx) for phraseIdx in
        #                                               range(lastPhraseIdx, lastPhraseIdx + numPhrasesInSent)]
        lastPhraseIdx += numPhrasesInSent
        # append sentence phrases to document level phrases:
        document_phrases.extend(sentence_phrases_text)

    return document_phrases, sent_to_phrases_mapping

    #docObj = nlp(text)
    #document_phrases = []
    #for sent in docObj.sents:
    #    # add the named entities and noun chunks of the sentence (cleaning the span if necessary):
    #    sentence_phrases_spacy = [spacy_span_clean(ent) for ent in sent.ents]
    #    sentence_phrases_spacy.extend([spacy_span_clean(chunk) for chunk in sent.noun_chunks])
    #    # convert from spacy spans to strings (ignoring filtered ones):
    #    sentence_phrases_text = [p.text for p in sentence_phrases_spacy if not filter_span(p)]
    #    # remove duplicates within the sentence:
    #    sentence_phrases_text = list_duplicate_remover(sentence_phrases_text)
    #    # append sentence phrases to document level phrases:
    #    document_phrases.extend(sentence_phrases_text)
    #return document_phrases

#def get_reference_summaries_by_topic(dataset_dirpath):
#    refsPerTopics = {}
#    topicsDirPath = os.path.join(dataset_dirpath, 'topics')
#    for topicFilename in os.listdir(topicsDirPath):
#        if topicFilename.endswith('.json'):
#           topicFilepath = os.path.join(topicsDirPath, topicFilename)
#           with open(topicFilepath) as fIn:
#               topicData = json.load(fIn)
#           # get the word-tokenized text of each reference summary
#           refSummsTexts = [' '.join(word_tokenize(' '.join(refSents)))
#                            for refSents in topicData['reference_summaries']]
#           topicId = topicData['id']
#           refsPerTopics[topicId] = refSummsTexts
#   return refsPerTopics