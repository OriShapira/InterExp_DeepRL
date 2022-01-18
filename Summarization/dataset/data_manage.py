import json
import os
import spacy
from cytoolz import concat

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

    def __init__(self, split: str, base_path: str) -> None:
        """
        Initialize the data manager with which to iterate through the data.
        :param split:
        :param base_path:
        """
        assert split in ['train', 'val', 'test']
        # We have a folder with the topics and a folder with the samples
        self._data_path_base = os.path.join(base_path, split)
        self._data_topics_path = os.path.join(self._data_path_base, 'topics')
        self._data_samples_path = os.path.join(self._data_path_base, 'samples')
        # keep the alphanumerically ordered list of samples for when the DataLoader calls __getitem__:
        self._data_samples_names = sorted([name for name in os.listdir(self._data_samples_path) if name.endswith('.json')])
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
                topicId = topicFilename[:-5] # remove the '.json' from the end
                with open(os.path.join(self._data_topics_path, topicFilename)) as fIn:
                    self._js_data_topics[topicId] = json.load(fIn)

    def __len__(self) -> int:
        return self._data_count

    def __getitem__(self, item_idx: int):
        js_data_sample = self._js_data_samples[self._data_samples_names[item_idx]]
        sample_id = js_data_sample['id']
        topic_id = js_data_sample['topic_id']
        initial_summary_indices = js_data_sample['initial_summary_sentence_ids']
        queriesInfo = js_data_sample['queries']

        js_data_topic = self._js_data_topics[topic_id]
        refsumms_sents = js_data_topic['reference_summaries']
        docs_sents = js_data_topic['documents']

        return sample_id, topic_id, docs_sents, refsumms_sents, initial_summary_indices, queriesInfo


def get_dataset_texts(dataset_path):
    # Gets the texts from the datasource specified
    #   - if full dataset, a dict of of lists of texts, topicId -> list of docs
    #   - if single topic json file, a list with one text per document in the topic
    if os.path.isdir(dataset_path) and 'topics' in os.listdir(dataset_path):
        return _get_all_topics_text(dataset_path)
    elif dataset_path.endswith('.json'):
        return _get_docs_text_in_topic_file(dataset_path)
    else:
        raise('Error: get_dataset_texts dataset path cannot be processed.')

def _get_all_topics_text(base_path):
    # Gets a dict of lists of texts, topicId -> list of docs - in the data folder specified.
    allTexts = {}
    topicsDirPath = os.path.join(base_path, 'topics')
    for topicFilename in os.listdir(topicsDirPath):
        if topicFilename.endswith('.json'):
            topicFilepath = os.path.join(topicsDirPath, topicFilename)
            topicDocsTexts, topicId = _get_docs_text_in_topic_file(topicFilepath, get_topic_id=True) # text per doc in topic
            allTexts[topicId] = topicDocsTexts
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