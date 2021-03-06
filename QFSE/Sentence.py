from QFSE.Utilities import nlp, bert_embedder
from QFSE.Utilities import REPRESENTATION_STYLE_W2V, REPRESENTATION_STYLE_BERT, REPRESENTATION_STYLE_SPACY
from QFSE.Utilities import STOP_WORDS, PUNCTUATION, TRANSITION_WORDS
import sklearn
from nltk.tokenize import word_tokenize
import numpy as np

class Sentence:

    def __init__(self, docId, sentIndex, text, representationStyle, doNotInitRepresentation=False): # spacyDoc=None, setBertEmbedding=False):
        self.docId = docId
        self.sentIndex = sentIndex
        self.sentId = '{}::{}'.format(docId, sentIndex)
        self.text = text.strip()
        self.textCompressed = ''.join(self.text.split()).lower()
        self.representationStyle = representationStyle

        self.tokens = word_tokenize(text)
        self.lengthInWords = len(self.tokens)
        self.lengthInChars = len(self.text)

        if doNotInitRepresentation:
            self.representation = None
            # the spacy object per sentence is time consuming, so set from Document with setRepresentation method
        else:
            self.__initRepresentation()

    def __initRepresentation(self):
        if self.representationStyle == REPRESENTATION_STYLE_SPACY:
            self.representation = nlp(self.text).vector  # a spacy doc object
        elif self.representationStyle == REPRESENTATION_STYLE_BERT:
            self.representation = bert_embedder.encode([self.text])[0]  # a numpy vector
        elif self.representationStyle == REPRESENTATION_STYLE_W2V:  # default for now is W2V
            wordVectors = [nlp.vocab.get_vector(w) for w in self.tokens if
                                           w not in STOP_WORDS and w not in PUNCTUATION and nlp.vocab.has_vector(w)]
            if len(wordVectors) > 0:
                self.representation = np.mean(wordVectors, axis=0)
            else:
                self.representation = np.random.uniform(-1, 1, (300,))
        else:
            self.representation = None

    def setRepresentation(self, representation):
        self.representation = representation

    def __len__(self):
        return self.lengthInWords

    def __repr__(self):
        return self.text

    def __eq__(self, other):
        return other != None and self.sentId == other.sentId

    def __ne__(self, other):
        return not self.__eq__(other)

    def similarity(self, otherSentence):
        return sklearn.metrics.pairwise.cosine_similarity([self.representation, otherSentence.representation])[0][1]