"""
Used for patent_classification use-case.
"""

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

class PreProcessor(object):
    def __init__(self):
        self.ps = PorterStemmer()

    def preprocess_text(self, doc):
        doc = doc.translate({ord(c): ' ' for c in "!@#$%^&*()'/[]{};:,./<>?\|_`~°=\"+"})
        tokens = doc.lower().strip().split()

        # remove stopwords
        general_stopwords = ['invention', 'present', 'field', 'technical', 'related', 'relates', 'generally',
                             'particular', 'particularly',
                             'background', 'relate']
        nltk_stopwords = set(stopwords.words('english'))
        tokens = [w.strip('-') for w in tokens if (w not in nltk_stopwords) and (w not in general_stopwords)]

        # remove short and long words
        tokens = [word for word in tokens if len(word) > 2 and len(word) < 20]

        # perform stemming
        preprocessed_tokens = [self.ps.stem(term.strip()) for term in tokens]

        return (' '.join(preprocessed_tokens))