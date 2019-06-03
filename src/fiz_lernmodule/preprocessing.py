"""
Used for patent_classification use-case.
"""

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pandas as pd

import fiz_lernmodule.tokenizer

class PreProcessor(object):
    def __init__(self):
        self.ps = PorterStemmer()
        self.tokenizer = fiz_lernmodule.tokenizer.TextTokenizer()

    def preprocess_text(self, doc, remove_short_long = False, stem=False):
        
        tokens = self.tokenizer.tokenize(doc)

        # remove stopwords
        general_stopwords = ['invention', 'present', 'field', 'technical', 'related', 'relates', 'generally', 'describes', 'describe',
                             'particular', 'particularly', 'considering', 'consider', 'considers', 'considered', 'capable'
                             'comprising', 'comprise', 'comprised', 'consisting', 'consist', 'consists',
                             'background', 'relate', 'wherein', 'also', 'use', 'used', 'uses', 'using', 'method', 'therefore', 
                             'within', 'therein', 'thereof', 'may', 'become', 'becomes', 'became', 'becoming', 
                             'achieves', 'achieve', 'achieved', 'achieving', 'claimed', 'claims', 'claim', 'null', '_NUMBER_', 'NULL'
                             'dadurch', 'gekennzeichnet', 'anspruch', 'verfahren', 'wobei']

        fiz_stopwords = pd.read_csv("./data/stopwords.csv").values

        nltk_stopwords = set(stopwords.words('english'))

        tokens = [w for w in tokens if (w not in nltk_stopwords) and (w not in general_stopwords) and (w not in fiz_stopwords) ]

        if remove_short_long:
            # remove short and long words
            tokens = [word for word in tokens if len(word) > 2 and len(word) < 50]

        if stem:
            # perform stemming
            tokens = [self.ps.stem(term.strip()) for term in tokens]

        return (tokens)