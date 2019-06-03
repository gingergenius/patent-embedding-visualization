"""
Used for patent landscaping use-case.
"""
from keras.preprocessing import text
import re
import string


class TextTokenizer:
    """ Re-usable Tokenizer based on keras-tokenizer. """

    punct_regex = re.compile('([%s])' % (string.punctuation + '‘’'))
    spaces_regex = re.compile(r'\s{2,}')
    number_regex = re.compile(r'\d+')
    # TODO keras_tokenizer = None

    def __init__(self):
        pass

    def tokenize_to_onehot_matrix(self, series_text, vocab_size, keras_tokenizer=None):
        """ Builds a one-hot matrix of a pd.Series.

        Refs and cpcs are tokenized so that each reference represents a single token.
        Builds a one-hot matrix where each column equals a unique ref-ID.
        Rows are the quoting patents. A cell-value of 1 means that patent C (column)
        was quoted by patent R (row).

        Args:
            series_text (pd.Series): Series of texts that contain references or cpc-label.
            vocab_size (int): number of elements in the vocabulary.
            keras_tokenizer:

        Returns:
            keras_tokenizer, one-hot matrix.
        """

        if keras_tokenizer is None:
            print('No Keras tokenizer supplied so using vocab size ({}) and series to build new one.'.format(vocab_size))

            keras_tokenizer = text.Tokenizer(
                num_words=vocab_size,
                split=",",
                # filter should be same as default, minus the '-'
                filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n',
                lower=False)
            keras_tokenizer.fit_on_texts(series_text)
            keras_tokenizer.index_word = {idx: word for word, idx in keras_tokenizer.word_index.items()}

        text_one_hot = keras_tokenizer.texts_to_matrix(series_text)

        return keras_tokenizer, text_one_hot

    def tokenize_series(self, series_text,
                        lowercase=True,
                        remove_punct=True,
                        normalize_numbers=True):
        # TODO add description
        """ Tokenizes texts within a pd.Series.

        Args:
            series_text (pd.Series): Series containing texts.
            lowercase (bool): True to lowercase input.
            remove_punct (bool): True to remove punctuation.
            normalize_numbers (bool): True to replace numbers by placeholder.

        Returns:
            pd.Series that contains lists of tokens.
        """
        return series_text.apply(lambda x: self.tokenize(x, lowercase,
                                                         remove_punct, normalize_numbers))

    def tokenize(self, text,
                 lowercase=True,
                 remove_punct=True,
                 normalize_numbers=True):
        # TODO: add description
        """ Tokenizes a single text.

        Args:
            text (str): string that will be tokenized.
            lowercase (bool): True to lowercase input.
            remove_punct (bool): True to remove punctuation.
            normalize_numbers (bool): True to replace numbers by placeholder.

        Returns:
            List that contains the tokens of a text.

        Raises:
            Exception: plain_text is not of type string.
        """
        plain_text = text

        if not isinstance(plain_text, str):
            raise Exception(plain_text, type(plain_text))

        preprocessed = plain_text.replace('\'', '')
        if lowercase:
            preprocessed = preprocessed.lower()

        # Replace punctuation with spaces and remove double spaces
        if remove_punct:
            preprocessed = self.punct_regex.sub(' ', preprocessed)
        else:
            preprocessed = self.punct_regex.sub(r' \1 ', preprocessed)

        preprocessed = self.spaces_regex.sub(' ', preprocessed)
        if normalize_numbers:
            preprocessed = self.number_regex.sub('_NUMBER_', preprocessed)

        return preprocessed.split()
