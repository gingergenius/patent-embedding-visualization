"""
Used for patent landscaping use-case.
"""

from fiz_lernmodule import tokenizer
from keras.preprocessing import sequence
import random
import numpy as np


class DataPreparation:
    """ Prepares the patent landscape data. """
    RAND_SEED = 314159
    inputs_as_text = None
    inputs_as_embedding_idxs = None
    labels_as_idxs = None
    refs_tokenizer = None
    refs_one_hot = None
    cpc_tokenizer = None
    cpc_one_hot = None
    max_seq_length = None
    padded_train_embed_idxs_x = None
    padded_test_embed_idxs_x = None
    trainEmbedIdxsX = None
    trainRefsOneHotX = None
    trainCpcOneHotX = None
    testEmbedIdxsX = None
    testRefsOneHotX = None
    testCpcOneHotX = None
    trainY = None
    testY = None

    def __init__(self, training_data, embedding_model):
        self.data = training_data
        self.embedding_model = embedding_model

        self.tokenizer = tokenizer.TextTokenizer()

    def prepare_data(self, percent_train):
        """ Takes the initially loaded data and prepares it for training.

        Abstracts are tokenized and transformed into embedding indexes.
        Labels are stored as 1 (antiseed) or 0 (seed).
        Refs and cpc-labels are transformed into one-hot matrices.
        After these transformations, training and test split are created.
        Eventually, the input sequences (abstracts) are padded based on the longest abstract.

        Args:
            percent_train (float): specifies split between train and test.
        """
        self.inputs_as_text = self.data.abstract_text
        self.inputs_as_embedding_idxs = self.text_to_embedding_idxs(self.inputs_as_text)
        
        labels_as_text = self.data.ExpansionLevel
        self.labels_as_idxs = self.labels_to_idxs(labels_as_text)

        refs_as_texts = self.data.refs
        refs_vocab_size = 50000
        self.refs_tokenizer, self.refs_one_hot = \
            self.tokenizer.tokenize_to_onehot_matrix(refs_as_texts, refs_vocab_size)

        cpc_as_texts = self.data.cpcs
        cpc_vocab_size = 500
        self.cpc_tokenizer, self.cpc_one_hot = \
            self.tokenizer.tokenize_to_onehot_matrix(cpc_as_texts, cpc_vocab_size)

        self.create_train_and_test_split(percent_train)

        self.check_dimensionality('train')
        self.check_dimensionality('test')

        length_of_input_sequences = list(map(len, self.trainEmbedIdxsX))

        median_seq_length = int(np.median(length_of_input_sequences))
        max_seq_length = np.max(length_of_input_sequences)

        print('Sequence lengths for embedding layer: median: {}, mean: {}, max: {}.'.format(
            median_seq_length, np.mean(length_of_input_sequences), max_seq_length))

        self.max_seq_length = max_seq_length

        print('Using sequence length of {} to pad LSTM sequences.'.format(max_seq_length))
        self.padded_train_embed_idxs_x = sequence.pad_sequences(
            self.trainEmbedIdxsX, maxlen=max_seq_length, padding='pre', truncating='post')
        self.padded_test_embed_idxs_x = sequence.pad_sequences(
            self.testEmbedIdxsX, maxlen=max_seq_length, padding='pre', truncating='post')

        print('Training data is prepared.')

    def create_train_and_test_split(self, percent_train):
        """ Splits the available data according to the given percentage.

        Data is shuffled and divided into train and test data.
        Numpy arrays are used to store the patent abstracts (as embedding indexes),
        References/ cpc-labels (as one hot matrices) and labels (1/0).

        Args:
            percent_train (float): specifies split between train and test data
        """
        training_data_to_shuffle = list(
            zip(
                self.inputs_as_embedding_idxs,
                self.refs_one_hot,
                self.cpc_one_hot,
                self.labels_as_idxs,
                range(len(self.inputs_as_embedding_idxs))))

        print("Randomizing data.")
        random.seed(self.RAND_SEED)
        random.shuffle(training_data_to_shuffle)

        inputs_as_embedding_idxs, refs_one_hot, cpc_one_hot, labels_as_idxs, self.idxs_after_shuffling = zip(*training_data_to_shuffle)
        self.final_train_idx = int(len(inputs_as_embedding_idxs) * percent_train)
        
        print('Creating NumPy arrays for train/test set out of randomized training data.')
        self.trainEmbedIdxsX = np.array(inputs_as_embedding_idxs[:self.final_train_idx])
        self.trainRefsOneHotX = np.array(refs_one_hot[:self.final_train_idx])
        self.trainCpcOneHotX = np.array(cpc_one_hot[:self.final_train_idx])

        self.testEmbedIdxsX = np.array(inputs_as_embedding_idxs[self.final_train_idx:])
        self.testRefsOneHotX = np.array(refs_one_hot[self.final_train_idx:])
        self.testCpcOneHotX = np.array(cpc_one_hot[self.final_train_idx:])

        self.trainY = np.array(labels_as_idxs[:self.final_train_idx])
        self.testY = np.array(labels_as_idxs[self.final_train_idx:])

    def text_to_embedding_idxs(self, raw_texts):
        """ Replaces words in patent abstracts by word embedding indexes.

        Args:
            raw_texts (pd.Series): pd.Series that contains abstracts

        Returns:
            List of patents where each patent is represented by a list of embedding indexes.
        """

        tokenized_texts = self.tokenizer.tokenize_series(raw_texts)
        word_to_index = self.embedding_model.word_to_index
        tokenized_indexed_text = []

        for idx in range(0, len(tokenized_texts)):
            single_tokenized_text = tokenized_texts[idx]
            text_word_indexes = []
            for word in single_tokenized_text:
                if word in word_to_index:
                    word_idx = word_to_index[word]
                else:
                    word_idx = word_to_index['UNK']

                text_word_indexes.append(word_idx)

            tokenized_indexed_text.append(text_word_indexes)
        return tokenized_indexed_text

    def check_dimensionality(self, train_or_test):
        """ Checks if data points and corresponding labels have the same number of elements.

        Args:
            train_or_test (str): specifies whether used for evaluation of train or test set.

        Raises:
            Exception: if number of elements does not match.
        """
        if train_or_test == 'train':
            x = self.trainEmbedIdxsX
            y = self.trainY
        else:
            x = self.testEmbedIdxsX
            y = self.testY

        if x.shape[0] != y.shape[0]:
            raise Exception('Number of elements in X ({}) and y ({}) in {} data does not match'.format(
                x.shape[0], y.shape[0], train_or_test))
        else:
            print('Number of elements in {}: {}'.format(train_or_test, x.shape[0]))

    def labels_to_idxs(self, raw_labels):
        """ Transforms a list of labels into a list of indexes.

        Args:
            raw_labels (list): elements contain either 'seed' or 'anti-seed'.

        Returns:
            List of indexes.
        """
        labels_indexed = []
        for idx in range(0, len(raw_labels)):
            label = raw_labels[idx]
            # 'tokenize' on the label is basically normalizing it
            tokenized_label = self.tokenizer.tokenize(label)[0]
            label_idx = self.label_text_to_id(tokenized_label)
            labels_indexed.append(label_idx)

        return labels_indexed

    def label_text_to_id(self, label_name):
        """ Transforms a label into an index."""
        if label_name == 'antiseed':
            return 1
        else:
            return 0

    def label_id_to_text(self, label_idx):
        """ Transforms an index into a label."""
        if label_idx == 1:
            return 'antiseed'
        else:
            return 'seed'

    def show_instance_details(self, train_instance_idx):
        """ Displays details about an instance given its index."""
        print('\nOriginal:\n{}\n\nTokenized:\n{}\n\nTextAsEmbeddingIdx:\n{}\n\nLabelAsIdx:\n{}'.format(
            self.inputs_as_text[train_instance_idx],
            self.to_text(self.inputs_as_embedding_idxs[train_instance_idx]),
            self.inputs_as_embedding_idxs[train_instance_idx],
            self.labels_as_idxs[train_instance_idx]))
            

    def to_text(self, idxs):
        """ Returns the text corresponding to a list of indexes."""
        words = []
        for word_int in idxs:
            words.append(self.embedding_model.index_to_word[word_int])
        return ' '.join(words)
