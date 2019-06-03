"""
Used for patent landscaping use-case.
"""

import os
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns; sns.set()


class Word2Vec(object):
    """ Word2Vec embedding. """

    def __init__(self, train_graph, index_to_word, word_to_index,
                 embedding_weights, normed_embedding_weights):
        """
        Args:
            train_graph (tf.Graph): Graph that contains embeddings.
            index_to_word (dict): maps from embedding idxs to words.
            word_to_index (dict): maps from words to embedding idxs.
            embedding_weights (np.array): embedding weights.
            normed_embedding_weights (np.array): normalized embedding weights.
        """
        self.train_graph = train_graph
        self.index_to_word = index_to_word
        self.word_to_index = word_to_index
        self.embedding_weights = embedding_weights
        self.normed_embedding_weights = normed_embedding_weights

    def visualize_embeddings(self, word, num_words=100):
        """ Creates a matplotlib plot based on similar words to `word`.

        Function identifies `num_words` most similar words of `word` in embedding space.
        Performs a dimensionality reduction with TSNE and displays the words in a plot.
        Since TNSE uses SGD visualisations might differ even with identical inputs.

        Args:
            word (str): reference word.
            num_words (int): Specifies number of words that is considered.

        Returns:
            Nothing, but creates a seaborn plot.

        """
        similar_words = self.find_similar(word, num_words)
        tsne = TSNE()
        
        word1_index = self.word_to_index[word]
        
        idxs = []
        words = []
        similarities = []
        # appends the indexes of all similar words
        for index in range(0, num_words):
            idxs.append(similar_words[index]['index'])
            words.append(similar_words[index]['word'])
            similarities.append(similar_words[index]['distance'])
        
        # appends index of `word` itself
        idxs.append(word1_index)   
        words.append(word)
        similarities.append(1)
        
        embed_tsne = tsne.fit_transform(self.normed_embedding_weights[idxs, :])

        fig, ax = plt.subplots(figsize=(14, 14))

        
        data = {
            "x": embed_tsne[:, 0],
            "y": embed_tsne[:, 1],
            "word": words,
            "sim": similarities
        }
        plot_data = pd.DataFrame(data)
        
        ax = sns.scatterplot(x="x", y="y", data = plot_data)
        
        color = "black"
        
        for idx in range(plot_data.shape[0]):
            if idx == plot_data.shape[0]-1:
                color = "red"
            ax.text(plot_data.x[idx]+1, plot_data.y[idx], plot_data.word[idx], horizontalalignment='left', size="large", color=color)
        
    def find_similar(self, word, top_k=10):
        """ Finds the `top_k` most similar words to a reference (cosine distance).

        Note: method is really slow!

        Args:
            word (str): reference word.
            top_k (int): Specifies how many similar words will be retrieved.

        Returns:
            Ordered list of dictionaries. Each dictionary corresponds to a single word.
        """
        distances = {}
        
        if word in self.word_to_index:
            word1_index = self.word_to_index[word]
            word1_embed = self.embedding_weights[word1_index]
        
            #print('Vocabulary size: {}'.format(len(self.embedding_weights)))
            for index in range(0, len(self.embedding_weights)):
                if index != word1_index:
                    word2_embed = self.embedding_weights[index]
                    word_dist = distance.cosine(word1_embed, word2_embed)
                    distances[index] = word_dist
                    
            top_k_similar = sorted(distances.items(), key=lambda x: x[1])[:top_k]
            
            similar_words = []
            for i in range(0, len(top_k_similar)):
                similar_word_index = top_k_similar[i][0]
                similar_word_dist = top_k_similar[i][1]
                similar_word = self.index_to_word[similar_word_index]
                similar_words.append(
                    {'word': similar_word,
                    'index': similar_word_index,
                    'distance': similar_word_dist})
            return similar_words

        else:
            print("Couldn't find {}".format(word))
            return []       
    
    def get_embedding(self, word, normed=True, verbose=False):
        """ Returns the normalized embedding of a given word.

        Args:
            word (str): reference word.

        Returns:
            Embedding vector within a numpy array.
        """
        if word in self.word_to_index:
            word_idx = self.word_to_index[word]
        else:
            if (verbose):
                print("Couldn't find {}. Using UNK instead. If this sounds wrong, consider normalizing text.".format(word))
            word_idx = self.word_to_index['UNK']

        if normed:
            return self.normed_embedding_weights[word_idx]
        else:
            return self.embedding_weights[word_idx]
        

class Word2VecReader(object):
    """ This class loads pre-trained word embeddings from Tensorflow checkpoints."""

    def __init__(self, src_dir, vocab_size=50000):
        """
        Args:
            src_dir (str): specifies source directory of data.
            vocab_size: vocabulary size
        """
        self.src_dir = src_dir

        if not os.path.exists(self.src_dir):
            raise Exception('Datapath does not exist:\n "{}"'.format(self.src_dir))
        self.model_name = '5.9m'
        self.vocab_size = vocab_size
        
        self.checkpoints_path = os.path.join(self.src_dir, self.model_name, 'checkpoints')
        self.checkpoints_file = os.path.join(self.checkpoints_path, '{}.ckpt'.format(self.model_name))
        self.vocab_dir = os.path.join(self.src_dir, self.model_name, 'vocab')
        self.vocab_file = os.path.join(self.vocab_dir, 'vocab.csv')
        self.config_file = os.path.join(self.vocab_dir, 'config.csv')
        self.train_words_path = os.path.join(self.src_dir, self.model_name, 'train_words.pkl')

    def load_mappings(self):
        """ Loads mappings (index word-pairs) from CSV into two dictionaries.

        Returns:
            First dictionary maps indexes to words. Second dict maps vice versa.
        """
        print("Load mappings from {}".format(self.vocab_file))
        index_to_word = pd.read_csv(self.vocab_file, keep_default_na=False,
                                    na_values=[], encoding='latin-1')
        word_to_index = pd.read_csv(self.vocab_file, index_col='word',
                                    keep_default_na=False, na_values=[], encoding='latin-1')
        word_to_index.columns = ['index']
        
        return index_to_word.to_dict()['word'], word_to_index.to_dict()['index']

    def load_model_config(self):
        """ Load loss-sampling-size and embedding size from config file.

        Returns:
            Dictionary with config settings.
        """
        print("Load config from {}".format(self.config_file))
        config = pd.read_csv(self.config_file)
        config.columns = ['name', 'value']
        config = config.set_index(config['name'])['value']
        
        return config.to_dict()
        
    def create_graph(self, vocab_size, embedding_size):
        """ Creates a Tensorflow graph.

        Args:
            vocab_size: number of words in the vocabulary.
            embedding_size: dimensionality of the word embedding.

        Returns:
            tf-graph, embeddings, normalized embeddings
        """
        train_graph = tf.Graph()
        
        n_vocab = vocab_size
        n_embedding = embedding_size
        
        with train_graph.as_default():            
            # create embedding weight matrix
            embedding = tf.Variable(tf.random_uniform([n_vocab, n_embedding], minval=-1, maxval=1))
            
            # normalize embeddings
            norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
            normalized_embedding = tf.div(embedding, norm)
            
        return train_graph, embedding, normalized_embedding

    def load_word_embeddings(self):
        """ Loads word embeddings from the checkpoint specified during instantiation.

        Returns:
            Pre-trained Word2Vec instance
        """
        index_to_word, word_to_index = self.load_mappings()
        model_config = self.load_model_config()
        embedding_size = int(model_config['embedding_size'])
        
        train_graph, embedding, normalized_embedding = self.create_graph(len(index_to_word), embedding_size)
            
        with tf.Session(graph=train_graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.checkpoints_path))
            embedding_weights, normed_embedding_weights = sess.run([embedding, normalized_embedding])
            
        return Word2Vec(train_graph, index_to_word, word_to_index, embedding_weights, normed_embedding_weights)
