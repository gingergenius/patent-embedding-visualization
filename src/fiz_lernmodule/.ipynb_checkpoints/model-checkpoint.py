"""
Used for patent landscaping use-case.
"""

import os

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Embedding, BatchNormalization, ELU, Concatenate
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.core import Dropout
from fiz_lernmodule.keras_metrics import precision, recall, f1score

class LandscapingModel:
    """ Neural network that can be trained to classify seed/ anti_seed patents. """

    target_names = ['seed', 'antiseed']

    def __init__(self, data, data_path, opt):

        self.tf_model = None
        self.data = data
        self.data_path = data_path
        self.seed_name = 'video_codec'

        self.batch_size = opt['batch_size']
        self.dropout = opt['dropout']
        self.num_epochs = opt['num_epochs']
        self.lstm_size = opt['lstm_size']

        self.max_seq_length = self.data.max_seq_length

    def set_up_model_architecture(self):
        """ Builds neural network architecture.

        Returns:
            -
        """
        refs_input = Input(shape=(self.data.trainRefsOneHotX.shape[1],), name='refs_input')
        refs = Dense(
            256,
            input_dim=self.data.trainRefsOneHotX.shape[1],
            activation=None)(refs_input)
        refs = Dropout(self.dropout)(refs)
        refs = BatchNormalization()(refs)
        refs = ELU()(refs)
        refs = Dense(64, activation=None)(refs)
        refs = Dropout(self.dropout)(refs)
        refs = BatchNormalization()(refs)
        refs = ELU()(refs)

        cpcs_input = Input(shape=(self.data.trainCpcOneHotX.shape[1],), name='cpcs_input')
        cpcs = Dense(
            32,
            input_dim=self.data.trainCpcOneHotX.shape[1],
            activation=None)(cpcs_input)
        cpcs = Dropout(self.dropout)(cpcs)
        cpcs = BatchNormalization()(cpcs)
        cpcs = ELU()(cpcs)

        # Use pre-trained Word2Vec embeddings
        embedding_layer_input = Input(shape=(self.max_seq_length,), name='embed_input')
        embedding_layer = Embedding(self.data.embedding_model.embedding_weights.shape[0],
                                    self.data.embedding_model.embedding_weights.shape[1],
                                    weights=[self.data.embedding_model.embedding_weights],
                                    input_length=self.max_seq_length,
                                    trainable=False)(embedding_layer_input)

        deep = LSTM(
            self.lstm_size,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
            return_sequences=False,
            name='LSTM_1')(embedding_layer)
        deep = Dense(300, activation=None)(deep)
        deep = Dropout(self.dropout)(deep)
        deep = BatchNormalization()(deep)
        deep = ELU()(deep)

        # model_inputs_to_concat = [cpcs, refs, deep]
        model_inputs_to_concat = [refs, deep]

        final_layer = Concatenate(name='concatenated_layer')(model_inputs_to_concat)
        output = Dense(64, activation=None)(final_layer)
        output = Dropout(self.dropout)(output)
        output = BatchNormalization()(output)
        output = ELU()(output)
        output = Dense(1, activation='sigmoid')(output)

        # model = Model(inputs=[cpcs_input, refs_input, embedding_layer_input], outputs=output, name='model')
        model = Model(inputs=[refs_input, embedding_layer_input], outputs=output, name='model')
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', precision, recall, f1score])

        self.tf_model = model
        print('Done building graph.')
        print(self.tf_model.summary())

    def train_or_load_model(self, train_or_load = "load"):
        """ Loads model parameters from file or train model was not trained yet.

        Returns:
            -
        """
        model_dir = os.path.join(self.data_path, self.seed_name)
        model_path = os.path.join(model_dir, 'model.pb')

        if (os.path.exists(model_path)) and (train_or_load == "load"):
            print('Model exists at {}; loading existing trained model.'.format(model_path))
            self.tf_model = keras.models.load_model(
                model_path,
                custom_objects={'precision': precision, 'recall': recall, 'fmeasure': f1score})
        else:
            print('Model has not been trained yet.')
            tf_model = self.train_model(self.tf_model, self.batch_size, self.num_epochs)
            print('Saving model to {}'.format(model_path))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            tf_model.save(model_path)
            print('Model persisted and ready for inference!')

    def train_model(self, model, batch_size, num_epochs):
        """

        Args:
            model (tf-model):
            batch_size (int):  number of samples in a batch.
            num_epochs (int): number of epochs.

        Returns:
            Updated model.
        """
        print('Training model.')
        self.history = model.fit(x={
            'refs_input': self.data.trainRefsOneHotX,
            'embed_input': self.data.padded_train_embed_idxs_x,
            # 'cpcs_input': self.data.trainCpcOneHotX
            },
            y=self.data.trainY,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(
                {
                    'refs_input': self.data.testRefsOneHotX,
                    # 'cpcs_input': self.data.testCpcOneHotX,
                    'embed_input': self.data.padded_test_embed_idxs_x},
                self.data.testY))
        return model

    def evaluate_model(self):
        """ Evaluates the trained models performance on the test set.

        Returns:
            Tuple with the observed metrics.
        """
        loss, acc, p, r, f1 = self.tf_model.evaluate(
            x={
                'refs_input': self.data.testRefsOneHotX,
                'embed_input': self.data.padded_test_embed_idxs_x
            },
            y=self.data.testY,
            batch_size=self.batch_size
        )

        print('')
        print('Test loss: {:.4f}'.format(loss))
        print('Test accuracy: {:.4f}'.format(acc))
        print('Test p/r (f1): {:.2f}/{:.2f} ({:.2f})'.format(p, r, f1))

        return (loss, acc, p, r, f1)

    def get_average_word_embeddings_as_document_embedding(self, train_or_test, filename):
        """ Extracts the word embeddings for training or test split
        
        Returns:
        
        """
        print('This may take a moment.')
        embedding_layer_model = Model(inputs=self.tf_model.input,
                                outputs=self.tf_model[5].output)
        if train_or_test == 'train':
            embeddings = embedding_layer_model.predict(x={
                        'refs_input': self.data.trainRefsOneHotX,
                        'embed_input': self.data.padded_train_embed_idxs_x
                    })
            idxs_after_shuffling=list(self.data.idxs_after_shuffling[:self.data.final_layer])
        else:
            embeddings = embedding_layer_model.predict(x={
                        'refs_input': self.data.testRefsOneHotX,
                        'embed_input': self.data.padded_test_embed_idxs_x
                    })
            idxs_after_shuffling=list(self.data.idxs_after_shuffling[self.data.final_layer:])
        average_embeddings = embeddings.mean(axis=1)
        
        assert (average_embeddings.shape[0] == len(idxs_after_shuffling)), \
            "Dim of embeddings %r and idxs %r do not match" % (average_embeddings.shape[0],len(idxs_after_shuffling))
        
        
        
        output = pd.DataFrame(average_embeddings, index=idxs_after_shuffling)
        
        
        print('Finished.')