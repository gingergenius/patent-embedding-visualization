"""
Used for patent landscaping use-case.
"""

import os
import pickle


class LandscapeDataReader(object):
    """ This class provides a data reader for the PatentLandscape use-case."""
    training_data_full_df = None
    seed_patents_df = None
    l1_patents_df = None
    l2_patents_df = None
    anti_seed_patents = None
    src_dir = None
    file_path = None
    seed_name = None
    filename = None

    def __init__(self, src_dir, filename="landscape_data.pkl"):
        self.src_dir = src_dir
        self.filename = filename

    def load_data(self, seed_name="video_codec", return_everything=False):
        """ Reads **landscape_data.pkl** from the directory specified during instantiation.

        Returns:
            pd.DataFrame with training data

        Raises:
            Exception: data path does not exist.

        """
        self.seed_name = seed_name
        self.file_path = os.path.join(self.src_dir, 'data', self.seed_name, self.filename)

        if not os.path.exists(self.file_path):
            raise Exception('Data path does not exist:\n "{}"'.format(self.file_path))
        else:
            print("Loading data from {}".format(self.file_path))

            with open(self.file_path, "rb") as input_file:
                dataset_deserialized = pickle.load(input_file)

                training_data_full_df, seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents = \
                    dataset_deserialized

            self.training_data_full_df = training_data_full_df
            self.seed_patents_df = seed_patents_df
            self.l1_patents_df = l1_patents_df
            self.l2_patents_df = l2_patents_df
            self.anti_seed_patents = anti_seed_patents
            print("Finished loading.")
        if return_everything:
            return training_data_full_df, seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents    
        else:
            return training_data_full_df

    def get_seed_names(self):
        """ Checks which seed names exist in the src directory.

        Returns: list with seed names

        """
        return os.listdir(os.path.join(self.src_dir, 'data'))