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

    def __init__(self, src_dir=None):
        if src_dir is None:
            self.dir_path = "/Users/sblank/PycharmProjects/patents-public-data/models/landscaping/data/video_codec"
        else:
            self.dir_path = os.path.join(src_dir, 'models/landscaping/data/video_codec')
        self.filename = "landscape_data.pkl"

    def load_data(self):
        """ Reads **landscape_data.pkl** from the directory specified during instantiation.

        Returns:
            pd.DataFrame with training data

        Raises:
            Exception: data path does not exist.

        """
        data_path = os.path.join(self.dir_path, self.filename)

        if not os.path.exists(data_path):
            raise Exception('Data path does not exist:\n "{}"'.format(data_path))
        else:
            print("Loading data from {}".format(data_path))

            with open(data_path, "rb") as input_file:
                dataset_deserialized = pickle.load(input_file)

                training_data_full_df, seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents = \
                    dataset_deserialized

            self.training_data_full_df = training_data_full_df
            self.seed_patents_df = seed_patents_df
            self.l1_patents_df = l1_patents_df
            self.l2_patents_df = l2_patents_df
            self.anti_seed_patents = anti_seed_patents
            print("Finished loading.")
            
        return training_data_full_df
