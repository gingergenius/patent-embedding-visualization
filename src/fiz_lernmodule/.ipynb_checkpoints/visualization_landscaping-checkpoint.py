import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict
import pandas as pd
import seaborn as sns; sns.set()

class EmbeddingVisualizer(object):
    cpc_details = {
        "A": "Human Necessities",
        "B": "Operations and Transport",
        "C": "Chemistry and Metallurgy",
        "D": "Textiles",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering",
        "G": "Physics",
        "H": "Electricity",
        "Y": "Emerging Cross-Sectional Technologies"
    }

    def __init__(self, embeddings, metadata):
        self.tsne = TSNE()
        self.embeddings = embeddings
        print('Fit TSNE')
        self.datapoints = self.tsne.fit_transform(self.embeddings)
        
        loc_idxs = self.embeddings.index
        self.subset_metadata = metadata.loc[loc_idxs]
        self.plot_data = None
        
        print('EmbeddingVisualizer initialized')
        
    def plot_embeddings(self, detailed=False):
        if self.plot_data is None:
            data = { 
                "x": self.datapoints[:, 0],
                "y": self.datapoints[:, 1],
                "labels": self.subset_metadata["ExpansionLevel"],
                "cpc": self.subset_metadata["cpcs"],
                "text": self.subset_metadata["abstract_text"],
                "cpc_class": self.subset_metadata["cpcs"].apply(self.get_cpc_class)
            }
            
            self.plot_data = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(12, 12))
        
        if detailed:
            ax = sns.scatterplot(x="x", y="y", data = self.plot_data, hue="cpc_class")
        else:
            ax = sns.scatterplot(x="x", y="y", data = self.plot_data, hue="labels")

    def get_cpc_class(self, cpc_label):
        first_char = cpc_label[0].upper()
        cpc_class = self.cpc_details[first_char]
        return cpc_class
        
    def get_colors(self, detailed):
        if detailed:
            colors = {
                "A": "lime",
                "B": "black",
                "C": "yellow",
                "D": "red",
                "E": "blue",
                "F": "lightgrey",
                "G": "pink",
                "H": "brown",
                "Y": "orange"
            }
        else:
            colors = {
                'Seed': 'steelblue',
                'AntiSeed': 'red'
            }
        return colors
    
class LearningCurveVisualizer(object):
    def __init__(self, model):
        if not hasattr(model, 'history'):
            raise Exception('Model has no attribute: history. You need to train model instead of loading it.')
        self.history = model.history.history
        
    def plot_metrics(self, *metrics):
        length = len(self.history[metrics[0]])
        
        fig, ax = plt.subplots(figsize=(6,5))
        
        for metric in metrics:
            values = self.history[metric]
            plt.plot(range(1, length+1), values, label=metric)
        
        plt.xlabel('Epoch')
        plt.ylabel('Performance')
        plt.legend()
        