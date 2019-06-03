import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import seaborn as sns 

sns.set()
sns.set_context("notebook")


class EmbeddingVisualizer(object):
    ipc_details = {
        "A": "Human Necessities",
        "B": "Operations and Transport",
        "C": "Chemistry and Metallurgy",
        "D": "Textiles",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering",
        "G": "Physics",
        "H": "Electricity"
    }

    def __init__(self, reducer, embeddings, metadata=None):
        self.reducer = reducer
        self.embeddings = embeddings
        print('Fit ', self.reducer)
        self.datapoints = self.reducer.fit_transform(self.embeddings)
        
        loc_idxs = self.embeddings.index
        if (metadata is not None):
            self.subset_metadata = metadata.loc[loc_idxs]
        self.plot_data = None
        
        print('EmbeddingVisualizer initialized')


    def plot_embeddings(self, detailed=False, label=True, density=5, terms=1):
        data = { 
            "x": self.datapoints[:, 0],
            "y": self.datapoints[:, 1],         
        }

        if detailed:
            data["ipc_class"] = self.subset_metadata["ipc_classes"].apply(self.get_ipc_class)

        if label:
            data["terms"] = self.subset_metadata["terms"].apply(self.get_terms, n_terms=terms)
            
        self.plot_data = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(20, 20))
        
        if detailed:
            ax = sns.scatterplot(x="x", y="y", data = self.plot_data, hue="ipc_class")
        else:
            ax = sns.scatterplot(x="x", y="y", data = self.plot_data)

        if label:
            self.label_point(self.plot_data.x, self.plot_data.y, self.plot_data.terms, plt.gca(), density)  

        ax.set_title(self.reducer)

    def label_point(self, x, y, val, ax, density):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            if i % density == 0:
                ax.text(point['x']+.02, point['y'], str(point['val']))


    def get_ipc_class(self, ipc_label):
        ipc_class="No IPC class available"
        ipc_label = str(ipc_label)
        if len(ipc_label) > 0:
            first_char = ipc_label[0].upper()
            if first_char in self.ipc_details:
                ipc_class = self.ipc_details[first_char]
        return ipc_class


    def get_terms(self, terms, n_terms):
        first_n_terms = terms[0:n_terms]
        label = "/".join(first_n_terms)
        return label


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
        