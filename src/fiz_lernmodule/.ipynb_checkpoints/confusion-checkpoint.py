import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    def __init__(self, target, prediction):
        self.cm = confusion_matrix(target, prediction)

    def plot_matrix(self, fig=None, ax=None, labels=None):
        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(figsize=(8, 6.5), dpi=75)
        if labels is None:
            labels = ["True", "False"]

        sns.heatmap(self.cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()