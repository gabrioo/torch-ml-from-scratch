import torch
import torch.nn as nn
from utils.data_operations import euclidean_distance

class KNN(nn.Module):
    """ K Nearest Neighbors classifier.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the 
        sample that we wish to predict.
    """
    def __init__(self, k= 5):
        super(KNN, self).__init__()
        self.k= k
        
    def _vote(self, neighbor_labels):
        """
        Return the most common class among the neighbor samples
        """
        counts= torch.bincount(neighbor_labels.int())
        return torch.argmax(counts)
        
    def predict(self, X_test, X_train, y_train):
        y_pred= torch.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx= torch.argsort(torch.Tensor([euclidean_distance(test_sample, x) for x in X_train]))[:self.k]
            
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors= torch.Tensor([y_train[i] for i in idx])
            
            # Label sample as the most common class label
            y_pred[i]= self._vote(k_nearest_neighbors)

        return y_pred.int()