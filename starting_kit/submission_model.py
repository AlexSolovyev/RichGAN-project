import numpy as np
import pandas as pd

class Model:
    def train(self, X, Y, n_bins=25):
        self.means = dict.fromkeys(Y.columns, np.zeros((n_bins, n_bins, n_bins)))
        self.stds = dict.fromkeys(Y.columns, np.zeros((n_bins, n_bins, n_bins)))
        self.bins = {}
        cols = X.columns
        for col in cols:
            self.bins[col] = np.percentile(X[col], 100*np.linspace(1./n_bins, (n_bins-1)/n_bins,n_bins-1))
        self.masks = create_masks(n_bins, X, self.bins)
        
        for i in range(n_bins):
            for j in range(n_bins):
                for k in range(n_bins):
                    f_masks = np.logical_and(self.masks[cols[0]][:,i], self.masks[cols[1]][:,j], self.masks[cols[2]][:,k]) 
                    for col in Y.columns:
                        self.means[col][i][j][k] = np.mean(Y[col][f_masks])
                        self.stds[col][i][j][k] = np.std(Y[col][f_masks])
        

    def predict(self, X):
        pred = pd.DataFrame()
        count = np.zeros((self.means['RichDLLk'].shape), dtype=np.int)
        n_bins = len(count)
        pred_masks = create_masks(n_bins, X, self.bins)
        cols = X.columns
        
        for col in self.means.keys():
            samples = np.array([])
            for i in range(n_bins):
                for j in range(n_bins):
                    for k in range(n_bins):
                        samples = np.append(samples,
                                         np.random.normal(
                                            loc=self.means[col][i][j][k],
                                            scale=self.stds[col][i][j][k],
                                            size=np.count_nonzero(np.logical_and(pred_masks[cols[0]][:,i],
                                                                   pred_masks[cols[1]][:,j],
                                                                   pred_masks[cols[2]][:,k]) )
                                         )
                                        )
            pred[col] = samples
        return pred
    
def create_masks(count, features, bins):
    for col in features.columns:
        mask = np.zeros((len(features[col]), count), dtype=np.bool)
        mask[:, 0], mask[:, -1] = features[col] <= bins[col][0], features[col] > bins[col][-1]
        for i in range(count - 2):
            mask[:,i+1] = (bins[col][i] < features[col]) & (features[col] <= bins[col][i+1])
        masks[col] = mask
    return masks
