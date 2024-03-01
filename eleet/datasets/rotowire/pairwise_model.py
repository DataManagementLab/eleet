import numpy as np

class PairwiseModel():
    def __init__(self, base_model):
        self.base_model = base_model

    def fit(self, X, y, groups):
        train_X, train_y = [], []
        for g in range(groups.max() + 1):
            selector = groups == g
            gX = X[selector]
            gy = y[selector]
            correct = gy == 1
            if not correct.any():
                continue

            num_incorrect = (~correct).sum()

            train_X1 = np.hstack((gX[correct].repeat(num_incorrect, 0), gX[~correct]))
            train_X2 = np.hstack((gX[~correct], gX[correct].repeat(num_incorrect, 0)))
            train_X.extend([train_X1, train_X2])
            train_y.extend([np.zeros(num_incorrect), np.ones(num_incorrect)])
        return self.base_model.fit(np.vstack(train_X), np.hstack(train_y))
    
    def predict(self, X):
        indizes = np.stack((np.repeat(np.arange(X.shape[0]), X.shape[0]), np.tile(np.arange(X.shape[0]), X.shape[0])))
        indizes = indizes[:, indizes[0] < indizes[1]]
        if not indizes.shape[1]:
            return 0
        paired_X = np.hstack((X[indizes[0]], X[indizes[1]]))
        pred = self.base_model.predict(paired_X)
        wins = indizes[pred.astype(int), np.arange(indizes.shape[1])]
        i, counts = np.unique(wins, return_counts=True)
        result = i[counts.argmax()]
        return result
