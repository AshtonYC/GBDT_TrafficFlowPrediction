### GBDT def
import numpy as np
from regression_model.CART import RegressionTree
import time
from tqdm import tqdm

### Loss function and gradient
class SquareLoss():
    # Square Loss
    def loss(self, y, y_pred):
        sigma = 1
        if np.abs(y - y_pred) <= sigma:
            return 0.5 * np.power((y - y_pred), 2)
        else:
            return sigma * (np.abs((y - y_pred) - sigma / 2))


    # Square Loss Gradient
    def gradient(self, y, y_pred):
        sigma = 1
        y_return = np.zeros_like(y_pred)
        for i in range(len(y_pred)):
            if np.abs(y[i] - y_pred[i]) <= sigma:
                y_return[i] = (y[i] - y_pred[i])
            else:
                if (y[i] - y_pred[i]) >= 0:
                    y_return[i] = sigma * (y[i] - y_pred[i])
        return y_return


class GBDT(object):
    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_gini_impurity, max_depth, regression):
        ### Super
        # number of trees
        self.n_estimators = n_estimators
        # learning_rate
        self.learning_rate = learning_rate
        # minimum split sample
        self.min_samples_split = min_samples_split
        # minimum gini impurity
        self.min_gini_impurity = min_gini_impurity
        # max depth
        self.max_depth = max_depth
        # regression
        self.regression = regression
        # loss function
        self.loss = SquareLoss()

        if not self.regression:
            self.loss = None
        # trees superposition
        self.estimators = []
        for i in range(self.n_estimators):
            self.estimators.append(RegressionTree(min_samples_split=self.min_samples_split,
                                                  min_gini_impurity=self.min_gini_impurity,
                                                  max_depth=self.max_depth))

    # fit
    def fit(self, X, y, X_valid, y_valid):
        # forward initialization
        self.estimators[0].fit(X, y)
        # first prediction
        y_pred = self.estimators[0].predict(X)
        # forward iteration
        with tqdm(total=self.n_estimators-1) as pbar:
            for i in range(1, self.n_estimators):
                gradient = self.loss.gradient(y, y_pred)
                self.estimators[i].fit(X, gradient)
                self.estimators[i].Post_Pruning(X_valid, y_valid)
                # self.estimators[i].Post_Pruning(self.estimators[i],X_valid, y_valid)
                x_out = self.estimators[i].predict(X)
                y_pred -= np.multiply(self.learning_rate, x_out)
                time.sleep(0.000001)
                pbar.update(1)


    # prediction
    def predict(self, X):
        y_pred = self.estimators[0].predict(X)
        for i in range(1, self.n_estimators):
            y_pred += np.multiply(self.learning_rate, self.estimators[i].predict(X))
        return y_pred


### GBDT regression
class GBDTRegressor(GBDT):
    def __init__(self, n_estimators=300, learning_rate=0.025, min_samples_split=2,
                 min_var_reduction=1e3, max_depth=50):
    # def __init__(self, n_estimators=30, learning_rate=0.025, min_samples_split=2,
    #              min_var_reduction=1e8, max_depth=75):
        super(GBDTRegressor, self).__init__(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            min_samples_split=min_samples_split,
                                            min_gini_impurity=min_var_reduction,
                                            max_depth=max_depth,
                                            regression=True)
