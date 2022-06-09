import time
from tqdm import tqdm
import numpy as np
from DecisionTreeMaster.PurnDT.Post_Pruning import PostPruning_IMEP
import re

### define split feature
def feature_split(X, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_left = np.array([sample for sample in X if split_func(sample)])
    X_right = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_left, X_right])


### gini impuirty
def calculate_gini(y):
    # to list
    y = y.tolist()
    probs = [y.count(i) / len(y) for i in np.unique(y)]
    gini = sum([p * (1 - p) for p in probs])
    return gini


### define nodes
class TreeNode():
    def __init__(self, feature_i=None, threshold=None,
                 leaf_value=None, left_branch=None, right_branch=None, n_samples=None):
        # feature index
        self.feature_i = feature_i
        # feature threshold
        self.threshold = threshold
        # leaf value
        self.leaf_value = leaf_value
        # left child
        self.left_branch = left_branch
        # right child
        self.right_branch = right_branch

        self.n_samples = n_samples



    ### prune
    def prune(self, min_criterion, n_samples):
        if self.feature_i is None:
            return

        self.left_branch.prune(min_criterion, n_samples)
        self.right_branch.prune(min_criterion, n_samples)

        pruning = False

        if self.left_branch.feature_i is None and self.right_branch.feature_i is None:
            if self.leaf_value is None:
                flag = 0
                # print(self.leaf_value * float(self.n_samples) / n_samples)
                if self.left_branch.leaf_value is None:
                    self.left_branch.leaf_value = 0
                    flag = 1
                if self.right_branch.leaf_value is None:
                    self.right_branch.leaf_value = 0
                    flag = 2
                Left = self.left_branch.n_samples * self.left_branch.leaf_value
                Right = self.right_branch.n_samples * self.right_branch.leaf_value
                self.leaf_value = (Left + Right) / self.n_samples
                if (self.leaf_value * float(self.n_samples) / n_samples) < min_criterion:
                    pruning = True
                self.leaf_value = None
                if flag == 1:
                    self.left_branch.leaf_value = None
                elif flag == 2:
                    self.right_branch.leaf_value = None
            else:
                if (self.leaf_value * float(self.n_samples) / n_samples) < min_criterion:
                    pruning = True

        # if pruning:
        self.left_branch = None
        self.right_branch = None
        self.feature_i = None


###  Binaray DT
class BinaryDecisionTree(object):
    ### Initialization
    def __init__(self, min_samples_split=2, min_gini_impurity=999,
                 max_depth=float("inf"), loss=None):
        # root
        self.root = None
        # minimum sample split
        self.min_samples_split = min_samples_split
        # initial gini impurity
        self.mini_gini_impurity = min_gini_impurity
        # max depth
        self.max_depth = max_depth
        # gini impurity
        self.impurity_calculation = None
        # leaf node value
        self._leaf_value_calculation = None
        # loss function
        self.loss = loss
        self.n_samples = None



    ### fit
    def fit(self, X, y, loss=None):
        self.root = self._build_tree(X, y)
        self.loss = None
        # self.root.prune(min_criterion=1e-7, n_samples=self.root.n_samples)
        # PostPruning_IMEP(self, y, X, 0.1)

    ### build
    def _build_tree(self, X, y, current_depth=0, leaf_value=None):
        # initialize gini impurity
        init_gini_impurity = float("inf")
        # initialize best criteria
        best_criteria = None
        # initial best set
        best_sets = None
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        Xy = np.concatenate((X, y), axis=1)
        # sample and feature numbers
        n_samples, n_features = X.shape
        self.n_samples = n_samples
        # number of sample should be larger than min and less than max
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # gini for all features
            for feature_i in range(n_features):
                # feature value for ith feature
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # # save the impurity value
                # impurity_record = np.zeros_like(Xy)
                # obtain best impurity
                for threshold in unique_values:
                    # split
                    Xy1, Xy2 = feature_split(Xy, feature_i, threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        impurity = self.impurity_calculation(y, y1, y2)

                        # min gini
                        # best feature index and criteria and sets
                        if impurity < init_gini_impurity:
                            init_gini_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],
                                "lefty": Xy1[:, n_features:],
                                "rightX": Xy2[:, :n_features],
                                "righty": Xy2[:, n_features:]
                            }
                # print('min_impurity = %f feature = %d' %(init_gini_impurity, feature_i))

        # update children
        if init_gini_impurity < self.mini_gini_impurity:
            left_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            right_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            # print('Leaf value = %f' % leaf_value)
            # print('feature = %f' % feature_i)
            return TreeNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], left_branch=left_branch, right_branch=right_branch, n_samples=n_samples)

        # leaf value
        leaf_value = self._leaf_value_calculation(y)
        # print('Leaf value = ', leaf_value)
        # print('feature = %f' % feature_i)
        return TreeNode(leaf_value=leaf_value, n_samples=n_samples)

    ### Prediction
    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        # return if leaf value exist
        if tree.leaf_value is not None:
            return tree.leaf_value

        # obtain feature value
        feature_value = x[tree.feature_i]

        # children tree
        branch = tree.right_branch
        # print('Tree_leaf_value = %f' % tree.leaf_value)
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch

        # test
        return self.predict_value(x, branch)

    # Leaf Error calculation
    def LeafError(self, X_valid, y_valid, tree=None):
        if tree is None:
            tree = self.root
            # return if leaf value exist
        if tree.leaf_value is not None:
            error = float(tree.leaf_value - y_valid)
            return error

        # obtain feature value
        # feature_value = X_valid[tree.feature_i]
        feature_value = self.predict(X_valid)
        # children tree
        branch = tree.right_branch
        # print('Tree_leaf_value = %f' % tree.leaf_value)
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch

        # test
        return self.LeafError(X_valid, y_valid, branch)

    ### predict function
    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def delNode(self, Node):
        if Node is not None:
            self.delNode(Node.left_branch)
            self.delNode(Node.right_branch)
            Node.left_branch = Node.right_branch = None
            Node.root = None

    ###
    def deletion(self, root, current_node):
        if root == None:
            return None
        if root == current_node:
            # root.left_branch = None
            # root.right_branch = None
            # root.root = None
            self.delNode(root)
            return 1
        else:
            return 0



    ### Post Pruning
    def current_accuracy(self, X_valid, y_valid):
    # def Post_Pruning(self, X_valid, y_valid):
        accuracy = 0
        for i in range(len(y_valid)):
            x = X_valid[[i], :]
            x_i = x.T
            x_i = x_i.reshape(len(x_i), )
            error = (self.predict_value(x_i) - y_valid[i])
            if np.abs(error) <= 1e-2:
                accuracy += 1
        return accuracy/len(y_valid)
        #     if error > 0.25:
        #         self.root.prune(min_criterion=1e-6, n_samples=self.root.n_samples)


    ###
    def Post_Pruning(self, decision_tree=TreeNode(), X_valid=[], y_valid=[]):
        leaf_facther =[]
        bianli_list =[]
        bianli_list.append(decision_tree)

        while len(bianli_list) > 0:
            current_node = bianli_list[0]
            # if current_node.root is None:
            #     children = [current_node.left_branch, current_node.right_branch]
            # else:
            children = current_node.root
            wanted = True
            if not (children.left_branch is None and children.right_branch is None):
                # while not (children.left_branch is None and children.right_branch is None):
                child_l = children.left_branch
                child_r = children.right_branch
                bianli_list.append(child_l)
                bianli_list.append(child_r)
                child_l.root = children.left_branch
                child_r.root = children.right_branch
                temp_bool = (child_l.left_branch is None and child_l.right_branch is None and child_r.left_branch is
                             None and child_r.right_branch is None)
                wanted = wanted and temp_bool
            else:
                wanted = False

            if wanted:
                leaf_facther.append(current_node)
            bianli_list.remove(current_node)

        while len(leaf_facther) > 0:
            current_node = leaf_facther.pop()
            before_accuracy = self.current_accuracy(X_valid=X_valid, y_valid=y_valid)
            temp_self = self

            flag = self.deletion(self.root, current_node)
            temp_list = [self.root]
            while flag != 1:
                temp = temp_list.pop()
                flag = self.deletion(temp, current_node)
                if flag != 1:
                    if temp.left_branch is not None:
                        temp_list.append(temp.left_branch)
                    if temp.right_branch is not None:
                        temp_list.append(temp.right_branch)
                # if temp.left_branch is not None:
                #     flag = self.deletion(temp.left_branch, current_node)
                #     temp_list.append(temp.left_branch)
                # if flag != 1:
                #     if temp.right_branch is not None:
                #         flag = self.deletion(temp.right_branch, current_node)
                #         temp_list.append(temp.right_branch)

            later_accuracy = self.current_accuracy(X_valid=X_valid, y_valid=y_valid)

            if before_accuracy > later_accuracy:
                self = temp_self

        return decision_tree

### CART regression
class RegressionTree(BinaryDecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = np.var(y, axis=0)
        var_y1 = np.var(y1, axis=0)
        var_y2 = np.var(y2, axis=0)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        variance_reduction = var_tot - (frac_1 * var_y1 + frac_2 * var_y2)

        return sum(variance_reduction)

    # mean leaf value
    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self.impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)



