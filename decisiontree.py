from classificationalgo import *
import bisect
import collections


class DecisionTree(ClassificationAlgorithm):
    """
    Class implementing Decision Tree classification algorithm.
    """

    def predict(self, samples):
        """
        Method for predicting data.
        :param samples: data to predict
        :return: tuple of predicted values
        """
        predictions = list()
        for row in samples:
            predictions.append(self.predict_node(self.tree, row))
        return (predictions)

    def train(self, samples, actual_classes):
        """
        Method for training data
        :param samples: data set to train
        :param actual_classes: actual classes of each row with data
        """
        self.tree = self.build_tree(samples, actual_classes, 80, 5)

    def test_split(self, value, dataset, column_sorted):
        """
        Make a test split of data based on attribute and attribute value.
        :param value: attribute value used to split
        :param dataset: sorted dataset by column for which the split will be made
        :param column_sorted: sorted column for which the split will be made
        :return: two groups
        """
        left, right = list(), list()
        # using bisect to fast find split index
        i = bisect.bisect(column_sorted, value)
        left += dataset[:i]
        right += dataset[i:]

        return left, right

    def gini_index(self, groups, classes):
        """
        Calculate the Gini index for a split dataset.
        :param groups: split datasets
        :param classes: list of all classes
        :return: gini index
        """
        # count all samples
        total_size = float(sum([len(group) for group in groups]))

        # calculate gini for all groups
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0

            # score the group based on the score for each class
            counter = collections.Counter([row[-1] for row in group])
            for c in classes:
                p = counter[c] / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / total_size)
        return gini

    def get_split(self, dataset):
        """
        Select the best split point for dataset
        :param dataset: dataset
        :return: best split node
        """
        class_values = list(set(row[-1] for row in dataset))
        best_index, best_value, best_gini, best_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 2):
            print(index)
            # sorting dataset by actual column index (algorithm optimization)
            dataset_tmp = sorted(dataset, key=lambda x: x[index])
            # getting actual sorted column
            column_sorted = [row[index] for row in dataset_tmp]
            for row in dataset:
                # make split
                groups = self.test_split(row[index], dataset_tmp, column_sorted)

                # check quality of split
                gini = self.gini_index(groups, class_values)

                # if split is better - remember it
                if gini < best_gini:
                    best_index, best_value, best_gini, best_groups = index, row[index], gini, groups
        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    def make_leaf(self, group):
        """
        Create leaf node (last node).
        :param group:
        :return:
        """
        classes = [row[-1] for row in group]
        return max(set(classes), key=classes.count)

    def split(self, node, max_depth, min_size, depth):
        """
        Create splits for all nodes.
        :param node: actual node
        :param max_depth: max tree depth
        :param min_size: minimum node size
        :param depth: actual depth
        """
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.make_leaf(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.make_leaf(left), self.make_leaf(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.make_leaf(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth + 1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.make_leaf(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth + 1)

    def build_tree(self, train, actual_classes, max_depth, min_size):
        """
        Build a decision tree.
        :param train: train data
        :param actual_classes: classes for each row of train data
        :param max_depth: max tree depth
        :param min_size: minimum size of node
        :return: root of a tree
        """
        # append actual row classes to the end of each row
        training_data = train.tolist()
        for i in range(0, len(train)):
            training_data[i].append(actual_classes[i])

        # find root node
        root = self.get_split(training_data)

        # create rest of the tree basing on root node
        self.split(root, max_depth, min_size, 1)
        return root

    def predict_node(self, node, row):
        """
        Make a prediction with actual decision tree node.
        :param node: actual node
        :param row: row for prediction
        :return: found class
        """
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_node(node['left'], row)
            else:
                # if not dict - it is leaf
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_node(node['right'], row)
            else:
                # if not dict - it is leaf
                return node['right']
