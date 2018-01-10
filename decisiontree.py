from classificationalgo import *
import bisect
import collections

class DecisionTree(ClassificationAlgorithm):
    def predict(self, samples):
        predictions = list()
        for row in samples:
            predictions.append(self.predict_node(self.tree, row))
        return (predictions)

    def train(self, samples, responses):
        self.tree = self.build_tree(samples, responses, 80, 5)

    # Split a dataset based on an attribute and an attribute value
    def test_split(self, value, dataset, column_sorted):
        left, right = list(), list()
        i = bisect.bisect(column_sorted, value)
        left += dataset[:i]
        right += dataset[i:]

        return left, right

    # Calculate the Gini index for a split dataset
    def gini_index(self, groups, classes):
        # count all samples at split point
        total_size = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
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

    # Select the best split point for a dataset
    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        best_index, best_value, best_gini, best_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 2):
            print(index)
            dataset_tmp = sorted(dataset, key=lambda x: x[index])
            column_sorted = [row[index] for row in dataset_tmp]
            for row in dataset:
                groups = self.test_split(row[index], dataset_tmp, column_sorted)
                gini = self.gini_index(groups, class_values)
                if gini < best_gini:
                    best_index, best_value, best_gini, best_groups = index, row[index], gini, groups
        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    # Create a leaf node value
    def make_leaf(self, group):
        classes = [row[-1] for row in group]
        return max(set(classes), key=classes.count)

    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, depth):
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

    # Build a decision tree
    def build_tree(self, train, responses, max_depth, min_size):
        training_data = train.tolist()
        for i in range(0, len(train)):
            training_data[i].append(responses[i])

        root = self.get_split(training_data)
        self.split(root, max_depth, min_size, 1)
        return root

    # Make a prediction with a decision tree
    def predict_node(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_node(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_node(node['right'], row)
            else:
                return node['right']



