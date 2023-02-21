# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import math
import random

from game import Directions


class Classifier:
    def __init__(self):
        self.classifier = None

    def reset(self):
        pass

    def fit(self, data, target):
        rf_classifier = RandomForestClassifier(data, target, n_estimators=200)
        rf_classifier.fit()
        self.classifier = rf_classifier

    def predict(self, data, legal=None):
        return self.classifier.predict(data, legal)


# Random Forest Classifier
# data is an array of arrays of integers (0 or 1) indicating state
# target is an array of integers 0-3 indicating the action taken in each state
# n_estimators is the number of decision trees to use in the classifier
class RandomForestClassifier:
    def __init__(self, data, target, n_estimators=100, bootstrap=True):
        self.data = data
        self.target = target
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.trees = []

    # Sample, with replacement n training examples, where n is the number of training examples
    def get_bootstrap(self):
        if self.bootstrap:
            indices = []
            for i in range(len(self.data)):
                indices.append(random.randrange(0, len(self.data)))
            data = [self.data[j] for j in indices]
            target = [self.target[j] for j in indices]
            return data, target
        return self.data, self.target

    # Train n decision trees with bootstrapping, and a subset of all features for each tree,
    # where n is n_estimators
    def fit(self):
        for i in range(self.n_estimators):
            number_of_features = len(self.data[0])
            feature_subset_size = round(math.sqrt(number_of_features))
            feature_subset = random.sample(range(number_of_features), feature_subset_size)
            data, target = self.get_bootstrap()
            dt = DecisionTree(data, target, features=feature_subset)
            dt.build_tree()
            self.trees.append(dt)

    # Predict best move based on input data and return the best move that is legal
    # Prediction is performed by choosing the common predictions by all decision trees
    def predict(self, data, legal):

        # create a dictionary containing each move and their respective occurrences
        predictions_count = dict()
        for dt in self.trees:
            prediction = dt.get_prediction(data)
            predictions_count[prediction] = predictions_count.get(prediction, 0) + 1

        # remove the STOP move from legal
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # convert legal moves to numbers
        move_to_number = {Directions.NORTH: 0,
                          Directions.EAST: 1,
                          Directions.SOUTH: 2,
                          Directions.WEST: 3}
        legal_num = [move_to_number[m] for m in legal]

        # filter predictions that are legal
        legal_predictions = [m for m in predictions_count.items() if m[0] in legal_num]

        # return the prediction with the greatest occurrence
        return max(legal_predictions, key=lambda x: x[1])[0]


# A node in a decision tree
# children is a list of all child nodes
# feature is the feature that a node represent
# value is the value of a feature that a node represent
# prediction is the prediction of a node, None for non-leaf nodes
class Node:
    def __init__(self, children=None, feature=None, value=None, prediction=None):
        if children is None:
            children = []
        self.children = children
        self.feature = feature
        self.value = value
        self.prediction = prediction


# Decision tree classifier
# root is the root node of the decision tree
class DecisionTree:
    def __init__(self, data, targets, features=None, root=None):
        self.data = data
        self.targets = targets
        self.root = root
        self.total_entropy = self.get_total_entropy()
        if features is None:
            features = range(len(self.data[0]))
        self.features = set(features)

    # Return the entropy of all targets (labels)
    def get_total_entropy(self):
        examples = [0, 0, 0, 0]
        for target in self.targets:
            examples[target] += 1

        return self.get_entropy(examples)

    # Return the entropy of some examples
    def get_entropy(self, examples):
        entropy = 0
        total_examples = sum(examples)
        for example in examples:
            if example != 0:
                probability = example / total_examples
                entropy -= probability * math.log2(probability)

        return entropy

    # Return the information gain of some examples
    def get_information_gain(self, feature, data, targets):
        examples = dict()
        for i in range(len(data)):
            value = data[i][feature]
            label = targets[i]
            example = examples.get(value, [0, 0, 0, 0])
            example[label] += 1
            examples[value] = example

        total_examples = sum([sum(values) for values in examples.values()])
        information_gain = self.total_entropy
        for examples in examples.values():
            information_gain -= (sum(examples) / total_examples) * self.get_entropy(examples)

        return information_gain

    # Return the plurality value by randomly choosing a target based on a weighted distribution
    def get_plurality_value(self, targets):
        frequency = dict()
        for target in targets:
            frequency[target] = frequency.get(target, 0) + 1
        sequence = list(frequency.keys())
        distribution = [f / len(targets) for f in frequency.values()]

        return random.choices(sequence, weights=distribution, k=1)[0]

    # Build the decision tree from root
    def build_tree(self):
        self.root = self.build(self.features, self.data, self.targets)

    # Recursively build a decision tree
    def build(self, features, data, targets, parent_targets=None):

        # if there are no examples left then perform plurality classification based on parent node
        if len(targets) == 0:
            return Node(prediction=self.get_plurality_value(parent_targets))

        # if there are no features left then preform plurality classification based on remaining examples
        elif len(features) == 0:
            return Node(prediction=self.get_plurality_value(targets))

        # if all remaining examples are of the same target (label) then that target is the prediction
        elif targets.count(targets[0]) == len(targets):
            return Node(prediction=targets[0])

        # otherwise choose the best feature to split the examples
        current_node = Node()

        best_feature = max([i for i in features], key=lambda x: self.get_information_gain(x, data, targets))

        current_node.feature = best_feature  # current node represents the current best feature to split
        features.remove(best_feature)  # remove the current best feature from remaining features

        possible_values = [0, 1]  # the possible values for each feature
        for value in possible_values:

            remaining_data = []
            remaining_targets = []
            for i in range(len(data)):
                if data[i][best_feature] == value:
                    remaining_data.append(data[i])
                    remaining_targets.append(targets[i])

            # build child node with the remaining features, data and targets
            child_node = self.build(
                features, remaining_data, remaining_targets, targets)
            child_node.value = value
            current_node.children.append(child_node)

        return current_node

    # Return a prediction for input data by traversing the decision tree until a leaf node is reached
    def get_prediction(self, input_data):
        node = self.root
        while node.prediction is None:
            for child_node in node.children:
                if child_node.value == input_data[node.feature]:
                    node = child_node
                    break
        return node.prediction
