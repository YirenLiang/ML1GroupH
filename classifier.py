# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import math
import random


class Classifier:
    def __init__(self):
        pass

    def reset(self):
        pass

    def fit(self, data, target):
        dt = DecisionTree(data, target, features=[0, 2, 4, 19])  # initialise decision tree
        dt.build_tree()  # build tree
        print(dt.get_prediction(data[15]))  # make prediction base on input
        dt.print_tree(dt.root)
        pass

    def predict(self, data, legal=None):
        return 1


class Node:
    def __init__(self, children=None, feature=None, value=None, prediction=None, is_leaf=False):
        if children is None:
            children = []
        self.children = children
        self.feature = feature
        self.value = value
        self.prediction = prediction


class DecisionTree:
    def __init__(self, data, targets, features=None, root=None):
        self.data = data
        self.targets = targets
        self.root = root
        self.total_entropy = self.get_total_entropy()
        if features is None:
            features = range(len(self.data[0]))
        self.features = set(features)

    def get_total_entropy(self):
        examples = [0, 0, 0, 0]
        for target in self.targets:
            examples[target] += 1

        return self.get_entropy(examples)

    def get_entropy(self, examples):
        entropy = 0
        total_examples = sum(examples)
        for example in examples:
            if example != 0:
                probability = example / total_examples
                entropy -= probability * math.log2(probability)

        return entropy

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

    def get_plurality_value(self, targets):
        frequency = dict()
        for target in targets:
            frequency[target] = frequency.get(target, 0) + 1
        sequence = list(frequency.keys())
        distribution = [f / len(targets) for f in frequency.values()]

        return random.choices(sequence, weights=distribution, k=1)

    def build_tree(self):
        self.root = self.build(self.features, self.data, self.targets)

    def build(self, features, data, targets, parent_targets=None):
        if len(targets) == 0:
            return Node(prediction=self.get_plurality_value(parent_targets)[0])

        elif len(features) == 0:
            return Node(prediction=self.get_plurality_value(targets)[0])

        elif targets.count(targets[0]) == len(targets):
            return Node(prediction=targets[0])

        current_node = Node()

        best_feature = max([i for i in features], key=lambda x: self.get_information_gain(x, data, targets))

        current_node.feature = best_feature
        features.remove(best_feature)

        possible_values = [0, 1]
        for value in possible_values:
            remaining_data = []
            remaining_targets = []
            for i in range(len(data)):
                if data[i][best_feature] == value:
                    remaining_data.append(data[i])
                    remaining_targets.append(targets[i])

            child_node = self.build(features, remaining_data, remaining_targets, targets)
            child_node.value = value
            current_node.children.append(child_node)

        return current_node

    def get_prediction(self, input_data):
        node = self.root
        while node.prediction is None:
            for child_node in node.children:
                if child_node.value == input_data[node.feature]:
                    node = child_node
                    break
        return node.prediction

    def get_left_node(self, node):
        if len(node.children) > 0:
            return node.children[0]
        return None

    def get_right_node(self, node):
        if len(node.children) == 2:
            return node.children[1]
        return None

    def print_tree(self, node, level=0):
        if node is not None:
            self.print_tree(self.get_left_node(node), level + 1)
            if node.prediction is not None:
                print(f"{' ' * 4 * level}->{node.value}({node.prediction})")
            else:
                print(f"{' ' * 4 * level}->{node.value}[{node.feature}]")
            self.print_tree(self.get_right_node(node), level + 1)

    def print_prediction(self, data):
        print("the prediction for:")
        print(data)
        print(self.get_prediction(data))
