# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import math


class Classifier:
    def __init__(self):
        pass

    def reset(self):
        pass

    def fit(self, data, target):
        pass

    def predict(self, data, legal=None):
        return 1


# if data is equal to 1 then choose right node
# otherwise choose left
class Node:
    def __init__(self, is_leaf=None, left=None, right=None, information_gain=None, feature=None, value=None, data=None):
        self.is_leaf = is_leaf
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.feature = feature
        self.value = value
        self.data = data


class DecisionTree:
    def __init__(self, data, root=None):
        self.data = data
        self.root = root
        self.total_entropy = self.get_total_entropy()

    def get_total_entropy(self):
        examples = [0, 0, 0, 0]
        for row in self.data:
            label = row[-1]
            examples[label] += 1

        return self.get_entropy(examples)

    def get_entropy(self, examples):
        entropy = 0
        total_examples = sum(examples)
        for example in examples:
            if example != 0:
                probability = example / total_examples
                entropy -= probability * math.log2(probability)

        return entropy

    # example_counts = {"0": [2,3,1,2], "1": [1,2,0,2]}
    def get_information_gain(self, feature, data):
        nodes_examples = dict()
        for row in data:
            value = row[feature]
            label = row[-1]
            examples = nodes_examples.get(nodes_examples, [0, 0, 0, 0])
            examples[label] += 1
            nodes_examples[value] = examples

        total_examples = sum(sum(nodes_examples.values()))
        information_gain = self.total_entropy
        for examples in nodes_examples.values():
            information_gain -= (sum(examples) / total_examples) * self.get_entropy(examples)

        return information_gain

    def is_leaf(self, node):
        labels = [row[-1] for row in node.data]
        return labels.count(labels[0]) == len(labels)

    def build_tree(self):
        root = Node()

    def build(self, current_node):
        current_node.feature = max(
            [(self.get_information_gain(i, current_node.data), i) for i in range(0, len(current_node.data[0] - 1))],
            key=lambda x: x[1])

        left_node_data = []
        right_node_data = []
        for row in current_node.data:
            if row[current_node.feature] == 0:
                left_node_data.append(row)
            else:
                right_node_data.append(row)

        left_node = Node(data=left_node_data)
        right_node = Node(data=right_node_data)

        for node in [left_node, right_node]:
            node.is_leaf = self.is_leaf(node)
            if node.is_leaf:
                node.value = node.data[0][-1]
            else:
                self.build(node)

        current_node.left = left_node
        current_node.right = right_node
