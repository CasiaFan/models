# -*- coding: utf-8 -*-
"""
Build clothing hierarchical caategories into a tree
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from treelib import Node, Tree
import pandas as pd
import numpy as np

# class Node():
#     """Node class"""
#     def __init__(self, val, children=[], parent=None):
#         self.value = val
#         self.children = children
#         self.parent = parent
#
#     def get_value(self):
#         return self.value
#
#     def set_value(self, value):
#         self.value = value
#
#     def get_children(self):
#         return self.children
#
#     def set_children(self, children):
#         self.children = children
#
#     def add_child(self, child):
#         self.children.append(child)
#
#     def get_parent(self):
#         return self.parent
#
#     def __repr__(self):
#         return "Tree.Node({},{})".format(self.value, ",".join(self.children))

class ClothingTreeFromFile():
    """
    construct a tree with node-leaf list
    label_file: label file containing hierarchical labels with leaf category integer index and followed by its higher nodes

     a list of list with path from node to leaf, eg:
        [ [0, node1-1, node2-1, leaf1],
          [1, node1-1, node2-1, leaf2],
        ...]


    """
    def __init__(self, label_file):
        self.label_file = label_file

    def mapping_name_to_id(self):
        self.node_leaf_list = np.asarray(pd.read_csv(self.label_file, sep=":", header=None))
        levels = self.node_leaf_list.shape[1] - 1
        name_to_id_dict = defaultdict(dict)
        for level in range(1, levels+1):
            class_level = "class"+str(level)
            name_to_id_dict[class_level] = defaultdict(dict)
            counter = 0
            for cate in self.node_leaf_list[:, level]:
                if not cate in name_to_id_dict[class_level]:
                    name_to_id_dict[class_level][cate] = counter
                    counter += 1
        print("Read in label file done! Labels name to id are:, ", name_to_id_dict)
        return name_to_id_dict


    def generate_tree(self):
        """
        return a treelib tree class with node name in each level
                   node1-1
                  /      \
             node2-1    node2-2
              /   \      /    \
         leaf1  leaf2 leaf3  leaf4
        """
        name_to_id_dict = self.mapping_name_to_id()
        tree = Tree()
        tree.create_node("ROOT", "ROOT")
        for node_leaf in self.node_leaf_list[:, 1:]:
            node_leaf = np.insert(node_leaf, 0, 'ROOT')
            for i in range(1, len(node_leaf)):
                if not tree.contains(node_leaf[i]):
                    tree.create_node(tag=node_leaf[i], identifier=node_leaf[i], parent=node_leaf[i-1], data=name_to_id_dict['class'+str(i)][node_leaf[i]])
        print("Construct tree done! Tree structure is:")
        print(tree.show())
        return tree


    def get_leaf_path(self, tree, leafname):
        """
        get path to given leaf name
        """
        paths_to_leaves = tree.paths_to_leaves()
        leaf_path = np.squeeze([x for x in paths_to_leaves if x[-1] == leafname])
        # exclude ROOT node
        return leaf_path[1:]


class TreeTools():
    """
    Hierarchical class tree operation tools class
    Reference: https://talbaumel.github.io/softmax/
    """
    def __init__(self):
        # get number of nodes in tree, not include the root node
        self._count_nodes_dict = {}

    def _get_subtrees(self, tree):
        # get all subtrees of given tree
        yield tree
        for subtree in tree:
            if type(subtree) == list:
                for x_tree in self._get_subtrees(subtree):
                    yield x_tree

    def _get_leaves_and_paths(self, tree):
        # get all leaves value and its corresponding path
        for i, subtree in enumerate(tree):
            if type(subtree) == list:
                for path, value in self._get_leaves_and_paths(subtree):
                    yield [i]+path, value
            else:
                yield [i], subtree

    def _get_nodes_count(self, tree):
        # storing node count of each subtree, use id(tree_object) as its key
        if id(tree) in self._count_nodes_dict:
            return self._count_nodes_dict[id(tree)]
        else:
            count = 0
            for node in tree:
                if type(node) == list:
                    count += 1 + self._get_nodes_count(node)
                    # print("current count of node {} is {}".format(node, count))
            self._count_nodes_dict[id(tree)] = count
            return count

    def _get_nodes(self, tree, path):
        # get nodes in a path
        next_node = 0  # root node
        nodes = []
        for decision in path:
            nodes.append(next_node)
            next_node += 1 + self._get_nodes_count(tree[:decision])
            tree = tree[decision]
        return nodes


if __name__ == "__main__":
    clothing_tree = ClothingTreeFromFile(label_file="/home/arkenstone/tensorflow/workspace/models/slim/models/deepfashion_l2/data/second_class_labels.txt")
    tree = clothing_tree.generate_tree()
    print(tree.get_node('Tee'))
    print([node for node in tree.leaves()])
    print(clothing_tree.get_leaf_path(tree, "Cardigan"))
    print(tree.all_nodes())

