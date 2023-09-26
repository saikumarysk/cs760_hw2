import random
import math
import data_import
import entropy
import scatter
import matplotlib.pyplot as plt
import numpy as np

class TreeNode:
    def __init__(self, num_split):
        self.left = None
        self.right = None
        self.num_split = num_split
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return self._repr_prefix(self, '')

    def _repr_prefix(self, node, prefix):
        if node == None:
            return prefix + '- None\n'
        else:
            return prefix + '-' + str(node.num_split) + '\n' + self._repr_prefix(node.left, prefix + ' |') + self._repr_prefix(node.right, prefix + '  ')

class DTree:

    num_nodes = 0

    def __init__(self, data):
        self.data = data
        self.root = self.make_subtree(data)

    def make_subtree(self, d):
        if len(d) == 0 :
            maj_label = self.find_majority_label(d)
            self.num_nodes += 1 # Counting leaf nodes too.
            return TreeNode((maj_label,)) # Leaf node has a tuple of length 1
        
        candidate_splits = self.determine_candidate_splits(d, 0) + self.determine_candidate_splits(d, 1)
        best_split = self.find_best_split(d, candidate_splits)

        # Stopping Criterion
        if ( len(candidate_splits) == 0 ) or ( best_split == None ) :
            maj_label = self.find_majority_label(d)
            self.num_nodes += 1
            return TreeNode((maj_label,)) # Leaf node has a tuple of length 1

        node = TreeNode(best_split)
        self.num_nodes += 1
        node.left = self.make_subtree(self.find_then_branch_data(d, best_split))
        node.right = self.make_subtree(self.find_else_branch_data(d, best_split))

        return node

    
    def determine_candidate_splits(self, d, j):
        candidate_splits = []
        data_copy = sorted(d, key = lambda x: x[j])
        for i in range(len(data_copy)-1):
            if data_copy[i][-1] != data_copy[i+1][-1]:
                c = data_copy[i+1][j]
                ent = self.entropy_of_split(d, (j, c))
                if ent == 0 : continue
                candidate_splits.append([(j, c), ent])
        
        return candidate_splits

    def find_then_branch_data(self, d, node_split):
        new_data = []
        j, c = node_split
        for i in range(len(d)):
            if d[i][j] >= c: new_data.append(d[i])
        
        return new_data

    def find_else_branch_data(self, d, node_split):
        new_data = []
        j, c = node_split
        for i in range(len(d)):
            if d[i][j] < c: new_data.append(d[i])
        
        return new_data

    def find_best_split(self, d, candidate_splits):
        if len(candidate_splits) == 0 : return None
        H_d_Y = self.entropy_of_data(d)
        new_cand_splits = []

        for cand_split in candidate_splits:
            j, c, H_d_S = cand_split[0][0], cand_split[0][1], cand_split[-1]

            yes_instances_s, no_instances_s = 0, 0
            yes_instances_then, no_instances_then = 0, 0
            yes_instances_else, no_instances_else = 0, 0

            for i in range(len(d)):
                if d[i][j] >= c: yes_instances_s += 1
                else: no_instances_s += 1

                if d[i][j] >= c and d[i][-1] == 1: yes_instances_then += 1
                elif d[i][j] >= c and d[i][-1] == 0: no_instances_then += 1

                if d[i][j] < c and d[i][-1] == 1: yes_instances_else += 1
                elif d[i][j] < c and d[i][-1] == 0: no_instances_else += 1
            
            H_d_Y_then = entropy.shannon_entropy([yes_instances_then/yes_instances_s, no_instances_then/yes_instances_s])
            H_d_Y_else = entropy.shannon_entropy([yes_instances_else/no_instances_s, no_instances_else/no_instances_s])

            H_d_Y_S = (yes_instances_s/len(d))*H_d_Y_then + (no_instances_s/len(d))*H_d_Y_else

            info_gain = H_d_Y - H_d_Y_S
            if info_gain == 0 : continue
            new_cand_splits.append([(j, c), round(info_gain/H_d_S, 5)])
        
        if len(new_cand_splits) == 0 : return None
        max_info_gain_ratio = max(new_cand_splits, key= lambda x: x[-1])[-1]
        best_splits = [cand_split for cand_split in new_cand_splits if cand_split[-1] == max_info_gain_ratio]

        return best_splits[-1][0]


    def entropy_of_data(self, d):
        yes_instances = sum([instance[-1] for instance in d])
        no_instances = len(d) - yes_instances

        return entropy.shannon_entropy([yes_instances/len(d), no_instances/len(d)])
    
    def entropy_of_split(self, d, cand_split):
        j, c = cand_split
        yes_instances, no_instances = 0, 0
        for i in range(len(d)):
            if d[i][j] >= c: yes_instances += 1
            else: no_instances += 1
        return entropy.shannon_entropy([yes_instances/len(d), no_instances/len(d)])
    
    def find_majority_label(self, d):
        yes_instances = sum([instance[-1] for instance in d])
        no_instances = len(d) - yes_instances
        return 1 if yes_instances >= no_instances else 0
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return repr(self.root)
    
    def evaluate(self, x):
        node = self.root
        return self._evaluate(node, x)
    def _evaluate(self, node, x):
        if len(node.num_split) == 1: return node.num_split[0]

        j, c = node.num_split
        if x[j] >= c: return self._evaluate(node.left, x)
        else: return self._evaluate(node.right, x)
    
    def decision_boundary(self, x1_min, x1_max, x2_min, x2_max):

        x1s, x2s = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
        x_in = np.c_[x1s.ravel(), x2s.ravel()]
        y_p = []
        for x in x_in:
            y_p.append(1 - self.evaluate(list(x))) # For color in plot, invert the prediction
        
        y_pred = np.round(y_p).reshape(x1s.shape)
        
        plt.contourf(x1s, x2s, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)

    
if __name__ == '__main__':
    filename = 'Dxor.txt'
    data = data_import.read_file_from_name(filename)
    # test_data = data_import.read_file_from_name('test_data.txt')

    root = DTree(data)
    print(root)
    #print(root.num_nodes)

    scatter.scatter(filename)
    #root.decision_boundary(-1.5,1.5,-1.5,1.5)
    plt.legend()
    plt.show()

    # error = 0
    # for x1, x2, y in test_data:
    #     y_p = root.evaluate([x1, x2])
    #     error += 1 if y_p != y else 0
    
    # print(root.num_nodes)
    # print(error/len(test_data))
