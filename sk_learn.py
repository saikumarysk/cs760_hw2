from sklearn import tree
import data_import

data = data_import.read_file_from_name('D8192.txt')
test_data = data_import.read_file_from_name('test_data.txt')
X = [[x_1, x_2] for x_1, x_2, y in data]
Y = [y for x_1, x_2, y in data]

clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)

test_X = [[x_1, x_2] for x_1, x_2, y in test_data]
test_Y = [y for x_1, x_2, y in test_data]

pred_Y = clf.predict(test_X)

error = 0
for i in range(len(test_Y)):
    if pred_Y[i] != test_Y[i]: error += 1

print(error/len(test_data), clf.tree_.node_count)