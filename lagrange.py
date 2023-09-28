import data_import
import random
import math
import matplotlib.pyplot as plt

def sample_with_error(sd, N):
    samples = []
    for _ in range(N):
        samples.append([random.uniform(0, 4), random.gauss(0, sd)])
    
    return samples

class Lagrange:

    def __init__(self, x_i, y_i):
        self.x_i = x_i
        self.y_i = y_i
    
    # No training. Just predicting during testing
    def get_basis(self, i, x):
        result = 1
        for m in range(len(self.x_i)):
            if m == i: continue
            result *= (x - self.x_i[m]) / (self.x_i[i] - self.x_i[m])
        
        return result
    
    def evaluate(self, x):
        result = 0
        for i in range(len(self.y_i)):
            result += self.y_i[i] * self.get_basis(i, x)
        
        return result

if __name__ == '__main__':

    sds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3]
    test_data = data_import.read_file_from_name('Dsin_test.txt')
    for sd in sds:
        print('Standard Deviation -', sd)
        training_data = data_import.read_file_from_name('Dsin_'+str(sd)+'_error.txt')

        train_x = [x+e for x, e, y in training_data]
        train_y = [y for x, e, y in training_data]

        plt.scatter(train_x, train_y)
        plt.show()

        L = Lagrange(train_x, train_y)

        test_error = 0
        for i in range(len(test_data)):
            x, y = test_data[i]
            y_p = L.evaluate(x)
            test_error += (y_p - y) ** 2
        
        print("Test Error is", test_error/len(test_data))
        print("log-MSE Test Error is", math.log10(test_error/len(test_data)))

        training_error = 0
        for i in range(len(training_data)):
            x, e, y = training_data[i]
            y_p = L.evaluate(x+e)
            training_error += (y_p - y) ** 2
        
        print("Training Error is", training_error/len(training_data))


    # training_data = data_import.read_file_from_name('Dsin.txt')
    # test_data = data_import.read_file_from_name('Dsin_test.txt')

    # train_x = [x for x, y in training_data]
    # train_y = [y for x, y in training_data]
    
    # L = Lagrange(train_x, train_y)

    # test_error = 0
    # for i in range(len(test_data)):
    #     x, y = test_data[i]
    #     y_p = L.evaluate(x)
    #     test_error += (y_p - y)**2 # Using Mean-Squared Error
    
    # print("Test Error is", test_error/len(test_data))

    # training_error = 0
    # for i in range(len(training_data)):
    #     x, y = training_data[i]
    #     y_p = L.evaluate(x)
    #     training_error += (y_p - y)**2 # Using Mean-Squared Error
    
    # print(training_error/len(training_data))