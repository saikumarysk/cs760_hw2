import matplotlib.pyplot as plt

n = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3]
#nodes = [9, 17, 49, 139, 309]
err_n = [34.63582, 61.81052, 40.32474, 37.23692, 51.86019, 30.78964, 30.3708, 17.59629, -3.20988, -5.9214, -4.86076]

#plt.plot(n, nodes, label='# of nodes', color='yellow')
plt.plot(n, err_n, label='Testing Error (log-MSE)', marker='.', color='green')
#plt.legend()
plt.xlabel('Standard Deviation')
plt.ylabel('log-MSE')
plt.title('Standard Deviation vs. log-MSE Testing Error')
plt.show()