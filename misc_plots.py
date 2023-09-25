import matplotlib.pyplot as plt

n = [32, 128, 512, 2048, 8192]
#nodes = [9, 17, 49, 139, 309]
err_n = [0.10674, 0.11559, 0.04480, 0.03097, 0.00940]

#plt.plot(n, nodes, label='# of nodes', color='yellow')
plt.plot(n, err_n, label='err_n', color='green')
#plt.legend()
plt.xlabel('n')
plt.ylabel('err_n')
plt.title('Learning Curve sklearn')
plt.show()