import matplotlib.pyplot as plt
import data_import

def scatter(filename):
    data = data_import.read_file_from_name(filename)
    red = [(x, y) for x, y, z in data if z == 1]
    blue = [(x, y) for x, y, z in data if z == 0]
    red_x = [x for x, y in red]
    red_y = [y for x, y in red]
    blue_x = [x for x, y in blue]
    blue_y = [y for x, y in blue]

    plt.scatter(red_x, red_y, label="Y = 1", color='red')
    plt.scatter(blue_x, blue_y, label="Y = 0", color='blue')
    plt.title("Scatter Plot for for "+filename)

if __name__ == '__main__':
    scatter('Dxor.txt')
    plt.show()
