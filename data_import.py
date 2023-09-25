import os

def read_file_from_name(filename):
    full_filename = os.path.join(os.getcwd(), 'Data', filename)
    if not os.path.isfile(full_filename): return None
    data = []
    with open(full_filename, 'r') as fname:
        for line in fname:
            data.append(list(map(float, line.strip().split(' '))))
    
    return data

if __name__ == '__main__':
    print(read_file_from_name('Druns.txt'))
