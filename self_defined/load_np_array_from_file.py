import numpy as np


def load_np_array_from_file_fn(file_name):

    with open(file_name, 'rb') as fh:
        first_line = fh.readline().decode('utf-8')
        first_line = first_line.split()
        rec_name = first_line[0]
        dtype = first_line[1]
        dim = first_line[2:]
        dim = [int(_item) for _item in dim]
        output = np.frombuffer(fh.read(), dtype=dtype).reshape(*dim)

        return rec_name, output