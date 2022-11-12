import os
import numpy as np


def save_np_array_to_file_fn(file_name, output, rec_name):
    assert isinstance(rec_name, str)
    assert ' ' not in rec_name
    with open(file_name, 'wb') as fh:
        fh.write(rec_name.encode('utf-8'))
        fh.write(b' ')
        dtype = '{}'.format(output.dtype).encode('utf-8')
        fh.write(dtype)
        for dim_size in output.shape:
            fh.write(b' ')
            fh.write('{:d}'.format(dim_size).encode('utf-8'))
        fh.write(b'\n')
        fh.write(output.data)
        fh.flush()
        os.fsync(fh.fileno())
