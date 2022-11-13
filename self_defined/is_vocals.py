import logging
logging.basicConfig(
    level=logging.WARNING
)
import medleydb as mdb
import numpy as np
import os


def assert_close_fn(t0, t1):

    t0 = np.round(t0, 6)
    t1 = np.round(t1, 6)

    assert t0 == t1


def is_vocals_m2m3_fn(track_name):

    return track_name


def is_vocals_singer_fn(track_id):

    return track_id


if __name__ == '__main__':

    track_names = mdb.TRACK_LIST_V1
    for idx, track_name in enumerate(track_names):
        print(idx)
        is_vocals = is_vocals_m2m3_fn(track_name)

