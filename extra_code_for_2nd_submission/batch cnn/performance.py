"""
performance of the patch-cnn model on RWC popular collection
ave - 0.537757020309
"""

import os
import glob
import numpy as np
import soundfile
import mir_eval

DEBUG = False


def get_rec_files_fn():

    dir_prefix = 'RWC-MDB-P-2001-M0'
    dir_path = os.path.join(os.environ['rwc'], 'popular', dir_prefix)
    num_recordings = []
    for dir_idx in range(7):
        dir_idx = dir_idx + 1
        disk_dir = dir_path + str(dir_idx)
        aiff_files = glob.glob(os.path.join(disk_dir, '*.aiff'))
        num_recordings.append(len(aiff_files))
    start_end_list = np.cumsum(num_recordings)
    assert start_end_list[-1] == 100
    start_end_list = np.pad(start_end_list, [[1, 0]], mode='constant')

    rec_files = []
    for rec_idx in range(100):
        disk_idx = np.searchsorted(start_end_list, rec_idx, side='right')
        assert disk_idx >= 1
        disk_idx = disk_idx - 1
        disk_path = dir_path + str(disk_idx + 1)
        disk_start_rec_idx = start_end_list[disk_idx]
        rec_idx_within_disk = rec_idx - disk_start_rec_idx
        rec_idx_within_disk = rec_idx_within_disk + 1

        recs = glob.glob(os.path.join(disk_path, '*.aiff'))
        assert len(recs) == num_recordings[disk_idx]
        for recording_name in recs:
            t = os.path.basename(recording_name)
            t = t.split()[0]
            if t == str(rec_idx_within_disk):
                rec_files.append(recording_name)
                break
        else:
            assert False
    assert len(rec_files) == 100
    t = set(rec_files)
    assert len(t) == 100

    return rec_files


def get_num_frames_fn(aiff_file):

    aiff_info = soundfile.info(aiff_file)
    assert aiff_info.samplerate == 44100
    num_samples = aiff_info.frames
    h = 441
    num_frames = (num_samples + h - 1) // h

    return num_frames


def load_melody_from_file_fn(rec_idx, aiff_file):

    melody_dir = os.path.join(os.environ['rwc'], 'popular', 'AIST.RWC-MDB-P-2001.MELODY')
    melody_prefix = 'RM-P'
    melody_suffix = '.MELODY.TXT'
    melody_file = melody_prefix + str(rec_idx + 1).zfill(3) + melody_suffix
    melody_file = os.path.join(melody_dir, melody_file)

    with open(melody_file, 'r') as fh:
        lines = fh.readlines()
        line = lines[-1]
        cols = line.split()
        num_frames = int(cols[0]) + 1
        aiff_num_frames = get_num_frames_fn(aiff_file)
        assert num_frames <= aiff_num_frames
        freqs = np.zeros([aiff_num_frames], np.float32)
        min_freq = 31.
        for line in lines:
            cols = line.split()
            assert len(cols) == 5
            assert cols[0] == cols[1]
            assert cols[2] == 'm'
            frame_idx = int(cols[0])
            assert frame_idx >= 0
            freq = float(cols[3])
            assert freq == 0 or freq > min_freq
            freqs[frame_idx] = freq

        return freqs


def gen_ref_label_fn(rec_idx, aiff_file):

    freqs_441 = load_melody_from_file_fn(rec_idx, aiff_file)
    num_frames_441 = len(freqs_441)
    times_441 = np.arange(num_frames_441) * 0.01

    return times_441, freqs_441


def read_pred_label_fn(rec_idx):

    melody_file = os.path.join('melodies', str(rec_idx) + '.txt')

    times_freqs = np.genfromtxt(melody_file)

    return times_freqs[:, 0], times_freqs[:, 1]


if __name__ == '__main__':

    rec_files = get_rec_files_fn()
    assert len(rec_files) == 100

    if DEBUG:
        rec_files = rec_files[:3]

    oas = []
    for rec_idx, aiff_file in enumerate(rec_files):
        ref_times, ref_freqs = gen_ref_label_fn(rec_idx, aiff_file)
        est_times, est_freqs = read_pred_label_fn(rec_idx)
        oa = mir_eval.melody.evaluate(
            ref_time=ref_times,
            ref_freq=ref_freqs,
            est_time=est_times,
            est_freq=est_freqs
        )['Overall Accuracy']
        oas.append(oa)

        print('{} - {}'.format(rec_idx, oa))
    oa = np.mean(oas)
    print('ave - {}'.format(oa))



