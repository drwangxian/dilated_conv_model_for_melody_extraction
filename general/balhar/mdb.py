# average - 0.6444291349548787

import os
import json
import librosa
import numpy as np
import mir_eval


def get_mdb_test_split_fn():
    split_file = os.environ['medleydb']
    split_file = os.path.join(split_file, '..', 'V1_auxiliary', 'data_splits_jiri.json')
    with open(split_file, 'rb') as fh:
        split_dict = json.load(fh)
    assert len(split_dict['test']) == 27
    assert len(split_dict['train']) == 67
    assert len(split_dict['validation']) == 15

    intersection_track_id = 'MatthewEntwistle_DontYouEver'
    assert intersection_track_id in split_dict['train']
    assert intersection_track_id in split_dict['test']
    split_dict['test'].remove(intersection_track_id)
    assert len(split_dict['test']) == 26

    return split_dict['test']


def validity_check_of_ref_freqs_fn(freqs):

    min_melody_freq = librosa.midi_to_hz(23.6)

    all_zeros = freqs == 0.
    all_positives = freqs > min_melody_freq
    all_valid = np.logical_or(all_zeros, all_positives)
    assert np.all(all_valid)


def read_ref_fn(track_id):

    melody2_dir = os.environ['melody2_dir']
    melody2_suffix = '_MELODY2.csv'

    annot_path = os.path.join(melody2_dir, track_id + melody2_suffix)
    times_labels = np.genfromtxt(annot_path, delimiter=',')
    assert times_labels.dtype == np.float64
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    assert np.all(np.logical_not(np.isnan(times_labels)))
    num_frames = len(times_labels)
    tmp = np.arange(num_frames) * (256. / 44100.)
    assert np.all(tmp == times_labels[:, 0])

    freqs = times_labels[:, 1]
    validity_check_of_ref_freqs_fn(freqs)

    return dict(times=times_labels[:, 0], freqs=times_labels[:, 1])


def read_est_fn(track_id):

    csv_dir = '/media/hd/datasets/jiri_melody_output/melody-outputs/hcnn_ctx-f0-outputs/mdb-test-melody-outputs'
    csv_file = os.path.join(csv_dir, track_id + '_MIX.csv')
    times_labels = np.genfromtxt(csv_file, delimiter=',')
    assert times_labels.dtype == np.float64
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    assert np.all(np.logical_not(np.isnan(times_labels)))

    return dict(times=times_labels[:, 0], freqs=times_labels[:, 1])


track_ids = get_mdb_test_split_fn()
oas = []
for idx, track_id in enumerate(track_ids):
    t = read_ref_fn(track_id)
    ref_times, ref_freqs = t['times'], t['freqs']

    t = read_est_fn(track_id)
    est_times, est_freqs = t['times'], t['freqs']

    oa = mir_eval.melody.evaluate(
        ref_time=ref_times,
        ref_freq=ref_freqs,
        est_time=est_times,
        est_freq=est_freqs
    )['Overall Accuracy']
    oas.append(oa)
    print(idx, track_id, oa)
oa = np.mean(oas)
print('average - {}'.format(oa))