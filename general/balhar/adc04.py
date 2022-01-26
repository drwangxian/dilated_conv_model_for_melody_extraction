"""
average - 0.7455114726075931
"""

import os
import numpy as np
import mir_eval
import glob


def get_adc04_track_ids_fn():

    wav_files = glob.glob(os.path.join(os.environ['adc04'], '*.wav'))
    track_ids = [os.path.basename(wav_file)[:-4] for wav_file in wav_files]
    assert len(track_ids) == 20

    return track_ids


def midi_to_hz(midi):

    hz = 440. * 2 ** ((midi - 69.) / 12.)

    return hz


def validity_check_of_ref_freqs_fn(freqs):

    min_melody_freq = midi_to_hz(23.6)

    all_zeros = freqs == 0.
    all_positives = freqs > min_melody_freq
    all_valid = np.logical_or(all_zeros, all_positives)
    assert np.all(all_valid)


def gen_ref_fn(track_id):
    # the reference melody uses a hop size of 256 samples
    melody2_suffix = 'REF.txt'

    annot_path = os.path.join(os.environ['adc04'], track_id + melody2_suffix)
    times_freqs = np.genfromtxt(annot_path, delimiter=None)
    assert times_freqs.ndim == 2 and times_freqs.shape[1] == 2
    assert np.all(np.logical_not(np.isnan(times_freqs)))
    validity_check_of_ref_freqs_fn(times_freqs[:, 1])

    return dict(times=times_freqs[:, 0], freqs=times_freqs[:, 1])


def read_est_fn(track_id):
    csv_file = '/media/hd/datasets/jiri_melody_output/melody-outputs/hcnn_ctx-f0-outputs/adc04-test-melody-outputs'
    csv_file = os.path.join(csv_file, track_id + '.csv')
    times_labels = np.genfromtxt(csv_file, delimiter=',')
    assert times_labels.dtype == np.float64
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    assert np.all(np.logical_not(np.isnan(times_labels)))

    return dict(times=times_labels[:, 0], freqs=times_labels[:, 1])


melody_fn = read_est_fn

track_ids = get_adc04_track_ids_fn()
oas = []
for idx, track_id in enumerate(track_ids):
    t = gen_ref_fn(track_id)
    ref_times, ref_freqs = t['times'], t['freqs']

    t = melody_fn(track_id)
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