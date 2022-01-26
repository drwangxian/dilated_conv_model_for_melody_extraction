# average - 0.7041878947641577

import glob
import os
import librosa
import numpy as np
import mir_eval


def get_mirex05_track_ids_fn():
    wav_files = glob.glob(os.path.join(os.environ['mirex05'], '*.wav'))
    track_ids = [os.path.basename(wav_file)[:-4] for wav_file in wav_files]
    assert len(track_ids) == 13

    return track_ids


def validity_check_of_ref_freqs_fn(freqs):

    min_melody_freq = librosa.midi_to_hz(23.6)

    all_zeros = freqs == 0.
    all_positives = freqs > min_melody_freq
    all_valid = np.logical_or(all_zeros, all_positives)
    assert np.all(all_valid)


def read_ref_fn(track_id):
    # reference melody uses a hop size of 441 samples

    if track_id == 'train13MIDI':
        m2_file = os.path.join(os.environ['mirex05'], 'train13REF.txt')
    else:
        m2_file = os.path.join(os.environ['mirex05'], track_id + 'REF.txt')
    times_labels = np.genfromtxt(m2_file, delimiter=None)
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    assert np.all(np.logical_not(np.isnan(times_labels)))
    num_frames = len(times_labels)
    t = times_labels[-1, 0]
    t = int(round(t / .01))
    assert t + 1 == num_frames
    assert times_labels[0, 0] == 0.
    freqs = times_labels[:, 1]

    validity_check_of_ref_freqs_fn(freqs)

    return dict(times=times_labels[:, 0], freqs=freqs)


def read_est_fn(track_id):

    if track_id == 'train13MIDI':
        track_id = 'train13'

    csv_dir = '/media/hd/datasets/jiri_melody_output/melody-outputs/hcnn_ctx-f0-outputs/mirex05-test-melody-outputs'
    csv_file = os.path.join(csv_dir, track_id + '.csv')
    times_labels = np.genfromtxt(csv_file, delimiter=',')
    assert times_labels.dtype == np.float64
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    assert np.all(np.logical_not(np.isnan(times_labels)))

    return dict(times=times_labels[:, 0], freqs=times_labels[:, 1])


track_ids = get_mirex05_track_ids_fn()
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