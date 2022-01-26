"""
average - 0.668764881975
"""

import os
import numpy as np
import librosa
import mir_eval
import glob


def get_adc04_track_ids_fn():

    wav_files = glob.glob(os.path.join(os.environ['adc04'], '*.wav'))
    track_ids = [os.path.basename(wav_file)[:-4] for wav_file in wav_files]
    assert len(track_ids) == 20

    return track_ids


def validity_check_of_ref_freqs_fn(freqs):

    min_melody_freq = librosa.midi_to_hz(23.6)

    all_zeros = freqs == 0.
    all_positives = freqs > min_melody_freq
    all_valid = np.logical_or(all_zeros, all_positives)
    assert np.all(all_valid)


def get_ref_time_and_freq_fn(track_id):
    # the reference melody uses a hop size of 256 samples

    melody2_suffix = 'REF.txt'

    annot_path = os.path.join(os.environ['adc04'], track_id + melody2_suffix)
    times_labels = np.genfromtxt(annot_path, delimiter=None)
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    assert np.all(np.logical_not(np.isnan(times_labels)))
    num_frames = len(times_labels)
    t = times_labels[-1, 0]
    t = int(round(t / (256. / 44100.)))
    assert t + 1 == num_frames
    assert times_labels[0, 0] == 0.
    validity_check_of_ref_freqs_fn(times_labels[:, 1])

    return dict(times=times_labels[:, 0], freqs=times_labels[:, 1])


def get_est_time_and_freq_fn(track_id):
    # the est melody uses a hop size of 512 samples
    annot_dir = os.path.join(os.environ['melody_outputs'], os.environ['barasan'], 'adc04-basaran-melody-outputs')

    annot_path = os.path.join(annot_dir, track_id + '.csv')
    times_labels = np.genfromtxt(annot_path, delimiter=',')
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    assert np.all(np.logical_not(np.isnan(times_labels)))
    num_frames = len(times_labels)
    t = times_labels[-1, 0]
    t = int(round(t / (512. / 44100.)))
    assert t + 1 == num_frames
    assert times_labels[0, 0] == 0.

    return dict(times=times_labels[:, 0], freqs=times_labels[:, 1])


melody_fn = get_est_time_and_freq_fn

track_ids = get_adc04_track_ids_fn()
oas = []
for idx, track_id in enumerate(track_ids):
    t = get_ref_time_and_freq_fn(track_id)
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






