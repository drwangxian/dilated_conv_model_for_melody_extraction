"""
average - 0.682941286917
"""

import os
import json
import numpy as np
import librosa
import mir_eval
import glob


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


def get_mdb_synth_test_split_fn():

    wav_files = os.path.join(os.environ['mdb_melody_synth'], 'audio_mix', '*.wav')
    wav_files = glob.glob(wav_files)
    assert len(wav_files) == 65
    suffix = '_MIX_melsynth.wav'
    len_suffix = len(suffix)
    mdb_synth_track_ids = [os.path.basename(wav_file)[:-len_suffix] for wav_file in wav_files]
    assert len(mdb_synth_track_ids) == 65

    mdb_test_track_ids = get_mdb_test_split_fn()
    assert len(mdb_test_track_ids) == 26
    t = []
    for track_id in mdb_synth_track_ids:
        if track_id in mdb_test_track_ids:
            t.append(track_id)
    assert len(t) == 13

    return t


def validity_check_of_ref_freqs_fn(freqs):

    min_melody_freq = librosa.midi_to_hz(23.6)

    all_zeros = freqs == 0.
    all_positives = freqs > min_melody_freq
    all_valid = np.logical_or(all_zeros, all_positives)
    assert np.all(all_valid)


def get_ref_time_and_freq_fn(track_id):
    # reference hop size - 128 samples

    m2_file = os.path.join(os.environ['mdb_melody_synth'], 'annotation_melody', track_id + '*.csv')
    m2_file = glob.glob(m2_file)
    assert len(m2_file) == 1
    m2_file = m2_file[0]
    times_labels = np.genfromtxt(m2_file, delimiter=',')
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    assert np.all(np.logical_not(np.isnan(times_labels)))
    num_frames = len(times_labels)
    t = times_labels[-1, 0]
    t = t / (128. / 44100.)
    t = int(round(t))
    assert t + 1 == num_frames
    assert times_labels[0, 0] == 0.
    validity_check_of_ref_freqs_fn(times_labels[:, 1])

    return dict(times=times_labels[:, 0], freqs=times_labels[:, 1])


def get_est_time_and_freq_fn(track_id):
    # the est melody uses a hop size of 512 samples
    annot_dir = os.path.join(os.environ['melody_outputs'], os.environ['barasan'], 'mdb_melody_synth-basaran-melody-outputs')

    annot_path = os.path.join(annot_dir, track_id + '_MIX_melsynth.csv')
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

track_ids = get_mdb_synth_test_split_fn()
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






