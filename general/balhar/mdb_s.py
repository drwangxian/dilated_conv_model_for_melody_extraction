# average - 0.6308317795130035

import os
import json
import glob
import numpy as np
import librosa
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

    return dict(
        training=split_dict['train'],
        validation=split_dict['validation'],
        test=split_dict['test']
    )


def get_mdb_synth_test_split_fn():

    tvt_split_dict = get_mdb_test_split_fn()
    mdb_track_ids = tvt_split_dict['training'] + tvt_split_dict['validation'] + tvt_split_dict['test']
    assert len(mdb_track_ids) == 108
    wav_files = os.path.join(os.environ['mdb_melody_synth'], 'audio_mix', '*.wav')
    wav_files = glob.glob(wav_files)
    assert len(wav_files) == 65
    suffix = '_MIX_melsynth.wav'
    len_suffix = len(suffix)
    mdb_synth_track_ids = [os.path.basename(wav_file)[:-len_suffix] for wav_file in wav_files]
    assert len(mdb_synth_track_ids) == 65

    for track_id in mdb_synth_track_ids:
        assert track_id in mdb_track_ids

    mdb_test_track_ids = tvt_split_dict['test']
    mdb_synth_test_track_ids = []
    for track_id in mdb_synth_track_ids:
        if track_id in mdb_test_track_ids:
            mdb_synth_test_track_ids.append(track_id)
    assert len(mdb_synth_test_track_ids) == 13

    return mdb_synth_test_track_ids


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

    freqs = times_labels[:, 1]
    validity_check_of_ref_freqs_fn(freqs)

    return dict(times=times_labels[:, 0], freqs=freqs)


def read_est_fn(track_id):

    csv_dir = '/media/hd/datasets/jiri_melody_output/melody-outputs/hcnn_ctx-f0-outputs/mdb_melody_synth-test-melody-outputs'
    csv_file = os.path.join(csv_dir, track_id + '_MIX_melsynth.csv')
    times_labels = np.genfromtxt(csv_file, delimiter=',')
    assert times_labels.dtype == np.float64
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    assert np.all(np.logical_not(np.isnan(times_labels)))

    return dict(times=times_labels[:, 0], freqs=times_labels[:, 1])


track_ids = get_mdb_synth_test_split_fn()
oas = []
for idx, track_id in enumerate(track_ids):
    t = get_ref_time_and_freq_fn(track_id)
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


