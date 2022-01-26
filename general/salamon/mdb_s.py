"""
re-run salamon's algorithm
average - 0.491707614691
"""

import os
import json
import numpy as np
import soundfile
import librosa
import mir_eval
import essentia
import essentia.standard as es
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

    mdb_test_track_ids = get_mdb_test_split_fn()

    jiri_mdb_synth_test_ids = os.path.join(
        '/media/hd/datasets/jiri_melody_output/melody-outputs/hcnn_ctx-f0-outputs/mdb_melody_synth-test-melody-outputs',
        '*.csv'
    )
    suffix = '_MIX_melsynth.csv'
    suffix_len = len(suffix)
    jiri_mdb_synth_test_ids = glob.glob(jiri_mdb_synth_test_ids)
    jiri_mdb_synth_test_ids = [os.path.basename(tid)[:-suffix_len] for tid in jiri_mdb_synth_test_ids]
    jiri_mdb_synth_test_ids = list(set(jiri_mdb_synth_test_ids))
    assert len(jiri_mdb_synth_test_ids) == 22

    mdb_synth_test_track_ids = []
    for track_id in jiri_mdb_synth_test_ids:
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

    if track_id == 'MusicDelta_Rock':
        m2_file = os.path.join(os.environ['mdb_melody_synth'], 'annotation_melody', 'MusicDelta_Rock_STEM_05.RESYN.csv')
    else:
        m2_file = os.path.join(os.environ['mdb_melody_synth'], 'annotation_melody', track_id + '*.csv')
        m2_file = glob.glob(m2_file)
        assert len(m2_file) == 1
        m2_file = m2_file[0]
    times_labels = np.genfromtxt(m2_file, delimiter=',')
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    num_frames = len(times_labels)
    t = times_labels[-1, 0]
    t = t / (128. / 44100.)
    t = int(round(t))
    assert t + 1 == num_frames
    assert times_labels[0, 0] == 0.

    freqs = times_labels[:, 1]
    validity_check_of_ref_freqs_fn(freqs)

    return dict(times=times_labels[:, 0], freqs=freqs)


class EstMelody:

    def __init__(self):

        fn = es.EqualLoudness()
        self.equal_loudness_fn = fn

        fn = es.PredominantPitchMelodia()
        self.es_melody_fn = fn

    @staticmethod
    def load_wav_fn(track_id):

        sr = 44100
        suffix = '_MIX_melsynth.wav'

        wav_file = os.path.join(os.environ['mdb_melody_synth'], 'audio_mix', track_id + suffix)
        samples, _sr = soundfile.read(wav_file, dtype='int16')
        assert _sr == sr
        assert samples.dtype == np.int16
        assert samples.ndim == 2 and samples.shape[1] == 2
        samples = samples.astype(np.float32)
        samples = np.mean(samples, axis=1)
        samples = samples / 32768.
        assert samples.dtype == np.float32

        return samples

    def melody_fn(self, track_id):

        outputs = EstMelody.load_wav_fn(track_id)
        for fn in (self.equal_loudness_fn, self.es_melody_fn):
            fn.reset()
            outputs = fn(outputs)
        freqs = outputs[0]
        num_frames = len(freqs)
        times = np.arange(num_frames) * (128. / 44100.)

        return dict(times=times, freqs=freqs)


est_melody_ins = EstMelody()
melody_fn = est_melody_ins.melody_fn

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






