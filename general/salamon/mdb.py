"""
re-run salamon's algorithm
average - 0.504080283409
"""

import os
import json
import numpy as np
import soundfile
import librosa
import mir_eval
import essentia
import essentia.standard as es


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

    freqs = times_labels[:, 1]
    validity_check_of_ref_freqs_fn(freqs)

    return dict(times=times_labels[:, 0], freqs=times_labels[:, 1])


class EstMelody:

    def __init__(self):

        fn = es.EqualLoudness()
        self.equal_loudness_fn = fn

        fn = es.PredominantPitchMelodia()
        self.es_melody_fn = fn

    @staticmethod
    def load_wav_fn(track_id):

        sr = 44100

        wav_file = os.path.join(os.environ['medleydb'], track_id, track_id + '_MIX.wav')
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
        # melodia uses an optimal hop size of 128, so need to downsample by 2
        freqs = freqs[::2]
        times = times[::2]

        return dict(times=times, freqs=freqs)


track_ids = get_mdb_test_split_fn()
assert len(track_ids) == 26
oas = []
est_melody_ins = EstMelody()
melody_fn = est_melody_ins.melody_fn
for idx, track_id in enumerate(track_ids):
    t = read_ref_fn(track_id)
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







