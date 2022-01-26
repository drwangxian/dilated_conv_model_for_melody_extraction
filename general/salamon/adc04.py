"""
re-run salamon's algorithm
average - 0.690281608516
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
    # the reference melody uses a hop size of 256 samples, but the numerical precision for times is insufficient

    melody2_suffix = 'REF.txt'
    annot_path = os.path.join(os.environ['adc04'], track_id + melody2_suffix)
    times_labels = np.genfromtxt(annot_path, delimiter=None)
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    num_frames = len(times_labels)
    t = times_labels[-1, 0]
    t = int(round(t / (256. / 44100.)))
    assert t + 1 == num_frames
    assert times_labels[0, 0] == 0.

    freqs = times_labels[::, 1]
    validity_check_of_ref_freqs_fn(freqs)

    return dict(times=times_labels[:, 0], freqs=freqs)


class EstMelody:

    def __init__(self):

        fn = es.EqualLoudness()
        self.equal_loudness_fn = fn

        fn = es.PredominantPitchMelodia()  # the default hop size is 128
        self.es_melody_fn = fn

    @staticmethod
    def load_wav_fn(track_id):

        sr = 44100

        wav_file = os.path.join(os.environ['adc04'], track_id + '.wav')
        samples, _sr = soundfile.read(wav_file, dtype='int16')
        assert _sr == sr
        assert samples.dtype == np.int16
        assert samples.ndim == 1
        samples = samples.astype(np.float32)
        samples = samples / 32768.
        assert samples.dtype == np.float32

        return samples

    def melody_fn(self, track_id):

        outputs = EstMelody.load_wav_fn(track_id)
        for fn in (self.equal_loudness_fn, self.es_melody_fn):
            fn.reset()
            outputs = fn(outputs)
        freqs = outputs[0]
        freqs = freqs[::2]
        num_frames = len(freqs)
        times = np.arange(num_frames) * (256. / 44100.)

        return dict(times=times, freqs=freqs)


est_melody_ins = EstMelody()
melody_fn = est_melody_ins.melody_fn

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






