"""
re-run salamon's algorithm
average - 0.225620996348
"""

import os
import numpy as np
import soundfile
import librosa
import mir_eval
import essentia
import essentia.standard as es
import glob


def get_orchset_track_ids_fn():
    wav_files = glob.glob(os.path.join(os.environ['orchset'], 'audio', 'mono', '*.wav'))
    assert len(wav_files) == 64
    track_ids = [os.path.basename(wav_file)[:-4] for wav_file in wav_files]
    assert len(track_ids) == 64

    return track_ids


def validity_check_of_ref_freqs_fn(freqs):

    min_melody_freq = librosa.midi_to_hz(23.6)

    all_zeros = freqs == 0.
    all_positives = freqs > min_melody_freq
    all_valid = np.logical_or(all_zeros, all_positives)
    assert np.all(all_valid)


def get_ref_time_and_freq_fn(track_id):
    # reference hop size - 441 samples

    m2_file = os.path.join(os.environ['orchset'], 'GT', track_id + '.mel')
    times_labels = np.genfromtxt(m2_file, delimiter=None)
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    num_frames = len(times_labels)
    t = times_labels[-1, 0]
    t = int(round(t / .01))
    assert t + 1 == num_frames
    assert times_labels[0, 0] == 0.

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

        wav_file = os.path.join(os.environ['orchset'], 'audio', 'mono', track_id + '.wav')
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
        num_frames = len(freqs)
        times = np.arange(num_frames) * (128. / 44100.)

        return dict(times=times, freqs=freqs)


est_melody_ins = EstMelody()
melody_fn = est_melody_ins.melody_fn

track_ids = get_orchset_track_ids_fn()
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






