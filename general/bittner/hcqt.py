import librosa
import os
import numpy as np
import functools


def hcqt_fn(wav_file):
    fmin = librosa.midi_to_hz(23.6)
    B = 60
    h = 512
    sr = 44100

    y, _sr = librosa.load(wav_file, sr=None)
    assert _sr == sr

    cqt_fn = functools.partial(librosa.cqt, y=y, sr=sr, hop_length=h, bins_per_octave=B)

    cqt = cqt_fn(fmin=fmin, n_bins=480)
    num_frames = cqt.shape[1]
    assert cqt.shape == (480, num_frames) and cqt.dtype == np.complex64
    cqt = np.abs(cqt.T)
    h1 = cqt[:, :360]
    h2 = cqt[:, 60:420]
    h4 = cqt[:, 120:]
    h_sub = cqt[:, :300]
    h_sub = np.pad(h_sub, [[0, 0], [60, 0]])
    assert h_sub.shape == (num_frames, 360)

    h3 = cqt_fn(fmin=fmin * 3., n_bins=360)
    h3 = np.abs(h3.T)
    assert h3.shape == (num_frames, 360)

    h5 = cqt_fn(fmin=fmin * 5., n_bins=360)
    h5 = np.abs(h5.T)
    assert h5.shape == (num_frames, 360)

    hs = np.stack([h_sub, h1, h2, h3, h4, h5], axis=-1)
    assert hs.shape == (num_frames, 360, 6)

    _num_frames = (len(y) + h - 1) // h
    assert num_frames == _num_frames or num_frames == _num_frames + 1
    num_frames = min(num_frames, _num_frames)
    hs = hs[:num_frames]
    hs = 1. / 80. * librosa.amplitude_to_db(hs, ref=np.max, top_db=80.) + 1.
    hs = np.require(hs, dtype=np.float32, requirements=['O', 'C'])

    return hs


if __name__ == '__main__':

    wav_file = os.environ['wav_file_short']
    hcqt = hcqt_fn(wav_file)

    print('done')





