"""
for Ping Gao: sr = 16000
"""
import numpy as np
import scipy.signal
import librosa
import tensorflow as tf
import os
import soundfile
import logging

tf.config.run_functions_eagerly(False)


def mac_loc_and_value(x):

    r_size, c_size = x.shape
    idx = x.argmax()
    r = idx // c_size
    c = idx % c_size
    assert x[r, c] == x.max()

    return (r, c), x[r, c]


@tf.function(
    input_signature=[tf.TensorSpec([None, None])], autograph=False
)
def tf_mac_loc_and_value(B):

    js = tf.argmax(B, axis=1, output_type=tf.int32)
    max_vals_of_rows = tf.gather_nd(B, js[:, None], batch_dims=1)
    i = tf.argmax(max_vals_of_rows, output_type=tf.int32)
    j = js[i]
    return (i, j), B[i, j]


class Config:

    def __init__(self):

        self.win_len = 768
        self.fr = 2
        self.sr = 8000

        fft_len = int(np.ceil(float(self.sr) / self.fr))
        if fft_len % 2 == 1:
            fft_len += 1
        self.fft_len = fft_len
        self.fr = float(self.sr) / self.fft_len
        assert self.fft_len >= self.win_len

        win_fun = scipy.signal.windows.blackmanharris(self.win_len, sym=False).astype(np.float32)
        win_fun = win_fun / np.linalg.norm(win_fun)
        self.win_fun = win_fun

        self.hop_size = 80

        self.gammas = [.24, .6, 1]
        self.fmin, self.fmax = 31, 1250
        self.bins_per_oct = 60
        self.central_freqs = self._gen_central_freqs_fn()
        assert len(self.central_freqs) == 321

        self.max_num_frames = 45 * self.sr // self.hop_size

    def _gen_central_freqs_fn(self):

        fmax = self.fmax
        fmin = self.fmin
        bins_per_oct = self.bins_per_oct
        central_freqs = []

        fac = 2. ** (1. / bins_per_oct)
        f = float(fmin)
        while f < fmax:
            central_freqs.append(f)
            f = f * fac

        return central_freqs


class CFP:

    def __init__(self):

        self.config = Config()

    def _gen_split_fn(self, total_num_frames):

        config = self.config

        step = config.max_num_frames
        t = np.arange(0, total_num_frames, step)
        assert t[-1] < total_num_frames
        t = np.append(t, total_num_frames)
        assert len(t) >= 2
        t = zip(t[:-1], t[1:])

        return t

    @tf.function(input_signature=[
        tf.TensorSpec([None]), tf.TensorSpec([])
    ], autograph=False)
    def _stft_tf_fn(self, samples, g):

        config = self.config

        hop = config.hop_size
        window_size = config.win_len
        win_fn = config.win_fun
        fft_len = config.fft_len
        assert fft_len % 2 == 0
        assert fft_len > window_size
        assert len(win_fn) == window_size

        samples = tf.convert_to_tensor(samples, tf.float32)
        samples.set_shape([None])
        g = tf.convert_to_tensor(g, tf.float32)
        num_samples = tf.size(samples, out_type=tf.int32)
        tf.debugging.assert_greater_equal(num_samples, window_size)
        tf.debugging.assert_equal((num_samples - window_size) % hop, 0)

        samples = tf.signal.frame(
            signal=samples,
            frame_length=window_size,
            frame_step=hop,
            pad_end=False
        )
        assert samples.dtype == tf.float32
        samples.set_shape([None, window_size])
        win_fn = tf.convert_to_tensor(win_fn, tf.float32)
        samples = samples * win_fn
        samples = tf.signal.rfft(samples, fft_length=[fft_len])
        assert samples.dtype == tf.complex64
        samples.set_shape([None, 1 + fft_len // 2])
        samples = tf.abs(samples)
        samples = tf.pow(samples, g)

        return samples

    def _coef_matrix_for_freq_2_log_freq_fn(self):

        # return shape: [int(fmax / fr) + 1, num_central_freq_bins - 1]

        config = self.config
        fmax = config.fmax
        central_freqs = config.central_freqs
        fr = config.fr
        fmax = float(fmax)
        fr = float(fr)

        HighFreqIdx = int(fmax / fr)
        freqs = np.arange(HighFreqIdx + 1) * fr
        assert freqs[-1] <= fmax
        num_central_freq_bins = len(central_freqs)
        freq_band_transformation = np.zeros([num_central_freq_bins - 1, HighFreqIdx + 1])

        for filter_idx in range(1, num_central_freq_bins - 1):

            low_cut_freq = central_freqs[filter_idx - 1]
            central_freq = central_freqs[filter_idx]
            high_cut_freq = central_freqs[filter_idx + 1]
            lbw = central_freq - low_cut_freq
            ubw = high_cut_freq - central_freq

            l = int(np.ceil(low_cut_freq / fr))
            r = int(high_cut_freq / fr)
            assert r <= HighFreqIdx

            if l >= r:
                if l <= HighFreqIdx:
                    freq_band_transformation[filter_idx, l] = 1
                continue

            for j in range(l, r + 1):
                assert freqs[j] >= low_cut_freq
                assert freqs[j] <= high_cut_freq
                if freqs[j] <= central_freq:
                    t = (freqs[j] - low_cut_freq) / lbw
                else:
                    t = (high_cut_freq - freqs[j]) / ubw
                assert t >= 0.
                assert t <= 1.
                freq_band_transformation[filter_idx, j] = t

        freq_band_transformation = freq_band_transformation.astype(np.float32)
        freq_band_transformation = freq_band_transformation.T
        freq_band_transformation = np.require(freq_band_transformation, requirements=['O', 'C'])

        return freq_band_transformation

    def _coef_matrix_for_quef_2_log_freq_fn(self):

        #  return shape: [int(float(fs) / fmin) + 1, num_central_freq_bins - 1]

        config = self.config
        fmin = config.fmin
        fmin = float(fmin)
        fs = config.sr
        fs = float(fs)
        HighQuefIdx = int(fs / fmin)  # inclusive

        central_freqs = config.central_freqs
        num_central_freq_bins = len(central_freqs)

        freq_band_transformation = np.zeros([num_central_freq_bins - 1, HighQuefIdx + 1])

        for filter_idx in range(1, num_central_freq_bins - 1):

            low_cut_freq = central_freqs[filter_idx - 1]
            central_freq = central_freqs[filter_idx]
            high_cut_freq = central_freqs[filter_idx + 1]
            lbw = central_freq - low_cut_freq
            ubw = high_cut_freq - central_freq

            bin_l = np.int(np.ceil(fs / high_cut_freq))
            bin_r = np.int(fs / low_cut_freq)
            assert bin_r <= HighQuefIdx

            for tbin in range(bin_l, bin_r + 1):
                bin_freq = fs / tbin
                assert bin_freq >= low_cut_freq
                assert bin_freq <= high_cut_freq

                if bin_freq <= central_freq:
                    t = (bin_freq - low_cut_freq) / lbw
                else:
                    t = (high_cut_freq - bin_freq) / ubw
                assert t >= 0
                assert t <= 1
                freq_band_transformation[filter_idx, tbin] = t
        freq_band_transformation = freq_band_transformation.astype(np.float32)
        freq_band_transformation = freq_band_transformation.T
        freq_band_transformation = np.require(freq_band_transformation, requirements=['O', 'C'])

        return freq_band_transformation

    @tf.function(
        input_signature=[tf.TensorSpec([None])], autograph=False
    )
    def _cfp_filterbank_tf_fn(self, samples):

        config = self.config

        g = config.gammas
        N = config.fft_len
        fs = config.sr
        fr = config.fr
        fmin, fmax = config.fmin, config.fmax
        hNp1 = N // 2 + 1

        fs = float(fs)
        fr = float(fr)
        fmax = float(fmax)
        fmin = float(fmin)

        assert len(g) == 3

        spec = self._stft_tf_fn(samples, g[0])
        spec.set_shape([None, hNp1])
        num_frames = tf.shape(spec)[0]
        assert spec.dtype == tf.float32

        ceps = tf.signal.irfft(
            tf.cast(spec, tf.complex64)
        )
        assert ceps.dtype == tf.float32
        ceps.set_shape([None, N])
        ceps = ceps[:, :hNp1]
        t = np.sqrt(N).astype(np.float32)
        t = tf.convert_to_tensor(t)
        ceps = ceps * t
        cutoff = int(fs / fmax)  # inclusive
        ceps = ceps[:, cutoff + 1:]
        ceps = tf.where(ceps < 0., tf.zeros_like(ceps), ceps)
        t = tf.convert_to_tensor(g[1], tf.float32)
        ceps = tf.pow(ceps, t)
        zeros = tf.zeros([num_frames, cutoff + 1], tf.float32)
        ceps = tf.concat([zeros, ceps], axis=1)
        ceps.set_shape([None, hNp1])

        t = tf.pad(ceps, [[0, 0], [0, N // 2 - 1]], mode='reflect')
        sqrt_N = 1. / np.sqrt(N)
        sqrt_N = sqrt_N.astype(np.float32)
        sqrt_N = tf.convert_to_tensor(sqrt_N)
        gcos = tf.signal.rfft(t)
        assert gcos.dtype == tf.complex64
        gcos.set_shape([None, hNp1])
        gcos = tf.math.real(gcos) * sqrt_N
        assert gcos.dtype == tf.float32
        cutoff = int(fmin / fr)  # inclusive
        gcos = gcos[:, cutoff + 1:-1]
        gcos = tf.where(gcos < 0., tf.zeros_like(gcos), gcos)
        if g[2] != 1:
            t = tf.convert_to_tensor(g[2], tf.float32)
            gcos = tf.pow(gcos, t)
        zeros = tf.zeros([num_frames, cutoff + 1])
        gcos = tf.concat([zeros, gcos], axis=1)
        gcos.set_shape([None, hNp1 - 1])

        spec = spec[:, :-1]
        ceps = ceps[:, :-1]

        HighFreqIdx = int(fmax / fr)  # inclusive
        spec = spec[:, :HighFreqIdx + 1]
        gcos = gcos[:, :HighFreqIdx + 1]

        HighQuefIdx = int(fs / fmin)
        ceps = ceps[:, :HighQuefIdx + 1]

        t = self._coef_matrix_for_freq_2_log_freq_fn()
        t = tf.convert_to_tensor(t, tf.float32)
        spec = tf.linalg.matmul(spec, t)
        gcos = tf.linalg.matmul(gcos, t)
        cfbins = len(config.central_freqs) - 1
        spec.set_shape([None, cfbins])
        gcos.set_shape([None, cfbins])
        t = self._coef_matrix_for_quef_2_log_freq_fn()
        t = tf.convert_to_tensor(t, tf.float32)
        ceps = tf.linalg.matmul(ceps, t)
        ceps.set_shape([None, cfbins])

        return spec, ceps, gcos

    @staticmethod
    @tf.function(input_signature=[tf.TensorSpec([None, None])], autograph=False)
    def _normalization_tf_fn(x):

        x = tf.convert_to_tensor(x)
        x = tf.math.log(x + 1.)
        _min, _max = tf.reduce_min(x), tf.reduce_max(x)
        valid = _max > _min + 1e-3
        tf.cond(valid, lambda: tf.no_op(), lambda: tf.print('warning - give up normalization because max and min are too close'))
        x = tf.cond(valid, lambda: (x - _min) / (_max - _min), lambda: x)

        return x

    def __call__(self, wav_file):

        config = self.config

        sr = config.sr
        win_len = config.win_len
        half_win_len = win_len // 2
        hop_size = config.hop_size
        num_freq_bins = len(config.central_freqs) - 1

        wav_info = soundfile.info(wav_file)
        if wav_info.samplerate != sr:
            logging.debug('the sample rate {} is not equal to the required value of {} so have to resample'.format(wav_info.samplerate, sr))

        samples, _sr = librosa.load(wav_file, sr=sr)
        assert np.all(np.logical_not(np.isnan(samples)))
        assert _sr == sr
        total_num_frames = (len(samples) + hop_size - 1) // hop_size
        samples = np.pad(samples, [[half_win_len, half_win_len - 1]], mode='reflect')
        t = (total_num_frames - 1) * hop_size + win_len
        assert t <= len(samples)
        samples = samples[:t]
        total_num_samples = len(samples)
        split_list = self._gen_split_fn(total_num_frames)
        spec_list = []
        ceps_list = []
        gcos_list = []
        for start_frame, end_frame in split_list:
            start_sample = start_frame * hop_size
            end_sample = (end_frame - start_frame - 1) * hop_size + start_sample + win_len
            assert end_sample <= total_num_samples
            spec, ceps, gcos = self._cfp_filterbank_tf_fn(samples[start_sample:end_sample])
            spec, ceps, gcos = spec.numpy(), ceps.numpy(), gcos.numpy()

            assert spec.shape == ceps.shape == gcos.shape == (end_frame - start_frame, num_freq_bins)
            assert spec.dtype == ceps.dtype == gcos.dtype == np.float32

            spec_list.append(spec)
            ceps_list.append(ceps)
            gcos_list.append(gcos)
        assert end_sample == total_num_samples

        spec = np.concatenate(spec_list, axis=0)
        ceps = np.concatenate(ceps_list, axis=0)
        gcos = np.concatenate(gcos_list, axis=0)

        assert spec.shape == ceps.shape == gcos.shape == (total_num_frames, num_freq_bins)
        assert spec.dtype == ceps.dtype == gcos.dtype == np.float32

        # for name, x in zip(['spec', 'ceps', 'gcos'], [spec, ceps, gcos]):
        #     (r, c), _max = mac_loc_and_value(x)
        #     print('{} - {} - {}'.format(name, (r, c), _max))

        ntf = CFP._normalization_tf_fn
        spec = ntf(spec)
        ceps = ntf(ceps)
        gcos = ntf(gcos)

        spec, ceps, gcos = spec.numpy(), ceps.numpy(), gcos.numpy()
        spec = np.stack([spec, ceps, gcos], axis=-1)
        assert spec.shape == (total_num_frames, num_freq_bins, 3)
        spec = np.require(spec, requirements=['O', 'C'])

        return spec


if __name__ == '__main__':

    cfp_ins = CFP()

    wav_file = os.path.join(os.environ['mir1k'], 'Wavfile', 'abjones_1_01.wav')

    spec = cfp_ins(wav_file)
    print(spec.shape)






