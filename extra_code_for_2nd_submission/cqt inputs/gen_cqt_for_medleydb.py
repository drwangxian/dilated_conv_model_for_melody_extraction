"""
cqt for medleydb, use cqt function modified by shaun
linear magnitude
previously incorrect fmin was used
"""

DEBUG = False

import os
import numpy as np
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import medleydb
import librosa
from self_defined import save_np_array, load_np_array

# where to store outputs. If does not exist, will be created automatically
FOLDER = os.environ['medleydb_cqt']


class MiscFns(object):

    @staticmethod
    def melody_tids_fn():
        melody_ids = []
        for tid in medleydb.TRACK_LIST_V1:
            if medleydb.MultiTrack(tid).has_melody:
                melody_ids.append(tid)
        assert len(melody_ids) == 108

        return melody_ids

    @staticmethod
    def librosa_cqt_fn(tid):

        wav_file = os.path.join(os.environ['medleydb'], tid, tid + '_MIX.wav')
        samples, sr = librosa.load(wav_file, sr=None)
        assert sr == 44100
        assert samples.ndim == 1
        num_frames = (len(samples) + 255) // 256
        fmin = librosa.midi_to_hz([24])[0] * 2. ** (-2. / 60)
        bins_per_octave = 60
        num_bins = 540
        cqt = librosa.cqt(samples, sr=sr, hop_length=256, fmin=fmin, n_bins=num_bins, bins_per_octave=bins_per_octave)
        cqt = np.abs(cqt)
        cqt = cqt.T

        diff = len(cqt) - num_frames
        assert 0 <= diff <= 1
        if diff == 1:
            cqt = cqt[:-1]

        cqt = np.require(cqt, dtype=np.float32, requirements=['C'])

        return cqt


class Config(object):

    def __init__(self):
        self.debug_mode = DEBUG
        self.folder = FOLDER
        self.tids = MiscFns.melody_tids_fn()
        if self.debug_mode:
            del self.tids[3:]


class GenCQT(object):
    def __init__(self, config):
        self.config = config
        folder = self.config.folder
        if not os.path.isdir(folder):
            logging.info('folder {} does not exist, so create one'.format(folder))
            os.system('mkdir {}'.format(folder))
        self.folder = folder

    def gen_cqt_fn(self):

        num_recs = len(self.config.tids)
        for t_idx, tid in enumerate(self.config.tids):
            logging.info('{}/{} - {}'.format(t_idx + 1, num_recs, tid))

            cqt_output_file_name = os.path.join(self.folder, tid + '.cqt')
            if os.path.isfile(cqt_output_file_name):
                try:
                    name_returned, _ = load_np_array(cqt_output_file_name)
                    if name_returned == tid:
                        logging.info('{} already exists so skip this recording'.format(cqt_output_file_name))
                        continue
                    else:
                        logging.info('{} already exists but seems cracked so re-generate it'.format(
                            cqt_output_file_name))
                except Exception as _e:
                    print _e
                    logging.info(
                        '{} already exists but seems cracked so re-generate it'.format(cqt_output_file_name))

            cqt = MiscFns.librosa_cqt_fn(tid)
            save_np_array(cqt_output_file_name, cqt, tid)

            if t_idx == 0:
                _tid, _cqt = load_np_array(cqt_output_file_name)
                assert _tid == tid
                assert np.all(cqt == _cqt)


def main():
    gen_cqt_ins = GenCQT(config=Config())
    gen_cqt_ins.gen_cqt_fn()


if __name__ == '__main__':
    main()









