#!/usr/bin/env python

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import os
import sys
import argparse
import glob
import tensorflow as tf
import librosa
import numpy as np
from utils import nsgt as nsgt_module
import general.shaun.acoustic_model_shaun as acoustic_model_module_general
import vocal.shaun.acoustic_model_shaun as acoustic_model_module_vocal


def parser():

    class SmartFormatter(argparse.HelpFormatter):

        def _split_lines(self, text, width):
            if text.startswith('R|'):
                return text[2:].splitlines()
                # this is the RawTextHelpFormatter._split_lines
            return argparse.HelpFormatter._split_lines(self, text, width)

    p = argparse.ArgumentParser(description='extract general or vocal melody from arbitrary audio files', formatter_class=SmartFormatter)

    p.add_argument('input_files', nargs='+', help='one or multiple input audio files, supporting wildcard characters')
    p.add_argument('--output_dir', nargs='?', default='output', help='output directory, defaults to output')
    p.add_argument('--melody_type', default='general', choices=['general', 'vocal'], help='melody type: general or vocal, defaults to general')
    p.add_argument('--gpu_idx', default=0, type=int, help='which GPU to use, starting from 0, defaults to 0')
    p.add_argument('--test27', action='store_true', help="R|Relevant only for general melody extraction.\n"
                   "Determine which checkpoint to use for general melody extraction.\n"
                   "Use flag --test27 to select the checkpoint for partition (66, 15, 27).\n"
                   "Otherwise, the checkpoint for partition (67, 15, 26) will be used"
                   )

    args = p.parse_args()

    return args


class AcousticModel:

    def __init__(self, extract_general_melody):

        if extract_general_melody:
            acoustic_model = acoustic_model_module_general.create_acoustic_model_fn(1e-4)
        else:
            acoustic_model = acoustic_model_module_vocal.create_acoustic_model_fn(1e-4)

        if extract_general_melody:
            input_f_dim = 540
            output_f_dim = 360
        else:
            input_f_dim = 500
            output_f_dim = 320
        self.input_f_dim = input_f_dim
        self.output_f_dim = output_f_dim

        assert len(acoustic_model.trainable_variables) > 0
        self.acoustic_model = acoustic_model
        self.trainable_variables = acoustic_model.trainable_variables
        self.voicing_threshold = tf.Variable(.15, trainable=False, name='voicing_threshold')
        self.model_for_ckpt = dict(acoustic_model=acoustic_model, voicing_threshold=self.voicing_threshold)

    def __call__(self, inputs):

        assert inputs.shape[2] == self.input_f_dim
        outputs = self.acoustic_model(inputs, training=False)
        assert outputs.shape[1] == self.output_f_dim

        return outputs


class TFDataset:

    def __init__(self, model):

        self.model = model
        self.nsgt_hop_size = 64
        self.nsgt_freq_bins = 568

        nsgt_instances = []
        for power in (17, 18, 19, 20, 21, 22):
            nsgt_ins = nsgt_module.NSGT(2 ** power)
            assert not nsgt_ins.use_double
            assert nsgt_ins.Ls == 2 ** power
            nsgt_instances.append(nsgt_ins)
        self.nsgt_instances = nsgt_instances
        self.nsgt_Lses = [ins.Ls for ins in nsgt_instances]
        self.freq_dim = 540 if model.extract_general_melody else 500

        self.np_dataset = self.gen_np_dataset_fn()
        self.gen_split_list_fn(self.np_dataset)
        self.tf_dataset = self.gen_tf_dataset_fn()

    @staticmethod
    def spec_transform_fn(spec):

        top_db = 120.
        spec = librosa.amplitude_to_db(spec, ref=np.max, amin=1e-7, top_db=top_db)
        spec = spec / top_db + 1.
        spec = spec.astype(np.float32)

        return spec

    def gen_spec_fn(self, wav_file):

        nsgt_instances = self.nsgt_instances
        nsgt_Lses = self.nsgt_Lses
        hop_size = self.nsgt_hop_size
        num_freq_bins = self.nsgt_freq_bins

        sr = 44100
        samples, _sr = librosa.load(wav_file, sr=sr)
        assert _sr == sr
        assert samples.max() < 1
        assert samples.min() >= -1
        num_samples = len(samples)
        t = np.searchsorted(nsgt_Lses, num_samples)
        if t == 0:
            duration = num_samples / float(sr)
            sys.exit('the duration of {} is {} s and is too short'.format(wav_file, duration))
        assert 1 <= t <= len(nsgt_instances)
        nsgt_fn = nsgt_instances[t - 1].nsgt_of_wav_file_fn
        nsgt = nsgt_fn(samples)
        num_frames = (num_samples + hop_size - 1) // hop_size
        assert nsgt.shape == (num_frames, num_freq_bins)
        nsgt = nsgt[::4, 1:self.freq_dim + 1]
        nsgt = TFDataset.spec_transform_fn(nsgt)
        nsgt = np.require(nsgt, np.float32, requirements=['O', 'C'])

        return nsgt

    def gen_np_dataset_fn(self):

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('generating spectrograms ...')

        wav_files = model.wav_files
        num_wav_files = len(wav_files)
        dataset = []
        for wav_idx, wav_file in enumerate(wav_files):
            logging.info('{}/{}'.format(wav_idx + 1, num_wav_files))
            spec = self.gen_spec_fn(wav_file)
            dataset.append(spec)

        return dataset

    def _split_fn(self, num_frames):

        snippet_len = self.model.snippet_len

        split_frames = range(0, num_frames + 1, snippet_len)
        split_frames = list(split_frames)
        if split_frames[-1] != num_frames:
            split_frames.append(num_frames)
        start_end_frame_pairs = zip(split_frames[:-1], split_frames[1:])
        start_end_frame_pairs = [list(it) for it in start_end_frame_pairs]

        return start_end_frame_pairs

    def gen_split_list_fn(self, np_dataset):

        model = self.model

        wav_files = model.wav_files
        self.rec_names = tuple(wav_files)
        rec_start_end_idx_list = []
        for rec_idx, spec in enumerate(np_dataset):
            split_list = self._split_fn(len(spec))
            t = [[rec_idx] + se for se in split_list]
            rec_start_end_idx_list.extend(t)
        self.rec_start_end_idx_list = rec_start_end_idx_list

        num_frames = [len(spec) for spec in np_dataset]
        self.num_frames_vector = np.asarray(num_frames, dtype=np.int64)

    def map_idx_to_data_fn(self, idx):

        snippet_len = self.model.snippet_len
        freq_dim = self.freq_dim

        def py_fn(idx):

            idx = idx.numpy()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            spec = self.np_dataset[rec_idx]
            assert start_frame % snippet_len == 0
            snippet_idx = start_frame // snippet_len
            spec = spec[start_frame:end_frame]

            return rec_idx, snippet_idx, spec

        rec_idx, snippet_idx, spec = tf.py_function(
            py_fn,
            inp=[idx],
            Tout=[tf.int32, tf.int32, tf.float32]
        )
        rec_idx.set_shape([])
        snippet_idx.set_shape([])
        spec.set_shape([None, freq_dim])

        return dict(
            rec_idx=rec_idx,
            snippet_idx=snippet_idx,
            spectrogram=spec
        )

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(10)

        return dataset


class SaveMelody:

    def __init__(self, model):

        self.model = model
        self.rec_idx = None
        self.snippet_idx = None
        num_recs = len(model.wav_files)
        self.melody_written = np.zeros([num_recs], dtype='bool')

        num_frames = model.tf_dataset.num_frames_vector
        snippet_len = model.snippet_len
        self.snippet_len = snippet_len
        self.num_snippets = [(n + snippet_len - 1) // snippet_len for n in num_frames]

    @tf.function(
        input_signature=[tf.TensorSpec([None, None], name='logits')],
        autograph=False
    )
    def est_notes_fn(self, logits):

        model = self.model
        general_melody = model.extract_general_melody

        logits = tf.convert_to_tensor(logits, tf.float32)
        num_pitches = 360 if general_melody else 320
        logits.set_shape([None, num_pitches])

        note_offset = 23.6
        note_range = np.arange(num_pitches) * .2
        note_range = note_range.astype(np.float32)
        note_range = tf.constant(note_range)

        frames = tf.range(num_pitches, dtype=tf.int32)
        peak_indices = tf.argmax(logits, axis=-1, output_type=tf.int32)
        peak_masks = peak_indices[:, None] - frames[None, :]
        peak_masks.set_shape([None, num_pitches])
        peak_masks = tf.abs(peak_masks) <= 1
        probs = tf.nn.sigmoid(logits)
        masked_probs = tf.where(peak_masks, probs, tf.zeros_like(probs))
        norm_probs = tf.reduce_sum(masked_probs, axis=-1)
        norm_probs.set_shape([None])
        est_notes = note_range[None, :] * masked_probs
        est_notes = tf.reduce_sum(est_notes, axis=-1)
        est_notes.set_shape([None])
        est_notes = est_notes / tf.maximum(norm_probs, 1e-3)
        est_notes = est_notes + note_offset
        peak_probs = tf.gather_nd(probs, indices=peak_indices[:, None], batch_dims=1)
        vth = model.acoustic_model.voicing_threshold
        assert isinstance(vth, tf.Variable)
        est_notes = tf.where(peak_probs > vth, est_notes, tf.zeros_like(est_notes))

        return est_notes

    @staticmethod
    def midi_to_hz_fn(notes):

        freqs = np.zeros_like(notes)
        t1 = np.greater_equal(notes, 0)
        assert np.all(t1)
        positives = np.where(notes > .1)
        freqs[positives] = librosa.midi_to_hz(notes[positives])

        return freqs

    def update_states(self, rec_idx, snippet_idx, logits):

        model = self.model

        t = [rec_idx, snippet_idx, logits]
        assert all(isinstance(v, tf.Tensor) for v in t)

        _rec_idx = rec_idx.numpy()
        num_snippets = self.num_snippets[_rec_idx]
        _snippet_idx = snippet_idx.numpy()
        assert _snippet_idx < num_snippets

        if _snippet_idx == 0:
            self.rec_idx = _rec_idx
            self.snippet_idx = _snippet_idx
            self.est_notes = []

        if _snippet_idx > 0:
            assert _snippet_idx == self.snippet_idx + 1
        self.snippet_idx = _snippet_idx

        est_notes = self.est_notes_fn(logits)
        self.est_notes.append(est_notes)

        if _snippet_idx == num_snippets - 1:
            est_notes = np.concatenate(self.est_notes)
            num_frames = model.tf_dataset.num_frames_vector[_rec_idx]
            assert len(est_notes) == num_frames
            times = np.arange(num_frames) * (256. / 44100.)
            freqs = SaveMelody.midi_to_hz_fn(est_notes)
            lines = []
            for t, f in zip(times, freqs):
                line = '{}, {}\n'.format(t, f)
                lines.append(line)

            wav_file = model.tf_dataset.rec_names[_rec_idx]
            base_name = os.path.basename(wav_file).split('.')[0]
            output_dir = model.output_dir
            suffix = '_general.csv' if model.extract_general_melody else '_vocal.csv'
            csv_file = os.path.join(output_dir, base_name + suffix)
            with open(csv_file, 'w') as fh:
                fh.writelines(lines)
            logging.info('melody written to {}'.format(csv_file))
            self.melody_written[_rec_idx] = True


class Model:

    def __init__(self, melody_type, use_p26, gpu_idx, wav_files, output_dir):

        Model.set_gpu_fn(gpu_idx)
        assert melody_type in ('general', 'vocal')
        extract_general_melody = melody_type == 'general'
        self.extract_general_melody = extract_general_melody
        self.wav_files = wav_files
        self.output_dir = output_dir
        self.snippet_len = 1200

        self.acoustic_model = AcousticModel(extract_general_melody)
        self.tf_dataset = TFDataset(self)

        if not extract_general_melody:
            ckpt_file = os.path.join('checkpoints', 'vocal', 'shaun', 'd0-28')
        else:
            if use_p26:
                ckpt_file = os.path.join('checkpoints', 'general', 'shaun', '67_15_26', 'd0-39')
            else:
                ckpt_file = os.path.join('checkpoints', 'general', 'shaun', '66_15_27', 'd0-21')
        self.ckpt_file = ckpt_file

    @staticmethod
    def set_gpu_fn(gpu_id):

        gpus = tf.config.list_physical_devices('GPU')
        num_gpus = len(gpus)
        assert num_gpus > 0
        assert gpu_id >= 0
        assert gpu_id < num_gpus
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)

    def __call__(self):

        ckpt_file = self.ckpt_file
        ckpt = tf.train.Checkpoint(model=self.acoustic_model.model_for_ckpt)
        status = ckpt.restore(ckpt_file)
        status.expect_partial()
        status.assert_existing_objects_matched()

        acoustic_model = self.acoustic_model
        assert not hasattr(self.tf_dataset, 'iterator')
        iterator = iter(self.tf_dataset.tf_dataset)
        save_melody = SaveMelody(self)
        batches_per_epoch = len(self.tf_dataset.rec_start_end_idx_list)

        for batch_idx in range(batches_per_epoch):

            batch = iterator.get_next()
            logits = acoustic_model(batch['spectrogram'])
            save_melody.update_states(
                rec_idx=batch['rec_idx'][0],
                snippet_idx=batch['snippet_idx'][0],
                logits=logits
            )
        batch = iterator.get_next_as_optional()
        assert not batch.has_value()
        assert np.all(save_melody.melody_written)


if __name__ == '__main__':

    args = parser()

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.system('mkdir -p {}'.format(output_dir))

    input_files = []
    for file in args.input_files:
        files = glob.glob(file)
        assert isinstance(files, list)
        if len(files) == 0:
            logging.warning('cannot find any files matching {}'.format(file))
        input_files.extend(files)

    model = Model(
        melody_type=args.melody_type,
        use_p26=not args.test27,
        gpu_idx=args.gpu_idx,
        wav_files=input_files,
        output_dir=args.output_dir
    )
    model()

