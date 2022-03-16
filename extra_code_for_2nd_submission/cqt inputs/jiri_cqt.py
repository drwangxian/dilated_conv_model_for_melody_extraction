"""
Monday 1-Nov-2021
jiri-style melody extraction
shaun acoustic model
vqt inputs
tf2
group lasso
"""

DEBUG = False
GPU_ID = 0

import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import tensorflow as tf
from argparse import Namespace
import glob
import os
import numpy as np
import librosa
from self_defined import ArrayToTableTFFn
from self_defined import load_np_array_from_file_fn
import acoustic_model_shaun as acoustic_model_module
import json


class Config:

    def __init__(self):

        self.debug_mode = DEBUG
        Config.set_gpu_fn()

        self.train_or_inference = Namespace(
            inference='d0-39',
            from_ckpt=None,
            ckpt_prefix=None
        )
        self.tb_dir = 'tb_inf'

        if self.train_or_inference.inference is None:
            self.model_names = ('training', 'validation')
        else:
            self.model_names = ('training', 'validation', 'test')

        # check if tb_dir and checkpoints with the same prefix already exist
        if not self.debug_mode:
            self.chk_if_tb_dir_and_model_with_same_prefix_exist_fn()

        self.snippet_len = 1200
        self.initial_learning_rate = 1e-4
        self.batches_per_epoch = None
        self.patience_epochs = 10

        self.tvt_split_dict = Config.get_dataset_split_fn()
        if self.debug_mode:
            for key in self.tvt_split_dict:
               self.tvt_split_dict[key] = self.tvt_split_dict[key][:3]

        self.acoustic_model_ins = AcousticModel(self)

        note_range = TFDataset.note_range
        # build optimizer so as to create relevant optimizer weights
        if self.train_or_inference.inference is None:
            self.learning_rate = tf.Variable(
                self.initial_learning_rate, dtype=tf.float32, name='learning_rate', trainable=False)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            spec = tf.random.normal([1, self.snippet_len, 540])
            lower_note = note_range[0]
            upper_note = note_range[-1]
            labels = np.random.uniform(lower_note, upper_note, [self.snippet_len])
            labels[[100, 200, 300]] = 18
            labels[[400, 500, 600]] = 0
            labels[[700, 800, 900]] = 100
            with tf.GradientTape() as tape:
                logits = self.acoustic_model_ins(spec, training=False)
                loss = self.acoustic_model_ins.loss_tf_fn(ref_notes=labels, logits=logits)
            grads = tape.gradient(loss, self.acoustic_model_ins.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.acoustic_model_ins.trainable_variables))
            assert len(self.optimizer.weights) > 0

            if self.debug_mode:
                print('weights of the optimizer: ')
                for idx, w in enumerate(self.optimizer.weights):
                    print(idx, w.name, w.shape)
                print()

    @staticmethod
    def set_gpu_fn():

        gpus = tf.config.list_physical_devices('GPU')
        assert len(gpus) >= 1
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        global GPU_ID
        if len(gpus) == 1:
            GPU_ID = 0
        tf.config.set_visible_devices(gpus[GPU_ID], 'GPU')

    def chk_if_tb_dir_and_model_with_same_prefix_exist_fn(self):

        # check if tb_dir exists
        assert self.tb_dir is not None
        is_tb_dir_exist = glob.glob('{}/'.format(self.tb_dir))
        if is_tb_dir_exist:
            assert False, 'directory {} already exists'.format(self.tb_dir)

        # check if model exists
        if self.train_or_inference.inference is None and self.train_or_inference.ckpt_prefix is not None:
            ckpt_dir, ckpt_prefix = os.path.split(self.train_or_inference.ckpt_prefix)
            assert ckpt_prefix != ''
            if ckpt_dir == '':
                ckpt_dir = 'ckpts'

            is_exist = glob.glob('{}/{}*'.format(ckpt_dir, ckpt_prefix))
            if is_exist:
                assert False, 'checkpoints with prefix {} already exist'.format(ckpt_prefix)

    @staticmethod
    def get_dataset_split_fn():
        split_file = os.environ['medleydb']
        split_file = os.path.join(split_file, '..', 'V1_auxiliary', 'data_splits_jiri.json')
        with open(split_file, 'rb') as fh:
            split_dict = json.load(fh)
        assert len(split_dict['test']) == 27
        assert len(split_dict['train']) == 67
        assert len(split_dict['validation']) == 15

        intersection_track_id = 'MatthewEntwistle_DontYouEver'
        assert intersection_track_id in split_dict['test']
        split_dict['test'].remove(intersection_track_id)

        split_dict = dict(
            test=split_dict['test'],
            training=split_dict['train'],
            validation=split_dict['validation']
        )

        return split_dict


class AcousticModel:

    def __init__(self, config, reg=1e-4):

        acoustic_model = acoustic_model_module.create_acoustic_model_fn()

        if config.debug_mode:
            acoustic_model.summary(line_length=150)
            print('trainable variables - ')
            for idx, w in enumerate(acoustic_model.trainable_variables):
                print(idx, w.name, w.shape)

        assert len(acoustic_model.trainable_variables) > 0

        self.acoustic_model = acoustic_model
        self.trainable_variables = acoustic_model.trainable_variables

        self.voicing_threshold = tf.Variable(.15, trainable=False, name='voicing_threshold')

        self.model_for_ckpt = dict(acoustic_model=acoustic_model, voicing_threshold=self.voicing_threshold)
        self.config = config
        self.cutoff_prob = 4e-3

        for w in acoustic_model.trainable_variables:
            if w.shape == (1, 97, 16, 128):
                break
        else:
            assert False

        self.global_kernel = w
        self.reg = reg
        logging.info('l2 regularization coefficient - {}'.format(self.reg))

    @tf.function
    def reg_l2_fn(self):

        w = self.global_kernel
        w.set_shape([1, 97, 16, 128])
        w = tf.square(w)
        w = tf.reduce_sum(w)
        w = tf.math.sqrt(w)
        w = w * self.reg

        return w

    def __call__(self, inputs, training):

        assert isinstance(training, bool)
        outputs = inputs

        outputs = self.acoustic_model(outputs, training=training)

        return outputs

    @tf.function(input_signature=[
        tf.TensorSpec([None], dtype='float32'),
        tf.TensorSpec([None, 360], dtype='float32')
    ], autograph=False)
    def loss_tf_fn(self, ref_notes, logits):

        note_range = TFDataset.note_range

        ref_notes = tf.convert_to_tensor(ref_notes, dtype='float32')
        logits = tf.convert_to_tensor(logits, dtype='float32')

        ref_notes.set_shape([None])
        assert np.isclose(note_range[0], 23.6) and len(note_range) == 360
        min_note = note_range[0] - .4
        min_note = min_note.astype(np.float32)
        t1 = tf.greater(ref_notes, .1)
        t2 = tf.less(ref_notes, min_note)
        t = tf.logical_and(t1, t2)
        ref_notes = tf.where(t, tf.fill([tf.size(ref_notes)], min_note), ref_notes)

        max_note = note_range[-1] + .4
        max_note = max_note.astype(np.float32)
        ref_notes = tf.minimum(ref_notes, max_note)

        note_range = tf.convert_to_tensor(note_range, dtype=tf.float32)
        ref_notes = ref_notes[:, None] - note_range[None, :]
        ref_notes.set_shape([None, 360])
        ref_notes = - ref_notes ** 2 / (2. * .18 ** 2)
        ref_notes = tf.exp(ref_notes)
        ref_notes = tf.where(ref_notes < self.cutoff_prob, tf.zeros_like(ref_notes), ref_notes)

        logits.set_shape([None, 360])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ref_notes, logits=logits)
        loss.set_shape([None, 360])
        loss = tf.reduce_mean(loss)

        return loss


class TFDataset:

    medleydb_dir = os.environ['medleydb']
    melody2_dir = os.environ['melody2_dir']
    cqt_dir = os.path.join(os.environ['medleydb'], '..', 'cqt_linear_mag')
    mix_suffix = '_MIX.wav'
    melody2_suffix = '_MELODY2.csv'
    min_melody_freq = librosa.midi_to_hz(23.6)

    note_min = 23.6
    note_range = np.arange(360) / 5.
    note_range = note_range + note_min
    note_range = note_range.astype(np.float32)
    note_range.flags['WRITEABLE'] = False

    def __init__(self, model):

        self.model = model

    @staticmethod
    def spec_transform_fn(spec):

        top_db = 120.
        spec = librosa.amplitude_to_db(spec, ref=np.max, amin=1e-7, top_db=top_db)
        spec = spec / top_db + 1.
        spec = spec.astype(np.float32)

        return spec

    @staticmethod
    def gen_spec_fn(track_id):

        _track_id, cqt = load_np_array_from_file_fn(os.path.join(TFDataset.cqt_dir, track_id + '.cqt'))
        assert _track_id == track_id
        cqt = TFDataset.spec_transform_fn(cqt)

        return cqt

    @staticmethod
    def hz_to_midi_fn(freqs):

        notes = np.zeros_like(freqs)
        positives = np.nonzero(freqs)
        notes[positives] = librosa.hz_to_midi(freqs[positives])

        return notes

    @staticmethod
    def gen_label_fn(track_id):

        melody2_dir = TFDataset.melody2_dir
        melody2_suffix = TFDataset.melody2_suffix
        min_melody_freq = TFDataset.min_melody_freq

        annot_path = os.path.join(melody2_dir, track_id + melody2_suffix)
        times_labels = np.genfromtxt(annot_path, delimiter=',')
        assert times_labels.ndim == 2 and times_labels.shape[1] == 2
        num_frames = len(times_labels)
        tmp = np.arange(num_frames) * (256. / 44100.)
        assert np.all(tmp == times_labels[:, 0])

        freqs = times_labels[:, 1]
        all_zeros = freqs == 0.
        all_positives = freqs > min_melody_freq
        all_valid = np.logical_or(all_zeros, all_positives)
        assert np.all(all_valid)

        notes = TFDataset.hz_to_midi_fn(freqs)

        return notes

    def gen_np_dataset_fn(self):

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('{} - generate spectrograms and labels'.format(model.name))

        config = model.config
        track_ids = config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.debug('{}/{}'.format(track_idx + 1, num_tracks))

            spec = TFDataset.gen_spec_fn(track_id)
            notes = TFDataset.gen_label_fn(track_id)

            diff = len(notes) - len(spec)
            assert 0 <= diff <= 1
            if diff == 1:
                spec = np.pad(spec, [[0, 1], [0, 0]])

            spec = np.require(spec, np.float32, ['O', 'C'])
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes))

        return dataset

    def note_out_of_range_chk_fn(self, np_dataset):

        model = self.model
        note_range = TFDataset.note_range

        logging.info('{} - note range checking ... '.format(model.name))
        note_min = min(note for rec_dict in np_dataset for note in rec_dict['notes'] if note > 0)
        note_max = max(note for rec_dict in np_dataset for note in rec_dict['notes'] if note > 0)

        lower_note = note_range[0]
        upper_note = note_range[-1]
        logging.info('note range - ({}, {})'.format(lower_note, upper_note))
        if note_min < lower_note or note_min > upper_note:
            logging.warning('note min - {} - out of range'.format(note_min))
        if note_max < lower_note or note_max > upper_note:
            logging.warning('note max - {} - out of range'.format(note_max))

    @staticmethod
    def gen_split_list_fn(num_frames, snippet_len):

        split_frames = range(0, num_frames + 1, snippet_len)
        split_frames = list(split_frames)
        if split_frames[-1] != num_frames:
            split_frames.append(num_frames)
        start_end_frame_pairs = zip(split_frames[:-1], split_frames[1:])
        start_end_frame_pairs = [list(it) for it in start_end_frame_pairs]

        return start_end_frame_pairs


class TFDatasetForTrainingModeTrainingSplit(TFDataset):

    def __init__(self, model):

        super(TFDatasetForTrainingModeTrainingSplit, self).__init__(model)

        is_inferencing = model.config.train_or_inference.inference is not None
        assert not is_inferencing
        assert model.is_training
        assert 'train' in model.name

        self.np_dataset = self.gen_np_dataset_fn()
        self.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_rec_start_end_idx_fn(self.np_dataset)
        self.tf_dataset = self.gen_tf_dataset_fn()
        self.iterator = iter(self.tf_dataset)
        self.set_num_batches_per_epoch_fn()

    def gen_rec_start_end_idx_fn(self, np_dataset):

        snippet_len = self.model.config.snippet_len
        rec_start_and_end_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=len(rec_dict['spectrogram']),
                snippet_len=snippet_len
            )
            tmp = [[rec_idx] + se for se in split_list]
            rec_start_and_end_list.extend(tmp)

        return rec_start_and_end_list

    def map_idx_to_data_fn(self, idx):

        def py_fn(idx):

            idx = idx.numpy().item()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = self.np_dataset[rec_idx]
            spec = rec_dict['spectrogram'][start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            return spec, notes

        spec, notes = tf.py_function(py_fn, inp=[idx], Tout=['float32', 'float32'])
        spec.set_shape([None, 540])
        notes.set_shape([None])

        return dict(spectrogram=spec, notes=notes)

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.shuffle(num_snippets, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(10)

        return dataset

    def set_num_batches_per_epoch_fn(self):

        model = self.model
        if model.config.batches_per_epoch is None:
            batches_per_epoch = len(self.rec_start_end_idx_list)
            model.config.batches_per_epoch = batches_per_epoch
            logging.info('batches per epoch set to {}'.format(batches_per_epoch))


class TFDatasetForInferenceMode(TFDataset):

    def __init__(self, model):

        super(TFDatasetForInferenceMode, self).__init__(model)

        is_inferencing = model.config.train_or_inference.inference is not None

        if not is_inferencing:
            assert not model.is_training
            assert 'train' not in model.name
            assert 'test' not in model.name
            assert 'validation' in model.name

        self.np_dataset = self.gen_np_dataset_fn()
        self.note_out_of_range_chk_fn(self.np_dataset)
        self.gen_split_list_fn(self.np_dataset)
        self.tf_dataset = self.gen_tf_dataset_fn()

    def gen_split_list_fn(self, np_dataset):

        model = self.model
        snippet_len = model.config.snippet_len

        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)

        rec_start_end_idx_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=len(rec_dict['spectrogram']),
                snippet_len=snippet_len
            )
            rec_dict['split_list'] = split_list
            t = [[rec_idx] + se for se in split_list]
            rec_start_end_idx_list.extend(t)
        self.rec_start_end_idx_list = rec_start_end_idx_list

        num_frames = [len(rec_dict['spectrogram']) for rec_dict in np_dataset]
        self.num_frames_vector = np.asarray(num_frames, dtype=np.int64)

    def map_idx_to_data_fn(self, idx):

        def py_fn(idx):

            idx = idx.numpy().item()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = self.np_dataset[rec_idx]
            snippet_len = self.model.config.snippet_len
            assert start_frame % snippet_len == 0
            snippet_idx = start_frame // snippet_len
            spec = rec_dict['spectrogram'][start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            return rec_idx, snippet_idx, spec, notes

        rec_idx, snippet_idx, spec, notes = tf.py_function(
            py_fn,
            inp=[idx],
            Tout=[tf.int32, tf.int32, tf.float32, tf.float32]
        )
        rec_idx.set_shape([])
        snippet_idx.set_shape([])
        spec.set_shape([None, 540])
        notes.set_shape([None])

        return dict(
            rec_idx=rec_idx,
            snippet_idx=snippet_idx,
            spectrogram=spec,
            notes=notes
        )

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(10)

        return dataset


class MetricsTrainingModeTrainingSplit:

    def __init__(self, model):

        assert model.config.train_or_inference.inference is None
        assert model.is_training
        assert 'train' in model.name

        self.model = model
        self.voicing_threshold = model.config.acoustic_model_ins.voicing_threshold
        self.var_dict = self.define_tf_variables_fn()

        self.oa = None
        self.loss = None
        self.current_voicing_threshold = None

    def reset(self):

        for var in self.var_dict['all_updated']:
            var = var.deref()
            var.assign(tf.zeros_like(var))

        self.oa = None
        self.loss = None
        self.current_voicing_threshold = None

    def define_tf_variables_fn(self):

        model = self.model

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                all_defined_vars_updated = dict()

                def gen_tf_var(name):

                    assert name

                    with tf.device('/cpu:0'):
                        var = tf.Variable(
                            initial_value=tf.zeros([], dtype=tf.int64),
                            trainable=False,
                            name=name
                        )
                    var_ref = var.ref()
                    assert var_ref not in all_defined_vars_updated
                    all_defined_vars_updated[var_ref] = False

                    return var

                melody_var_dict = dict(
                    gt=dict(
                        voiced=gen_tf_var('voiced'),
                        unvoiced=gen_tf_var('unvoiced')
                    ),
                    voicing=dict(
                        correct_voiced=gen_tf_var('correct_voiced'),
                        incorrect_voiced=gen_tf_var('incorrect_voiced'),
                        correct_unvoiced=gen_tf_var('correct_unvoiced')
                    ),
                    correct_pitches=dict(
                        wide=gen_tf_var('correct_pitches_wide'),
                        strict=gen_tf_var('correct_pitches_strict')
                    ),
                    correct_chromas=dict(
                        wide=gen_tf_var('correct_chromas_wide'),
                        strict=gen_tf_var('correct_chromas_strict')
                    )
                )

                batch_counter = tf.Variable(
                    initial_value=tf.zeros([], dtype=tf.int32),
                    trainable=False,
                    name='batch_counter'
                )
                all_defined_vars_updated[batch_counter.ref()] = False
                loss = tf.Variable(
                    initial_value=tf.zeros([], tf.float32),
                    trainable=False,
                    name='acc_loss'
                )
                all_defined_vars_updated[loss.ref()] = False

                return dict(
                    melody=melody_var_dict,
                    batch_counter=batch_counter,
                    loss=loss,
                    all_updated=all_defined_vars_updated
                )

    def update_melody_var_fn(self, l1, l2, value):

        assert l1 is not None
        assert l2 is not None
        assert value is not None

        var_dict = self.var_dict['melody']
        all_updated = self.var_dict['all_updated']
        var = var_dict[l1][l2]

        var_ref = var.ref()
        assert not all_updated[var_ref]

        var.assign_add(value)
        all_updated[var_ref] = True

    def update_loss_fn(self, value):

        assert value is not None

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['loss']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        assert value.dtype == var.dtype
        var.assign_add(value)
        all_updated[var_ref] = True

    def increase_batch_counter_fn(self):

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['batch_counter']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        var.assign_add(1)
        all_updated[var_ref] = True

    @tf.function(input_signature=[
        tf.TensorSpec([1, None], 'float32'),
        tf.TensorSpec([None, 360], 'float32'),
        tf.TensorSpec([], 'float32')
    ], autograph=False)
    def update_states_tf_fn(self, ref_notes, logits, loss):

        voicing_threshold = self.voicing_threshold
        count_nonzero_fn = MetricsBase.count_nonzero_fn

        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)
        loss = tf.convert_to_tensor(loss, tf.float32)

        ref_notes = tf.squeeze(ref_notes, axis=0)

        est_probs = tf.sigmoid(logits)

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_peak_indices = tf.argmax(est_probs, axis=1, output_type=tf.int32)
        est_peak_probs = tf.gather_nd(est_probs, est_peak_indices[:, None], batch_dims=1)
        est_voicing = est_peak_probs > voicing_threshold
        est_voicing.set_shape([None])
        n_est_voicing = tf.logical_not(est_voicing)

        est_notes = MetricsBase.est_notes_fn(est_peak_indices=est_peak_indices, est_probs=est_probs)

        est_ref_note_diffs = tf.abs(est_notes - ref_notes)
        est_ref_note_diffs.set_shape([None])

        voiced_frames = count_nonzero_fn(ref_voicing)
        unvoiced_frames = tf.size(ref_voicing, tf.int64) - voiced_frames
        correct_voiced_frames = tf.logical_and(ref_voicing, est_voicing)
        correct_voiced_frames = count_nonzero_fn(correct_voiced_frames)
        incorrect_voiced_frames = tf.logical_and(n_ref_voicing, est_voicing)
        incorrect_voiced_frames = count_nonzero_fn(incorrect_voiced_frames)
        correct_unvoiced_frames = tf.logical_and(n_ref_voicing, n_est_voicing)
        correct_unvoiced_frames = count_nonzero_fn(correct_unvoiced_frames)
        self.update_melody_var_fn('gt', 'voiced', voiced_frames)
        self.update_melody_var_fn('gt', 'unvoiced', unvoiced_frames)
        self.update_melody_var_fn('voicing', 'correct_voiced', correct_voiced_frames)
        self.update_melody_var_fn('voicing', 'incorrect_voiced', incorrect_voiced_frames)
        self.update_melody_var_fn('voicing', 'correct_unvoiced', correct_unvoiced_frames)

        correct_pitches_wide = est_ref_note_diffs < .5
        correct_pitches_wide = tf.logical_and(ref_voicing, correct_pitches_wide)
        correct_pitches_strict = tf.logical_and(est_voicing, correct_pitches_wide)
        correct_pitches_wide = count_nonzero_fn(correct_pitches_wide)
        correct_pitches_strict = count_nonzero_fn(correct_pitches_strict)
        self.update_melody_var_fn('correct_pitches', 'wide', correct_pitches_wide)
        self.update_melody_var_fn('correct_pitches', 'strict', correct_pitches_strict)

        correct_chromas_wide = est_ref_note_diffs
        octave = MetricsBase.octave(correct_chromas_wide)
        correct_chromas_wide = tf.abs(correct_chromas_wide - octave) < .5
        correct_chromas_wide = tf.logical_and(ref_voicing, correct_chromas_wide)
        correct_chromas_strict = tf.logical_and(est_voicing, correct_chromas_wide)
        correct_chromas_wide = count_nonzero_fn(correct_chromas_wide)
        correct_chromas_strict = count_nonzero_fn(correct_chromas_strict)
        self.update_melody_var_fn('correct_chromas', 'wide', correct_chromas_wide)
        self.update_melody_var_fn('correct_chromas', 'strict', correct_chromas_strict)

        self.update_loss_fn(loss)
        self.increase_batch_counter_fn()

        assert all(self.var_dict['all_updated'].values())

    def update_states(self, ref_notes, logits, loss):

        t = [ref_notes, logits, loss]
        assert all(isinstance(v, tf.Tensor) for v in t)

        return self.update_states_tf_fn(ref_notes, logits, loss)

    def results(self):

        melody_dict = self.var_dict['melody']
        var_loss = self.var_dict['loss']
        var_batch_counter = self.var_dict['batch_counter']
        f8f4div = MetricsBase.to_f8_divide_and_to_f4_fn

        correct_frames = melody_dict['correct_pitches']['strict'] + melody_dict['voicing']['correct_unvoiced']
        num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        m_oa = f8f4div(correct_frames, num_frames_vector)
        m_oa.set_shape([])

        m_vrr = f8f4div(melody_dict['voicing']['correct_voiced'], melody_dict['gt']['voiced'])
        m_vfa = f8f4div(melody_dict['voicing']['incorrect_voiced'], melody_dict['gt']['unvoiced'])
        m_va = f8f4div(
            melody_dict['voicing']['correct_voiced'] + melody_dict['voicing']['correct_unvoiced'],
            num_frames_vector
        )
        m_rpa_strict = f8f4div(
            melody_dict['correct_pitches']['strict'], melody_dict['gt']['voiced']
        )
        m_rpa_wide = f8f4div(
            melody_dict['correct_pitches']['wide'], melody_dict['gt']['voiced']
        )
        m_rca_strict = f8f4div(
            melody_dict['correct_chromas']['strict'], melody_dict['gt']['voiced']
        )
        m_rca_wide = f8f4div(
            melody_dict['correct_chromas']['wide'], melody_dict['gt']['voiced']
        )
        m_loss = var_loss / tf.cast(var_batch_counter, tf.float32)

        self.oa = m_oa.numpy().item()
        self.loss = m_loss.numpy().item()
        self.current_voicing_threshold = self.voicing_threshold.read_value().numpy().item()

        results = dict(
            loss=m_loss,
            vrr=m_vrr,
            vfa=m_vfa,
            va=m_va,
            rpa_strict=m_rpa_strict,
            rpa_wide=m_rpa_wide,
            rca_strict=m_rca_strict,
            rca_wide=m_rca_wide,
            oa=m_oa
        )

        return results


class MetricsBase:

    """
    base metric class for validation split in training mode and inference mode
    """

    def __init__(self, model):

        self.model = model
        self.num_recs = len(model.config.tvt_split_dict[model.name])

        self.oa = None
        self.loss = None
        self.rec_idx = None
        self.snippet_idx = None
        self.current_voicing_threshold = None

    def reset(self):

        for var in self.var_dict['all_updated']:
            var = var.deref()
            var.assign(tf.zeros_like(var))

        self.oa = None
        self.loss = None
        self.rec_idx = None
        self.snippet_idx = None
        self.current_voicing_threshold = None

    def update_melody_var_fn(self, rec_idx, l1, l2, value):

        assert rec_idx is not None
        assert l1 is not None
        assert l2 is not None
        assert value is not None

        var_dict = self.var_dict['melody']
        all_updated = self.var_dict['all_updated']
        var = var_dict[l1][l2]

        var_ref = var.ref()
        assert not all_updated[var_ref]

        with tf.device(var.device):
            value = tf.identity(value)
        var.scatter_add(tf.IndexedSlices(values=value, indices=rec_idx))
        all_updated[var_ref] = True

    def update_loss_fn(self, value):

        assert value is not None

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['loss']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        assert value.dtype == var.dtype
        var.assign_add(value)
        all_updated[var_ref] = True

    def increase_batch_counter_fn(self):

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['batch_counter']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        var.assign_add(1)
        all_updated[var_ref] = True

    @staticmethod
    def to_f8_divide_and_to_f4_fn(numerator, denominator):

        numerator = tf.cast(numerator, tf.float64)
        denominator = tf.cast(denominator, tf.float64)
        numerator = numerator / (denominator + 1e-7)
        numerator = tf.cast(numerator, tf.float32)

        return numerator

    @staticmethod
    def count_nonzero_fn(inputs):

        outputs = inputs
        outputs = tf.math.count_nonzero(outputs, dtype=tf.int32)
        outputs = tf.cast(outputs, tf.int64)

        return outputs

    @staticmethod
    def est_notes_fn(est_peak_indices, est_probs):

        note_range = TFDataset.note_range
        note_offset = note_range[0]
        assert np.isclose(note_offset, 23.6)
        note_range = note_range - note_offset
        note_range = tf.constant(note_range, dtype=tf.float32)

        frames_360 = tf.range(360, dtype=tf.int32)
        peak_masks = est_peak_indices[:, None] - frames_360[None, :]
        peak_masks.set_shape([None, 360])
        peak_masks = tf.abs(peak_masks) <= 1
        masked_probs = tf.where(peak_masks, est_probs, tf.zeros_like(est_probs))
        masked_probs.set_shape([None, 360])
        normalization_probs = tf.reduce_sum(masked_probs, axis=1)
        normalization_probs.set_shape([None])
        frames_72 = note_range
        est_notes = frames_72[None, :] * masked_probs
        est_notes = tf.reduce_sum(est_notes, axis=1)
        est_notes.set_shape([None])
        est_notes = est_notes / tf.maximum(1e-3, normalization_probs)
        est_notes = est_notes + note_offset

        return est_notes

    def update_states(self, rec_idx, snippet_idx, ref_notes, logits, loss):

        model = self.model

        t = [rec_idx, snippet_idx, ref_notes, logits, loss]
        assert all(isinstance(v, tf.Tensor) for v in t)

        _rec_idx = rec_idx[0].numpy().item()
        rec_dict = model.tf_dataset.np_dataset[_rec_idx]
        num_snippets = len(rec_dict['split_list'])
        snippet_idx.set_shape([1])
        _snippet_idx = snippet_idx[0].numpy().item()
        assert _snippet_idx < num_snippets

        if _snippet_idx == 0:
            self.rec_idx = _rec_idx
            self.snippet_idx = _snippet_idx

        assert _rec_idx == self.rec_idx

        if _snippet_idx > 0:
            assert _snippet_idx == self.snippet_idx + 1

        self.snippet_idx = _snippet_idx

        results = self.update_states_tf_fn(
            rec_idx=rec_idx,
            ref_notes=ref_notes,
            logits=logits,
            loss=loss
        )

        return results

    @staticmethod
    def octave(distance):

        distance = tf.floor(distance / 12. + .5) * 12.

        return distance


class MetricsValidation(MetricsBase):

    def __init__(self, model):

        super(MetricsValidation, self).__init__(model)

        assert model.config.train_or_inference.inference is None
        assert 'validation' in model.name
        assert not model.is_training

        t = np.arange(.01, 1., .01, dtype=np.float64)
        t = t.astype(np.float32)
        self.voicing_thresholds = t
        self.num_voicing_thresholds = len(self.voicing_thresholds)

        self.var_dict = self.define_tf_variables_fn()
        self.num_frames_vector = tf.convert_to_tensor(model.tf_dataset.num_frames_vector, tf.int64)

    def define_tf_variables_fn(self):

        model = self.model
        num_recs = self.num_recs
        num_ths = self.num_voicing_thresholds

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                all_defined_vars_updated = dict()

                def gen_tf_var(name, shape):

                    assert name
                    assert shape == [num_recs] or shape == [num_recs, num_ths]

                    with tf.device('/cpu:0'):
                        var = tf.Variable(
                            initial_value=tf.zeros(shape, dtype=tf.int64),
                            trainable=False,
                            name=name
                        )
                    var_ref = var.ref()
                    assert var_ref not in all_defined_vars_updated
                    all_defined_vars_updated[var_ref] = False

                    return var

                melody_var_dict = dict(
                    gt=dict(
                        voiced=gen_tf_var('voiced', [num_recs]),
                        unvoiced=gen_tf_var('unvoiced', [num_recs])
                    ),
                    voicing=dict(
                        correct_voiced=gen_tf_var('correct_voiced', [num_recs, num_ths]),
                        incorrect_voiced=gen_tf_var('incorrect_voiced', [num_recs, num_ths]),
                        correct_unvoiced=gen_tf_var('correct_unvoiced', [num_recs, num_ths])
                    ),
                    correct_pitches=dict(
                        wide=gen_tf_var('correct_pitches_wide', [num_recs]),
                        strict=gen_tf_var('correct_pitches_strict', [num_recs, num_ths])
                    ),
                    correct_chromas=dict(
                        wide=gen_tf_var('correct_chromas_wide', [num_recs]),
                        strict=gen_tf_var('correct_chromas_strict', [num_recs, num_ths])
                    )
                )

                batch_counter = tf.Variable(
                    initial_value=tf.zeros([], tf.int32),
                    trainable=False,
                    name='batch_counter'
                )
                all_defined_vars_updated[batch_counter.ref()] = False
                loss = tf.Variable(
                    initial_value=tf.zeros([], tf.float32),
                    trainable=False,
                    name='acc_loss'
                )
                all_defined_vars_updated[loss.ref()] = False

                return dict(
                    melody=melody_var_dict,
                    batch_counter=batch_counter,
                    loss=loss,
                    all_updated=all_defined_vars_updated
                )

    @tf.function(input_signature=[
        tf.TensorSpec([1], 'int32'),
        tf.TensorSpec([1, None], 'float32'),
        tf.TensorSpec([None, 360], 'float32'),
        tf.TensorSpec([], 'float32')
    ], autograph=False)
    def update_states_tf_fn(self, rec_idx, ref_notes, logits, loss):

        num_voicing_thresholds = self.num_voicing_thresholds
        voicing_thresholds = tf.convert_to_tensor(self.voicing_thresholds, tf.float32)
        count_nonzero_fn = self.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)
        loss = tf.convert_to_tensor(loss, tf.float32)

        rec_idx = rec_idx[0]
        ref_notes = tf.squeeze(ref_notes, axis=0)

        est_probs = tf.sigmoid(logits)

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_peak_indices = tf.argmax(est_probs, axis=1, output_type=tf.int32)
        est_peak_probs = tf.gather_nd(est_probs, est_peak_indices[:, None], batch_dims=1)
        est_voicing = est_peak_probs[:, None] > voicing_thresholds[None, :]
        est_voicing.set_shape([None, num_voicing_thresholds])
        n_est_voicing = tf.logical_not(est_voicing)

        est_notes = self.est_notes_fn(est_peak_indices=est_peak_indices, est_probs=est_probs)
        est_ref_note_diffs = tf.abs(est_notes - ref_notes)
        est_ref_note_diffs.set_shape([None])

        def count_nonzero_for_multi_ths_fn(inputs):

            inputs.set_shape([None, num_voicing_thresholds])
            outputs = tf.math.count_nonzero(inputs, axis=0, dtype=tf.int32)
            outputs = tf.cast(outputs, tf.int64)

            return outputs

        voiced_frames = count_nonzero_fn(ref_voicing)
        unvoiced_frames = tf.size(ref_voicing, tf.int64) - voiced_frames
        correct_voiced_frames = tf.logical_and(ref_voicing[:, None], est_voicing)
        correct_voiced_frames.set_shape([None, num_voicing_thresholds])
        correct_voiced_frames = count_nonzero_for_multi_ths_fn(correct_voiced_frames)
        correct_voiced_frames.set_shape([num_voicing_thresholds])
        incorrect_voiced_frames = tf.logical_and(n_ref_voicing[:, None], est_voicing)
        incorrect_voiced_frames = count_nonzero_for_multi_ths_fn(incorrect_voiced_frames)
        correct_unvoiced_frames = tf.logical_and(n_ref_voicing[:, None], n_est_voicing)
        correct_unvoiced_frames = count_nonzero_for_multi_ths_fn(correct_unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'gt', 'voiced', voiced_frames)
        self.update_melody_var_fn(rec_idx, 'gt', 'unvoiced', unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_voiced', correct_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'incorrect_voiced', incorrect_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_unvoiced', correct_unvoiced_frames)

        correct_pitches_wide = est_ref_note_diffs < .5
        correct_pitches_wide = tf.logical_and(ref_voicing, correct_pitches_wide)
        correct_pitches_wide.set_shape([None])
        correct_pitches_strict = tf.logical_and(correct_pitches_wide[:, None], est_voicing)
        correct_pitches_strict.set_shape([None, num_voicing_thresholds])
        correct_pitches_wide = count_nonzero_fn(correct_pitches_wide)
        correct_pitches_strict = count_nonzero_for_multi_ths_fn(correct_pitches_strict)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', 'wide', correct_pitches_wide)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', 'strict', correct_pitches_strict)

        correct_chromas_wide = est_ref_note_diffs
        octave = self.octave(correct_chromas_wide)
        correct_chromas_wide = tf.abs(correct_chromas_wide - octave) < .5
        correct_chromas_wide = tf.logical_and(ref_voicing, correct_chromas_wide)
        correct_chromas_wide.set_shape([None])
        correct_chromas_strict = tf.logical_and(est_voicing, correct_chromas_wide[:, None])
        correct_chromas_strict.set_shape([None, num_voicing_thresholds])
        correct_chromas_wide = count_nonzero_fn(correct_chromas_wide)
        correct_chromas_strict = count_nonzero_for_multi_ths_fn(correct_chromas_strict)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', 'wide', correct_chromas_wide)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', 'strict', correct_chromas_strict)

        self.update_loss_fn(loss)
        self.increase_batch_counter_fn()

        assert all(self.var_dict['all_updated'].values())

    def best_voicing_threshold_fn(self):

        model = self.model
        melody_dict = self.var_dict['melody']
        f8f4div = self.to_f8_divide_and_to_f4_fn
        num_recs = self.num_recs
        num_ths = self.num_voicing_thresholds
        num_frames_vector = self.num_frames_vector

        _num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        tf.debugging.assert_equal(_num_frames_vector, num_frames_vector)
        va = f8f4div(
            numerator=melody_dict['voicing']['correct_voiced'] + melody_dict['voicing']['correct_unvoiced'],
            denominator=num_frames_vector[:, None]
        )
        va.set_shape([num_recs, num_ths])
        va = tf.reduce_mean(va, axis=0)
        best_th_idx = tf.argmax(va)
        best_th = self.voicing_thresholds[best_th_idx]
        var_voicing_threshold = model.config.acoustic_model_ins.voicing_threshold
        assert isinstance(var_voicing_threshold, tf.Variable)
        _best_th = var_voicing_threshold.assign(best_th)
        _best_th = _best_th.numpy()
        assert np.isclose(best_th, _best_th)
        logging.debug('voicing threshold set to {}'.format(_best_th))

        self.current_voicing_threshold = best_th

        return best_th_idx

    def results(self):

        melody_dict = self.var_dict['melody']
        var_loss = self.var_dict['loss']
        var_batch_counter = self.var_dict['batch_counter']
        f8f4div = self.to_f8_divide_and_to_f4_fn
        num_frames_vector = self.num_frames_vector

        best_th_idx = self.best_voicing_threshold_fn()

        m_vrr = f8f4div(
            numerator=melody_dict['voicing']['correct_voiced'][:, best_th_idx],
            denominator=melody_dict['gt']['voiced']
        )
        m_vfa = f8f4div(
            numerator=melody_dict['voicing']['incorrect_voiced'][:, best_th_idx],
            denominator=melody_dict['gt']['unvoiced']
        )
        m_va = f8f4div(
            numerator=(melody_dict['voicing']['correct_voiced'][:, best_th_idx] +
                       melody_dict['voicing']['correct_unvoiced'][:, best_th_idx]),
            denominator=num_frames_vector
        )
        m_rpa_strict = f8f4div(
            numerator=melody_dict['correct_pitches']['strict'][:, best_th_idx],
            denominator=melody_dict['gt']['voiced']
        )
        m_rpa_wide = f8f4div(
            numerator=melody_dict['correct_pitches']['wide'],
            denominator=melody_dict['gt']['voiced']
        )
        m_rca_strict = f8f4div(
            numerator=melody_dict['correct_chromas']['strict'][:, best_th_idx],
            denominator=melody_dict['gt']['voiced']
        )
        m_rca_wide = f8f4div(
            numerator=melody_dict['correct_chromas']['wide'],
            denominator=melody_dict['gt']['voiced']
        )
        m_oa = f8f4div(
            numerator=(melody_dict['correct_pitches']['strict'][:, best_th_idx] +
                       melody_dict['voicing']['correct_unvoiced'][:, best_th_idx]),
            denominator=num_frames_vector
        )
        m_loss = var_loss / tf.cast(var_batch_counter, tf.float32)

        self.oa = tf.reduce_mean(m_oa).numpy().item()
        self.loss = m_loss.numpy().item()

        results = dict(
            loss=m_loss,
            vrr=m_vrr,
            vfa=m_vfa,
            va=m_va,
            rpa_strict=m_rpa_strict,
            rpa_wide=m_rpa_wide,
            rca_strict=m_rca_strict,
            rca_wide=m_rca_wide,
            oa=m_oa
        )

        return results


class MetricsInference(MetricsBase):

    def __init__(self, model):

        super(MetricsInference, self).__init__(model)

        assert model.config.train_or_inference.inference is not None

        self.voicing_threshold = model.config.acoustic_model_ins.voicing_threshold
        assert isinstance(self.voicing_threshold, tf.Variable)

        self.var_dict = self.define_tf_variables_fn()

    def define_tf_variables_fn(self):

        model = self.model
        num_recs = self.num_recs

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                all_defined_vars_updated = dict()

                def gen_tf_var(name):

                    assert name

                    with tf.device('/cpu:0'):
                        var = tf.Variable(
                            initial_value=tf.zeros([num_recs], dtype=tf.int64),
                            trainable=False,
                            name=name
                        )
                    var_ref = var.ref()
                    assert var_ref not in all_defined_vars_updated
                    all_defined_vars_updated[var_ref] = False

                    return var

                melody_var_dict = dict(
                    gt=dict(
                        voiced=gen_tf_var('voiced'),
                        unvoiced=gen_tf_var('unvoiced')
                    ),
                    voicing=dict(
                        correct_voiced=gen_tf_var('correct_voiced'),
                        incorrect_voiced=gen_tf_var('incorrect_voiced'),
                        correct_unvoiced=gen_tf_var('correct_unvoiced')
                    ),
                    correct_pitches=dict(
                        wide=gen_tf_var('correct_pitches_wide'),
                        strict=gen_tf_var('correct_pitches_strict')
                    ),
                    correct_chromas=dict(
                        wide=gen_tf_var('correct_chromas_wide'),
                        strict=gen_tf_var('correct_chromas_strict')
                    )
                )

                batch_counter = tf.Variable(
                    initial_value=tf.zeros([], dtype=tf.int32),
                    trainable=False,
                    name='batch_counter'
                )
                all_defined_vars_updated[batch_counter.ref()] = False
                loss = tf.Variable(
                    initial_value=tf.zeros([], tf.float32),
                    trainable=False,
                    name='acc_loss'
                )
                all_defined_vars_updated[loss.ref()] = False

                return dict(
                    melody=melody_var_dict,
                    batch_counter=batch_counter,
                    loss=loss,
                    all_updated=all_defined_vars_updated
                )

    @tf.function(input_signature=[
        tf.TensorSpec([1], 'int32'),
        tf.TensorSpec([1, None], 'float32'),
        tf.TensorSpec([None, 360], 'float32'),
        tf.TensorSpec([], 'float32')
    ], autograph=False)
    def update_states_tf_fn(self, rec_idx, ref_notes, logits, loss):

        voicing_threshold = self.voicing_threshold
        count_nonzero_fn = self.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)
        loss = tf.convert_to_tensor(loss, tf.float32)

        rec_idx = rec_idx[0]
        ref_notes = tf.squeeze(ref_notes, axis=0)

        est_probs = tf.sigmoid(logits)

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_peak_indices = tf.argmax(est_probs, axis=1, output_type=tf.int32)
        est_peak_probs = tf.gather_nd(est_probs, est_peak_indices[:, None], batch_dims=1)
        est_voicing = est_peak_probs > voicing_threshold
        est_voicing.set_shape([None])
        n_est_voicing = tf.logical_not(est_voicing)

        est_notes = self.est_notes_fn(est_peak_indices=est_peak_indices, est_probs=est_probs)

        est_ref_note_diffs = tf.abs(est_notes - ref_notes)
        est_ref_note_diffs.set_shape([None])

        voiced_frames = count_nonzero_fn(ref_voicing)
        unvoiced_frames = tf.size(ref_voicing, tf.int64) - voiced_frames
        correct_voiced_frames = tf.logical_and(ref_voicing, est_voicing)
        correct_voiced_frames = count_nonzero_fn(correct_voiced_frames)
        incorrect_voiced_frames = tf.logical_and(n_ref_voicing, est_voicing)
        incorrect_voiced_frames = count_nonzero_fn(incorrect_voiced_frames)
        correct_unvoiced_frames = tf.logical_and(n_ref_voicing, n_est_voicing)
        correct_unvoiced_frames = count_nonzero_fn(correct_unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'gt', 'voiced', voiced_frames)
        self.update_melody_var_fn(rec_idx, 'gt', 'unvoiced', unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_voiced', correct_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'incorrect_voiced', incorrect_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_unvoiced', correct_unvoiced_frames)

        correct_pitches_wide = est_ref_note_diffs < .5
        correct_pitches_wide = tf.logical_and(ref_voicing, correct_pitches_wide)
        correct_pitches_strict = tf.logical_and(est_voicing, correct_pitches_wide)
        correct_pitches_wide = count_nonzero_fn(correct_pitches_wide)
        correct_pitches_strict = count_nonzero_fn(correct_pitches_strict)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', 'wide', correct_pitches_wide)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', 'strict', correct_pitches_strict)

        correct_chromas_wide = est_ref_note_diffs
        octave = self.octave(correct_chromas_wide)
        correct_chromas_wide = tf.abs(correct_chromas_wide - octave) < .5
        correct_chromas_wide = tf.logical_and(ref_voicing, correct_chromas_wide)
        correct_chromas_strict = tf.logical_and(est_voicing, correct_chromas_wide)
        correct_chromas_wide = count_nonzero_fn(correct_chromas_wide)
        correct_chromas_strict = count_nonzero_fn(correct_chromas_strict)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', 'wide', correct_chromas_wide)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', 'strict', correct_chromas_strict)

        self.update_loss_fn(loss)
        self.increase_batch_counter_fn()

        assert all(self.var_dict['all_updated'].values())

    def results(self):

        model = self.model
        num_recs = self.num_recs
        melody_dict = self.var_dict['melody']
        var_loss = self.var_dict['loss']
        var_batch_counter = self.var_dict['batch_counter']
        f8f4div = self.to_f8_divide_and_to_f4_fn
        num_frames_vector = tf.convert_to_tensor(model.tf_dataset.num_frames_vector, tf.int64)

        correct_frames = melody_dict['correct_pitches']['strict'] + melody_dict['voicing']['correct_unvoiced']
        _num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        tf.debugging.assert_equal(_num_frames_vector, num_frames_vector)
        m_oa = f8f4div(correct_frames, num_frames_vector)
        m_oa.set_shape([num_recs])

        m_vrr = f8f4div(melody_dict['voicing']['correct_voiced'], melody_dict['gt']['voiced'])
        m_vfa = f8f4div(melody_dict['voicing']['incorrect_voiced'], melody_dict['gt']['unvoiced'])
        m_va = f8f4div(
            melody_dict['voicing']['correct_voiced'] + melody_dict['voicing']['correct_unvoiced'],
            num_frames_vector
        )
        m_rpa_strict = f8f4div(
            melody_dict['correct_pitches']['strict'], melody_dict['gt']['voiced']
        )
        m_rpa_wide = f8f4div(
            melody_dict['correct_pitches']['wide'], melody_dict['gt']['voiced']
        )
        m_rca_strict = f8f4div(
            melody_dict['correct_chromas']['strict'], melody_dict['gt']['voiced']
        )
        m_rca_wide = f8f4div(
            melody_dict['correct_chromas']['wide'], melody_dict['gt']['voiced']
        )
        m_loss = var_loss / tf.cast(var_batch_counter, tf.float32)

        self.oa = tf.reduce_mean(m_oa).numpy().item()
        self.loss = m_loss.numpy().item()
        self.current_voicing_threshold = self.voicing_threshold.read_value().numpy().item()

        results = dict(
            loss=m_loss,
            vrr=m_vrr,
            vfa=m_vfa,
            va=m_va,
            rpa_strict=m_rpa_strict,
            rpa_wide=m_rpa_wide,
            rca_strict=m_rca_strict,
            rca_wide=m_rca_wide,
            oa=m_oa
        )

        return results


class TBSummary:

    def __init__(self, model):

        assert hasattr(model, 'metrics')

        self.model = model
        self.is_inferencing = model.config.train_or_inference.inference is not None

        self.tb_path = os.path.join(model.config.tb_dir, model.name)
        self.tb_summary_writer = tf.summary.create_file_writer(self.tb_path)

        if hasattr(model.tf_dataset, 'rec_names'):
            self.rec_names = model.tf_dataset.rec_names
            self.num_recs = len(self.rec_names)

        self.header = ['vrr', 'vfa', 'va', 'rpa_strict', 'rpa_wide', 'rca_strict', 'rca_wide', 'oa']
        self.num_columns = len(self.header)

        self.table_ins = self.create_tf_table_writer_ins_fn()

    def create_tf_table_writer_ins_fn(self):

        is_inferencing = self.is_inferencing
        model = self.model
        header = self.header
        description = 'metrics'
        tb_summary_writer = self.tb_summary_writer

        if hasattr(self, 'rec_names'):
            assert is_inferencing or not is_inferencing and not model.is_training
            names = list(self.rec_names) + ['average']
            table_ins = ArrayToTableTFFn(
                writer=tb_summary_writer,
                header=header,
                scope=description,
                title=description,
                names=names
            )
        else:
            assert not is_inferencing and model.is_training
            table_ins = ArrayToTableTFFn(
                writer=tb_summary_writer,
                header=header,
                scope=description,
                title=description,
                names=['average']
            )

        return table_ins

    def prepare_table_data_fn(self, result_dict):

        header = self.header

        if hasattr(self, 'rec_names'):

            data = [result_dict[name] for name in header]
            data = tf.stack(data, axis=-1)
            tf.ensure_shape(data, [self.num_recs, self.num_columns])
            ave = tf.reduce_mean(data, axis=0, keepdims=True)
            data = tf.concat([data, ave], axis=0)

        else:
            data = [result_dict[name] for name in header]
            data = [data]
            data = tf.convert_to_tensor(data)

        return data

    def write_tb_summary_fn(self, step_int):

        model = self.model
        is_inferencing = self.is_inferencing

        assert isinstance(step_int, int)

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                result_dict = model.metrics.results()

                if not is_inferencing:
                    with self.tb_summary_writer.as_default():
                        for metric_name in ('loss', 'oa'):
                            value = getattr(model.metrics, metric_name)
                            assert value is not None
                            tf.summary.scalar(metric_name, value, step=step_int)
                        if not model.is_training:
                            value = model.metrics.current_voicing_threshold
                            assert value is not None
                            tf.summary.scalar('voicing_threshold', value, step=step_int)

                else:
                    with self.tb_summary_writer.as_default():
                        loss = model.metrics.loss
                        assert loss is not None
                        tf.summary.text('loss', str(loss), step=step_int)

                data = self.prepare_table_data_fn(result_dict)
                self.table_ins.write(data, step_int)


class Model:

    def __init__(self, config, name):

        assert name in config.model_names

        inferencing = config.train_or_inference.inference is not None

        if not inferencing:
            assert 'test' not in name

        self.name = name
        self.is_training = True if 'train' in name else False
        self.config = config

        if inferencing:
            self.tf_dataset = TFDatasetForInferenceMode(self)
        else:
            if self.is_training:
                self.tf_dataset = TFDatasetForTrainingModeTrainingSplit(self)
            else:
                self.tf_dataset = TFDatasetForInferenceMode(self)

        if inferencing:
            self.metrics = MetricsInference(self)
        else:
            if self.is_training:
                self.metrics = MetricsTrainingModeTrainingSplit(self)
            else:
                self.metrics = MetricsValidation(self)

        self.tb_summary_ins = TBSummary(self)


def main():

    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()
    for name in MODEL_DICT['config'].model_names:
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    aug_info = []
    aug_info.append('tb dir - {}'.format(MODEL_DICT['config'].tb_dir))
    aug_info.append('debug mode - {}'.format(MODEL_DICT['config'].debug_mode))
    aug_info.append('snippet length - {}'.format(MODEL_DICT['config'].snippet_len))
    if MODEL_DICT['config'].train_or_inference.inference is None:
        aug_info.append('batch size - 1')
        aug_info.append('num of batches per epoch - {}'.format(MODEL_DICT['config'].batches_per_epoch))
        aug_info.append('num of patience epochs - {}'.format(MODEL_DICT['config'].patience_epochs))
        aug_info.append('initial learning rate - {}'.format(MODEL_DICT['config'].initial_learning_rate))
    aug_info = '\n\n'.join(aug_info)
    logging.info(aug_info)
    with MODEL_DICT['training'].tb_summary_ins.tb_summary_writer.as_default():
        tf.summary.text('auxiliary_information', aug_info, step=0)

    def training_fn(global_step=None):

        assert isinstance(global_step, int)

        config = MODEL_DICT['config']
        model = MODEL_DICT['training']

        assert config.train_or_inference.inference is None
        assert model.is_training

        iterator = model.tf_dataset.iterator
        acoustic_model = config.acoustic_model_ins
        reg_loss_fn = acoustic_model.reg_l2_fn
        trainable_vars = acoustic_model.trainable_variables
        metrics = model.metrics
        write_tb_summary_fn = model.tb_summary_ins.write_tb_summary_fn
        batches_per_epoch = config.batches_per_epoch
        optimizer = model.config.optimizer
        loss_fn = acoustic_model.loss_tf_fn

        metrics.reset()
        for batch_idx in range(batches_per_epoch):
            logging.debug('batch {}/{}'.format(batch_idx + 1, batches_per_epoch))
            batch = iterator.get_next()
            with tf.GradientTape() as tape:
                logits = acoustic_model(batch['spectrogram'], training=True)
                loss_melody = loss_fn(ref_notes=batch['notes'][0], logits=logits)
                loss_reg = reg_loss_fn()
                total_loss = loss_melody + loss_reg
            grads = tape.gradient(total_loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            metrics.update_states(ref_notes=batch['notes'], logits=logits, loss=loss_melody)
        write_tb_summary_fn(global_step)

        loss = model.metrics.loss
        oa = model.metrics.oa
        v_th = model.metrics.current_voicing_threshold
        logging.info('{} - step - {} - loss - {} - oa - {} - voicing threshold - {}'.format(model.name, global_step, loss, oa, v_th))

    def inference_fn(model_name, global_step=None):

        config = MODEL_DICT['config']

        is_inferencing = config.train_or_inference.inference is not None

        if not is_inferencing:
            assert model_name == 'validation'

        assert isinstance(global_step, int)

        model = MODEL_DICT[model_name]

        acoustic_model = config.acoustic_model_ins
        assert not hasattr(model.tf_dataset, 'iterator')
        iterator = iter(model.tf_dataset.tf_dataset)
        metrics = model.metrics
        batches_per_epoch = len(model.tf_dataset.rec_start_end_idx_list)
        loss_fn = acoustic_model.loss_tf_fn

        metrics.reset()
        for batch_idx in range(batches_per_epoch):

            batch = iterator.get_next()
            logits = acoustic_model(batch['spectrogram'], training=False)
            loss = loss_fn(ref_notes=batch['notes'][0], logits=logits)
            metrics.update_states(
                rec_idx=batch['rec_idx'],
                snippet_idx=batch['snippet_idx'],
                ref_notes=batch['notes'],
                logits=logits,
                loss=loss
            )
        batch = iterator.get_next_as_optional()
        assert not batch.has_value()

        model.tb_summary_ins.write_tb_summary_fn(global_step)

        loss = model.metrics.loss
        oa = model.metrics.oa
        v_th = model.metrics.current_voicing_threshold
        logging.info('{} - step - {} - loss - {} - oa - {} - voicing threshold - {}'.format(model.name, global_step, loss, oa, v_th))

    if MODEL_DICT['config'].train_or_inference.inference is not None:
        ckpt_file = MODEL_DICT['config'].train_or_inference.inference
        ckpt_dir, ckpt_name = os.path.split(ckpt_file)
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
        ckpt = tf.train.Checkpoint(model=MODEL_DICT['config'].acoustic_model_ins.model_for_ckpt)
        status = ckpt.restore(ckpt_file)
        status.expect_partial()
        status.assert_existing_objects_matched()

        logging.info('inferencing ... ')
        for model_name in MODEL_DICT['config'].model_names:
            logging.info(model_name)
            inference_fn(model_name, global_step=0)
            MODEL_DICT[model_name].tb_summary_ins.tb_summary_writer.close()

    elif MODEL_DICT['config'].train_or_inference.from_ckpt is not None:
        ckpt = tf.train.Checkpoint(
            model=MODEL_DICT['config'].acoustic_model_ins.model_for_ckpt,
            optimizer=MODEL_DICT['config'].optimizer
        )
        ckpt_file = MODEL_DICT['config'].train_or_inference.from_ckpt
        ckpt_dir, ckpt_name = os.path.split(ckpt_file)
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
        status = ckpt.restore(ckpt_file)
        assert status.assert_consumed()
        logging.info('reproducing results ...')

        model_name = 'validation'
        logging.info(model_name)
        inference_fn(model_name, global_step=0)
        best_oa = MODEL_DICT[model_name].metrics.oa
        assert best_oa is not None
        best_epoch = 0
    else:
        logging.info('training from scratch ...')
        best_oa = None

    # training
    if MODEL_DICT['config'].train_or_inference.inference is None:

        assert MODEL_DICT['config'].train_or_inference.ckpt_prefix is not None
        assert 'ckpt_manager' not in MODEL_DICT
        ckpt = tf.train.Checkpoint(
            model=MODEL_DICT['config'].acoustic_model_ins.model_for_ckpt,
            optimizer=MODEL_DICT['config'].optimizer
        )
        ckpt_dir, ckpt_prefix = os.path.split(MODEL_DICT['config'].train_or_inference.ckpt_prefix)
        assert ckpt_prefix != ''
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
        ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            directory=ckpt_dir,
            max_to_keep=1,
            checkpoint_name=ckpt_prefix
        )
        MODEL_DICT['ckpt_manager'] = ckpt_manager

        patience_epochs = MODEL_DICT['config'].patience_epochs
        training_epoch = 1

        while True:

            logging.info('\nepoch - {}'.format(training_epoch))

            for model_name in MODEL_DICT['config'].model_names:

                logging.info(model_name)

                if 'train' in model_name:
                    training_fn(training_epoch)
                elif 'validation' in model_name:
                    inference_fn(model_name, training_epoch)

            valid_oa = MODEL_DICT['validation'].metrics.oa
            should_save = best_oa is None or best_oa < valid_oa
            if should_save:
                best_oa = valid_oa
                best_epoch = training_epoch
                save_path = MODEL_DICT['ckpt_manager'].save(checkpoint_number=training_epoch)
                logging.info('weights checkpointed to {}'.format(save_path))

            d = training_epoch - best_epoch
            if d >= patience_epochs:
                logging.info('training terminated at epoch {}'.format(training_epoch))
                break

            training_epoch = training_epoch + 1

        for model_name in MODEL_DICT['config'].model_names:
            model = MODEL_DICT[model_name]
            model.tb_summary_ins.tb_summary_writer.close()


if __name__ == '__main__':

    main()

