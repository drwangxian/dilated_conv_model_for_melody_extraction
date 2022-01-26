"""
the results given by jiri have problems:
1. timing is incorrect and inconsistent across melody outputs for different tracks.
For example, in KidOry_GutBucketBlues_Solo.txt, hop size in secs = 0.009998086518934093 and thus hop size in samples = 440.9156154849935.
However in MichaelBrecker_NeverAlone_Solo.txt, hop size in secs = 0.009999737424661077 and thus hop size in samples = 440.9884204275535.
My analysis:
1) jiri used a hop size of 441 samples, so the correct hop size in secs = .01 and hop size in samples = 441
   jiri referred to the code snippet for melody detection give at https://essentia.upf.edu/essentia_python_examples.html
   when generating times for each frame. The line of code for generating times at the above address is as follows:
   pitch_times = numpy.linspace(0.0,len(audio)/44100.0,len(pitch_values) )

   The correct one should be:
   pitch_times = numpy.arange(num_frames) * (441. / 44100.)
   This explains the tiny difference among the hop sizes given in different melody timings.
2) When directly running mir_eval against the melody results give by jiri, the average oa is 0.608136712412, far lower than the result of 0.667 given
    in jiri's paper. This may due to jiri used different tracks when testing salamon's algorithm on wjazzd.
    If the times of jiri's results are adjusted to have hop size of 0.01 sec, the average oa is 0.611625824762.
3) hop size of 441 samples is not the optimal hop size recommended by salamon, instead it is 128.
3) Given these complex reasons, we decided to run the test again:
    1) tracks: 74 tracks
    2) hop size: 128 samples
    3) average oa: 0.642325724685

"""

import os
import json
import sqlite3
import numpy as np
import soundfile
import librosa
import mir_eval
import essentia
import essentia.standard as es


def validity_check_of_ref_freqs_fn(freqs):

    min_melody_freq = librosa.midi_to_hz(23.6)

    all_zeros = freqs == 0.
    all_positives = freqs > min_melody_freq
    all_valid = np.logical_or(all_zeros, all_positives)
    assert np.all(all_valid)


def gen_ref_fn(pitches, onsets, offsets):

    sr = 44100
    h = 128
    half_h = h // 2

    def sec_to_frame_fn(time, onset):

        sample = time * sr

        if onset:
            sample = np.int(np.ceil(sample))
        else:
            sample = np.int(sample)

        frame = (sample + half_h) // h

        return frame

    num_frames = 1 + sec_to_frame_fn(offsets[-1], onset=False)
    freqs = np.zeros([num_frames], np.float64)

    for pitch, onset, offset in zip(pitches, onsets, offsets):
        assert pitch > 10.
        onset_frame = sec_to_frame_fn(onset, True)
        offset_frame = 1 + sec_to_frame_fn(offset, False)
        assert offset_frame > onset_frame
        freq = librosa.midi_to_hz(pitch)
        freqs[onset_frame:offset_frame] = freq

    validity_check_of_ref_freqs_fn(freqs)
    times = np.arange(num_frames) * (h / float(sr))

    return times, freqs


class EstMelody:

    def __init__(self):

        fn = es.EqualLoudness()
        self.equal_loudness_fn = fn

        fn = es.PredominantPitchMelodia()
        self.es_melody_fn = fn

    @staticmethod
    def load_wav_fn(track_id):

        sr = 44100

        wav_file = os.path.join(os.environ['wjazzd'], 'audios', track_id + '_Solo.wav')
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

        return times, freqs


split_file = os.path.join(os.environ['wjazzd'], 'jiri_wjazzd_split.json')
with open(split_file, 'rb') as fh:
    split_dict = json.load(fh)
assert len(split_dict['test']) == 74
test_solos = split_dict['test']

db_file = os.environ['wjazzd']
db_file = os.path.join(db_file, 'wjazzd.db')
est_melody_ins = EstMelody()
melody_fn = est_melody_ins.melody_fn
with sqlite3.connect(db_file) as con:

    cur = con.cursor()

    oas = []
    for filename_idx, filename in enumerate(test_solos):

        print(filename_idx)

        if filename == 'PaulDesmond_BlueRondoALaTurk':
            filename_solo = 'PaulDesmond_BlueRondoAlaTurk'
        else:
            filename_solo = filename
        filename_solo = filename_solo + '_Solo'

        cur.execute("select melid from transcription_info where filename_solo = \"{}\"".format(filename_solo))
        melid = cur.fetchall()
        assert len(melid) == 1
        melid = melid[0][0]
        cur.execute("select melid, eventid, pitch, onset, duration from melody where melid = {}".format(melid))
        melodies = cur.fetchall()
        assert len(melodies)
        melodies = np.asarray(melodies, dtype=[
            ('melid', np.int32),
            ('eventid', np.int64),
            ('pitch', np.float32),
            ('onset', np.float64),
            ('duration', np.float64)
        ])
        assert np.all(melodies['melid'] == melid)
        t = np.diff(melodies['eventid']) == 1
        assert np.all(t)
        t = np.diff(melodies['onset']) >= 0.
        assert np.all(t)
        t = melodies['duration'] > 0.
        assert np.all(t)
        pitches = melodies['pitch']
        onsets = melodies['onset']
        offsets = melodies['onset'] + melodies['duration']

        wav_file = os.path.join(os.environ['wjazzd'], 'audios', filename + '_Solo.wav')
        assert os.path.isfile(wav_file)

        wav_info = soundfile.info(wav_file)
        assert wav_info.samplerate == 44100
        wav_duration = wav_info.duration
        melody_duration = offsets[-1]
        assert wav_duration >= melody_duration

        ref_times, ref_freqs = gen_ref_fn(pitches, onsets, offsets)

        est_times, est_freqs = melody_fn(filename)

        oa = mir_eval.melody.evaluate(
            ref_time=ref_times,
            ref_freq=ref_freqs,
            est_time=est_times,
            est_freq=est_freqs
        )['Overall Accuracy']
        print(oa)
        oas.append(oa)
oas = np.asarray(oas)
oa = np.mean(oas)
print('mean oa - {}'.format(oa))









