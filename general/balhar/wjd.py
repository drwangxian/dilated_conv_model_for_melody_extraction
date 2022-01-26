# mean oa - 0.730197805102494

import os
import json
import sqlite3
import numpy as np
import soundfile
import librosa
import mir_eval


def validity_check_of_ref_freqs_fn(freqs):

    min_melody_freq = librosa.midi_to_hz(23.6)

    all_zeros = freqs == 0.
    all_positives = freqs > min_melody_freq
    all_valid = np.logical_or(all_zeros, all_positives)
    assert np.all(all_valid)


def gen_ref_fn(pitches, onsets, offsets):

    sr = 44100
    h = 256
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


def read_est_fn(csv_file):

    sr = 44100
    h = 256
    base_name = os.path.basename(csv_file)
    if base_name == "KennyDorham_Doodlin'_Solo.csv":
        times_freqs = np.genfromtxt(csv_file, delimiter=None)
    else:
        times_freqs = np.genfromtxt(csv_file, delimiter=',')
    t1 = times_freqs[1, 0] - times_freqs[0, 0]
    t2 = times_freqs[-1, 0] - times_freqs[-2, 0]
    t1 = np.int(np.round(t1 * sr))
    t2 = np.int(np.round(t2 * sr))
    assert t1 == t2 == h

    return times_freqs[:, 0], times_freqs[:, 1]


split_file = os.path.join(os.environ['wjazzd'], 'jiri_wjazzd_split.json')
with open(split_file, 'rb') as fh:
    split_dict = json.load(fh)
assert len(split_dict['test']) == 74
test_solos = split_dict['test']

db_file = os.environ['wjazzd']
db_file = os.path.join(db_file, 'wjazzd.db')
with sqlite3.connect(db_file) as con:

    cur = con.cursor()

    est_melody_dir = '/media/hd/datasets/jiri_melody_output/melody-outputs/hcnn_ctx-f0-outputs/wjazzd-test-melody-outputs'
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

        est_csv_file = os.path.join(est_melody_dir, filename + '_Solo.csv')
        est_times, est_freqs = read_est_fn(est_csv_file)

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









