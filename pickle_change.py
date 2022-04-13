import argparse
from code import interact
import os
import pickle
import json
from isort import file

import pretty_midi

import third_party.midi_processor.processor as midi_processor


def pickle2other_encoding(p_file, interval=False, logscale=False, octave=False, fusion=False, absolute = False):

    with open(p_file, "rb") as fr:
        data = pickle.load(fr)

    event_sequence = [midi_processor.Event.from_int(idx) for idx in data]
    snote_seq = midi_processor._event_seq2snote_seq(event_sequence)
    note_seq = midi_processor._merge_note(snote_seq)
    note_seq.sort(key=lambda x:x.start)

    notes  = []
    events = []
    last_pitch = -1

    cur_time = 0
    cur_vel = 0

    for snote in note_seq:
        
        events += midi_processor._make_time_sift_events(prev_time=cur_time, post_time=snote.start, logscale=logscale)
        
        if last_pitch == -1 and interval:
            events += midi_processor._snote2events_JE(snote=snote, prev_vel=cur_vel, duration=snote.end-snote.start, pitch=0, logscale=logscale, octave = octave)
        elif interval:
            if not octave:  # interval encoding
                events += midi_processor._snote2events_JE(snote=snote, prev_vel=cur_vel, duration=snote.end-snote.start, pitch=(snote.pitch-last_pitch)+127, logscale=logscale, octave = octave, interval = interval)
            else:           # octave_interval encoding
                events += midi_processor._snote2events_JE(snote=snote, prev_vel=cur_vel, duration=snote.end-snote.start, pitch=snote.pitch-last_pitch, logscale=logscale, octave = octave, interval = interval)
        else:               # octave encoding
            events += midi_processor._snote2events_JE(snote=snote, prev_vel=cur_vel, duration=snote.end-snote.start, pitch=snote.pitch, logscale=logscale, octave = octave, interval = interval)
        cur_time = snote.start
        cur_vel = snote.velocity
        
        last_pitch=snote.pitch

    if absolute and octave:
        events = [midi_processor.Event(event_type='octave', value=note_seq[0].pitch//12), midi_processor.Event(event_type='absolute_pitch', value=note_seq[0].pitch%12)] + events
    elif absolute:
        events = [midi_processor.Event(event_type='absolute_note_on', value=note_seq[0].pitch)] + events

    return [e.to_int_JE(octave=octave, interval=interval, fusion=fusion, absolute = absolute) for e in events]

def pop_pickle2dataset(file_root, output_dir, logscale=False, octave=False, interval=False, fusion=False, absolute = False):

    # 저장할 dir생성
    output_path = os.path.join(output_dir)
    os.makedirs(output_path, exist_ok=True)

    pickle_list = os.listdir(file_root)

    for piece in pickle_list:
        mid = os.path.join(file_root, piece)
        f_name = mid.split("/")[-1]

        o_file = os.path.join(output_path, f_name)

        prepped = pickle2other_encoding(mid, interval=interval, logscale=logscale, octave=octave, fusion=fusion, absolute=absolute)

        if len(prepped) == 0:
            print(piece)
            exit()
        
        o_stream = open(o_file, "wb")
        pickle.dump(prepped, o_stream)
        o_stream.close()




if __name__ == '__main__':

    octave = False
    interval = True
    fusion = False
    absolute = True
    logscale = True

    pickle_dataset = '/home/bang/PycharmProjects/MusicGeneration/MusicTransformer-Pytorch/dataset/pop_pickle'
    output_dir = '/home/bang/PycharmProjects/MusicGeneration/MusicTransformer-Pytorch/dataset/relative_pop909'

    encoded = pop_pickle2dataset(pickle_dataset, output_dir, logscale=logscale, octave=octave, interval=interval, fusion = fusion, absolute=absolute)

    print("Done!")