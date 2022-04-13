from posixpath import split
from mido import MidiFile, MidiTrack
from mido import tick2second
from tqdm import tqdm

import os
from glob import glob
from argparse import ArgumentParser

parser = ArgumentParser(
    description='''
        make midi file to wav file
    '''
)
parser.add_argument(
    '-input_dir', '--input_dir',
    required=True, type=str, help='midi file directory'
)
parser.add_argument(
    '-output_dir', '--output_dir',
    required=True, type=str, help='output directory for wav files'
)

def split_music(mid_path, output_path):
    mid = MidiFile(mid_path)
    o_file = MidiFile()

    for i, track in enumerate(mid.tracks):
        total_time = 0

        introTrack = MidiTrack()

        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            total_time  += tick2second(msg.time, mid.ticks_per_beat, tempo)
            
            introTrack.append(msg)

            if total_time > 40:
                break

        o_file.tracks.append(introTrack)

    o_file.save(output_path)

## main
args = parser.parse_args()

midi_files = sorted( glob(os.path.join(args.input_dir, '*')) )
output_path = os.path.join(args.output_dir)
os.makedirs(output_path, exist_ok=True)

print("split start!")

for file in tqdm(midi_files):
    f_name = file.split("/")[-1]
    output_file = os.path.join(output_path, f_name)

    split_music(file, output_file)

print("Done!")
