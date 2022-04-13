import argparse
import os
import pickle

import third_party.midi_processor.processor as midi_processor

# prep_midi
def prep_midi(maestro_root, output_dir, interval=False, logscale=False, octave = False, fusion=False):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pre-processes the maestro dataset, putting processed midi data (train, eval, test) into the
    given output folder
    ----------
    """

    output_path = os.path.join(output_dir)
    os.makedirs(output_path, exist_ok=True)

    midi_list = os.listdir(maestro_root)

    for piece in midi_list:
        mid         = os.path.join(maestro_root, piece)
        f_name      = mid.split("/")[-1] + ".pickle"

        o_file = os.path.join(output_path, f_name)

        if interval or octave or fusion or logscale:
            prepped = midi_processor.encode_midi_JE(mid, interval=interval, logspace=logscale, octave=octave, fusion=fusion)
        else:
            prepped = midi_processor.encode_midi(mid)

        if len(prepped)==0:
            print(piece)
            exit()

        o_stream = open(o_file, "wb")
        pickle.dump(prepped, o_stream)
        o_stream.close()

    return True



# parse_args
def parse_args():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Parses arguments for preprocess_midi using argparse
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("maestro_root", type=str, help="Root folder for the Maestro dataset")
    parser.add_argument("-output_dir", type=str, default="./dataset/e_piano", help="Output folder to put the preprocessed midi into")

    return parser.parse_args()

# main
def main():
    """
    ----------
    Author: Damon Gwinn, EY
    ----------
    Entry point. Preprocesses maestro and saved midi to specified output folder.
    ----------
    """

    args            = parse_args()
    maestro_root    = args.maestro_root
    output_dir      = args.output_dir

    ##====================##
    interval = True
    logscale = True
    octave = False
    absolute = True
    fusion = False
    ##====================##

    # python preprocess_all_midi.py "maestro_root" -output_dir "output_dir_path"

    print("maestro root :", maestro_root)
    print("Preprocessing midi files and saving to", output_dir)



    prep_midi(maestro_root, output_dir, interval = interval, logscale=logscale, octave = octave, fusion = fusion, absolute=absolute)
    print("Done!")
    print("")

if __name__ == "__main__":
    main()
