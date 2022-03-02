import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi, decode_midi_JE, encode_midi_JE

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False, interval=args.interval,
                                               octave=args.octave)
    # Can be None, an integer index to dataset, or a file path
    if(args.primer_file is None):
        f = str(random.randrange(len(dataset)))
    else:
        f = args.primer_file

    if(f.isdigit()):
        idx = int(f)
        primer, _, _  = dataset[idx]
        primer = primer.to(get_device())
        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")

    else:
        if args.interval and args.octave:
            raw_mid = encode_midi_JE(f, interval = args.interval, octave = args.octave)
        elif args.interval and not args.octave:
            raw_mid = encode_midi_JE(f, interval = args.interval, octave = args.octave)
        elif not args.interval and args.octave:
            raw_mid = encode_midi_JE(f, interval = args.interval, octave = args.octave)
        else:
            raw_mid = encode_midi(f)

        if(len(raw_mid) == 0):
            print("Error: No midi messages in primer file:", f)
            return

        primer, _  = process_midi(raw_mid, args.num_prime, random_seq = False, condition_token = args.condition_token, interval = args.interval, octave = args.octave)
        primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

        print("Using primer file:", f)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr, condition_token = args.condition_token, interval = args.interval, octave = args.octave).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    # Saving primer first
    f_path = os.path.join(args.output_dir, f"primer_{args.primer_file}.mid")
    if args.interval:
        decode_midi_JE(primer[:args.num_prime].cpu().numpy(), file_path=f_path)
    elif args.octave:
        decode_midi_JE(primer[:args.num_prime].cpu().numpy(), file_path=f_path, octave = True)
    else:
        decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        if(args.beam > 0):
            print("BEAM:", args.beam)
            beam_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam, condition_token=args.condition_token, interval = args.interval, octave = args.octave)

            f_path = os.path.join(
                args.output_dir, f"beam_{args.primer_file}_{args.model_weights}.mid")
            if not args.interval:
                decode_midi(beam_seq[0].cpu().numpy(), file_path=f_path)
            else:
                decode_midi_JE(beam_seq[0].cpu().numpy(), file_path=f_path)
        else:
            print("RAND DIST")
            rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0, condition_token=args.condition_token, interval = args.interval, octave = args.octave, topp = args.topp)

            f_path = os.path.join(
                args.output_dir, f"rand_{args.primer_file}.mid")
                # args.output_dir, f"rand_{args.primer_file}_{args.model_weights}.mid")
            if args.interval:
                decode_midi_JE(rand_seq[0].cpu().numpy(), file_path=f_path)
            elif args.octave:
                decode_midi_JE(rand_seq[0].cpu().numpy(), file_path=f_path, octave = True)
            else:
                decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)


    print("Done!", f_path)




if __name__ == "__main__":
    main()
