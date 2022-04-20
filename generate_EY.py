import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi, decode_midi_JE, encode_midi_JE

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, create_pop909_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

# main
def main():
    """
    ----------
    Author: Damon Gwinn, EY
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """
    
    # argument parsing
    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    # 저장할 directory 만들기
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+'/classic/', exist_ok=True)
    os.makedirs(args.output_dir+'/pop/', exist_ok=True)

    # Grabbing dataset if needed
    # pickle file path - EY
    classic_path = 'dataset/logscale_epiano0420/'
    pop_path = 'dataset/logscale_pop0420/'

    # train, val, test
    classic_train, classic_eval, classic_test = create_epiano_datasets(classic_path, args.num_prime, random_seq=False, condition_token=args.condition_token, interval = args.interval, octave=args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale=args.logscale, label = 0)
    pop_dataset = create_pop909_datasets(pop_path, args.num_prime, random_seq=False, condition_token=args.condition_token, interval = args.interval, octave=args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale=args.logscale, label = 1)

    # dataset 지정
    classic_dataset = [classic_train, classic_eval, classic_test]
    pop_dataset = [pop_dataset]
    dataset_folder = ['train/', 'val/', 'test/']

    # Can be None, an integer index to dataset, or a file path
    # if(args.primer_file is None):
    #     f = str(random.randrange(len(dataset)))
    # else:
    #     f = args.primer_file

    # if(f.isdigit()):
    #     idx = int(f)
    #     primer, _, _  = dataset[idx]
    #     primer = primer.to(get_device())
    #
    #     print("Using primer index:", idx, "(", dataset.data_files[idx], ")")
    #
    # else:
    #     raw_mid = encode_midi(f)
    #     if(len(raw_mid) == 0):
    #         print("Error: No midi messages in primer file:", f)
    #         return
    #
    #     primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False)
    #     primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())
    #
    #     print("Using primer file:", f)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr, condition_token=args.condition_token, interval = args.interval, octave=args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale=args.logscale).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))


    # GENERATION
    model.eval()

    # classic generation
    if args.data != 'pop':
        for dataset, folder in zip(classic_dataset, dataset_folder):

            classic_index_list = list(range(len(dataset)))
            folder_name_length = len(folder)

            for classic_index in classic_index_list:
                primer, _, _ = dataset[classic_index]
                primer = primer.to(get_device())
                print("Using primer index:", classic_index, "(", dataset.data_files[classic_index], ")")


                # # Saving primer first
                # f_path = os.path.join(args.output_dir+'/classic/', f"primer_{dataset.data_files[classic_index][len(classic_path)+folder_name_length:]}.mid")
                # decode_midi_JE(primer[:args.num_prime].cpu().numpy(), file_path=f_path, interval=args.interval, octave=args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale=args.logscale)

                # Genearte music and save file
                print("RAND DIST")
                rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0, topp=args.topp, condition_token=args.condition_token, interval = args.interval, octave=args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale = args.logscale)

                f_path = os.path.join(
                    args.output_dir+'/classic/', f"rand_{dataset.data_files[classic_index][len(classic_path)+folder_name_length:]}.mid")

                try:
                    if args.octave or args.interval or args.fusion_encoding:
                        decode_midi_JE(rand_seq[0].cpu().numpy(), file_path=f_path, interval=args.interval, octave=args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale = args.logscale)
                    else:
                        decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)
                except:
                    print("생성 안되는디")
                    continue

    # pop generation
    if args.data != 'classic':
        for dataset in pop_dataset:

            pop_index_list = list(range(len(dataset)))

            for pop_index in pop_index_list:
                primer, _, _ = dataset[pop_index]
                primer = primer.to(get_device())

                print("Using primer index:", pop_index, "(", dataset.data_files[pop_index], ")")

                # Genearte music and save file
                print("RAND DIST")
                rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0, topp=args.topp, condition_token=args.condition_token, interval = args.interval, octave=args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale = args.logscale)

                f_path = os.path.join(
                    args.output_dir+'/pop/', f"rand_{dataset.data_files[pop_index][len(pop_path):]}.mid")
            
                try:
                    if args.octave or args.interval or args.fusion_encoding:
                        decode_midi_JE(rand_seq[0].cpu().numpy(), file_path=f_path, interval=args.interval, octave=args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale = args.logscale)
                    else:
                        decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)
                except:
                    print("생성 안되는디")
                    continue

    # with torch.set_grad_enabled(False):
    #     if(args.beam > 0):
    #         print("BEAM:", args.beam)
    #         beam_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)
    # 
    #         f_path = os.path.join(
    #             args.output_dir, f"beam_{args.primer_file}_{args.model_weights}.mid")
    #         decode_midi(beam_seq[0].cpu().numpy(), file_path=f_path)
    #     else:
    #         print("RAND DIST")
    #         rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)
    # 
    #         f_path = os.path.join(
    #             args.output_dir, f"rand_{args.primer_file}_{args.model_weights}.mid")
    #         decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)




if __name__ == "__main__":
    main()
