from cmath import log
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, create_pop909_datasets

from model.music_transformer import MusicTransformer

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.argument_funcs import parse_eval_args, print_eval_args
from utilities.run_model import eval_model

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Evaluates a model specified by command line arguments
    ----------
    """

    args = parse_eval_args()
    print_eval_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    # Test dataset
    if args.interval and args.octave:
        classic_train, classic_val, classic_test = create_epiano_datasets('dataset/octave_interval_e_piano/', args.max_sequence, interval = args.interval, octave = args.octave)
        pop909_dataset = create_pop909_datasets('dataset/pop_pickle/', args.max_sequence, interval = True)
    elif args.logscale:
        print("logscale dataset 0420")
        classic_train, classic_val, classic_test = create_epiano_datasets('dataset/logscale_epiano0420/', args.max_sequence, interval = args.interval, octave = args.octave, absolute=args.absolute, logscale=args.logscale)
        pop909_dataset = create_pop909_datasets('dataset/logscale_pop0420/', args.max_sequence, interval = args.interval, absolute=args.absolute, logscale=args.logscale)
    elif args.interval and not args.octave:
        classic_train, classic_val, classic_test = create_epiano_datasets('dataset/logscale_e_piano/', args.max_sequence, interval = args.interval, octave = args.octave)
        pop909_dataset = create_pop909_datasets('dataset/pop_pickle/', args.max_sequence, interval = True)
    elif args.octave and args.fusion_encoding and args.absolute:
        print("absolute dataset")
        classic_train, classic_val, classic_test = create_epiano_datasets('dataset/octave_fusion_absolute_e_piano/', args.max_sequence,
                                                                          interval=args.interval, octave=args.octave)
        pop909_dataset = create_pop909_datasets('dataset/pop909_absolute', args.max_sequence, octave=True)
    elif args.octave and args.fusion_encoding:
        print("octave_fusion dataset")
        classic_train, classic_val, classic_test = create_epiano_datasets('dataset/octave_fusion_e_piano/', args.max_sequence,
                                                                          interval=args.interval, octave=args.octave)
        pop909_dataset = create_pop909_datasets('dataset/pop_pickle/', args.max_sequence, octave=True)
    elif not args.interval and args.octave:
        print("octave dataset!")
        classic_train, classic_val, classic_test = create_epiano_datasets('dataset/octave_e_piano/', args.max_sequence, interval = args.interval, octave = args.octave)
        pop909_dataset = create_pop909_datasets('dataset/pop_pickle/', args.max_sequence, octave = True)
    else:
        classic_train, classic_val, classic_test = create_epiano_datasets('dataset/e_piano/', args.max_sequence)
        pop909_dataset = create_pop909_datasets('dataset/pop_pickle/', args.max_sequence)

    pop_train, pop_valid, pop_test = torch.utils.data.random_split(pop909_dataset, [int(len(pop909_dataset) * 0.8),
                                                                                    int(len(pop909_dataset) * 0.1),
                                                                                    len(pop909_dataset) - int(
                                                                                        len(pop909_dataset) * 0.8) - int(
                                                                                        len(pop909_dataset) * 0.1)],
                                                                   generator=torch.Generator().manual_seed(42))

    if args.data == 'both':
        print("Dataset: both")
        test_dataset = torch.utils.data.ConcatDataset([ classic_test, pop_test])
    elif args.data == 'classic':
        print("Dataset: classic")
        test_dataset = torch.utils.data.ConcatDataset([ classic_test])
    else:
        print("Dataset: pop")
        test_dataset = torch.utils.data.ConcatDataset([pop_test])

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr, condition_token = args.condition_token, interval = args.interval, octave = args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale = args.logscale).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    # No smoothed loss
    if args.interval and args.octave:
        loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_OCTAVE_INTERVAL)
    elif args.interval and not args.octave:
        loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_INTERVAL)
    elif not args.interval and args.octave:
        loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_OCTAVE)
    elif args.octave and args.fusion_encoding and args.absolute:
        loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_OCTAVE_FUSION_ABSOLUTE)
    elif args.octave and args.fusion_encoding:
        loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_OCTAVE_FUSION)
    elif args.logscale:
        loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_RELATIVE)
    else:
        loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    print("Evaluating:")
    model.eval()

    avg_loss, avg_acc = eval_model(model, test_loader, loss, args)

    print("Avg loss:", avg_loss)
    print("Avg acc:", avg_acc)
    print(SEPERATOR)
    print("")


if __name__ == "__main__":
    main()
