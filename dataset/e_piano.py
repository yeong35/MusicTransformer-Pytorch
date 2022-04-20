import os
import pickle
import random
from numpy import logspace
import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utilities.constants import *
from utilities.device import cpu_device

SEQUENCE_START = 0

# EPianoDataset
class EPianoDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.
    ----------
    """

    def __init__(self, root, max_seq=2048, random_seq=False, condition_token=False, interval = False, octave = False, fusion=False, absolute=False, logscale=False, label=0):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq
        self.condition_token = condition_token
        self.interval = interval
        self.octave = octave
        self.fusion = fusion
        self.absolute = absolute
        self.logscale = logscale

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]
        self.label = [label] * len(self.data_files)

    # __len__
    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        """

        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.

        Returns the input and the target.
        ----------
        """

        # All data on cpu to allow for the Dataloader to multithread
        i_stream    = open(self.data_files[idx], "rb")
        # return pickle.load(i_stream), None
        raw_mid     = torch.tensor(pickle.load(i_stream), dtype=TORCH_LABEL_TYPE, device=cpu_device())
        i_stream.close()

        x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq,
                              condition_token = self.condition_token, interval = self.interval, octave = self.octave, fusion=self.fusion, absolute=self.absolute, logscale=self.logscale, label = self.label)

        return x, tgt, torch.tensor(self.label[idx])

# process_midi
def process_midi(raw_mid, max_seq, random_seq, condition_token=False, interval = False, octave = False, fusion=False, absolute=False, logscale=False, label = 0):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """
    if interval and octave:
        x   = torch.full((max_seq, ), TOKEN_PAD_OCTAVE_INTERVAL, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        tgt = torch.full((max_seq, ), TOKEN_PAD_OCTAVE_INTERVAL, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    elif interval and not octave:
        x   = torch.full((max_seq, ), TOKEN_PAD_INTERVAL, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        tgt = torch.full((max_seq, ), TOKEN_PAD_INTERVAL, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    elif octave and fusion and absolute:
        x = torch.full((max_seq,), TOKEN_PAD_OCTAVE_FUSION_ABSOLUTE, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        tgt = torch.full((max_seq,), TOKEN_PAD_OCTAVE_FUSION_ABSOLUTE, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    elif octave and fusion:
        x = torch.full((max_seq,), TOKEN_PAD_OCTAVE_FUSION, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        tgt = torch.full((max_seq,), TOKEN_PAD_OCTAVE_FUSION, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    elif not interval and octave:
        x   = torch.full((max_seq, ), TOKEN_PAD_OCTAVE, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        tgt = torch.full((max_seq, ), TOKEN_PAD_OCTAVE, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    elif logscale:
        x   = torch.full((max_seq, ), TOKEN_PAD_RELATIVE, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        tgt = torch.full((max_seq, ), TOKEN_PAD_RELATIVE, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    else:
        x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        tgt = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt

    if(raw_len < full_seq):

        if interval and logscale and absolute:
            
            start_pitch = -1
            last_pitch = -1
       
            data_temp = numpy.array([])

            for token in raw_mid:
                token_cpu = token.cpu().detach().numpy()
                if token_cpu in range(128, 128+255):
                    if start_pitch == -1:
                        start_pitch = token_cpu - 127
                        last_pitch = token_cpu -127
                        token_cpu = 127

                        data_temp = numpy.append(start_pitch, data_temp)    # 앞에 절대음 토큰
                        
                    else:
                        token_cpu = (token_cpu-last_pitch)+127
                        last_pitch = last_pitch + token_cpu - 127
                        data_temp = numpy.append(data_temp, token_cpu)
                else:
                    data_temp = numpy.append(data_temp, token_cpu)

            raw_mid = torch.tensor(data_temp[:], dtype=TORCH_LABEL_TYPE, device=cpu_device())

        x[:raw_len]         = raw_mid
        tgt[:raw_len-1]     = raw_mid[1:]
        if interval and octave:
            tgt[raw_len]    = TOKEN_END_OCTAVE_INTERVAL
        elif interval and not octave:
            tgt[raw_len]    = TOKEN_END_INTERVAL
        elif octave and fusion and absolute:
            tgt[raw_len] = TOKEN_END_OCTAVE_FUSION_ABSOLUTE
        elif octave and fusion:
            tgt[raw_len] = TOKEN_END_OCTAVE_FUSION
        elif not interval and octave:
            tgt[raw_len]    = TOKEN_END_OCTAVE
        elif logscale:
            tgt[raw_len]    = TOKEN_END_RELATIVE
        else:
            tgt[raw_len]    = TOKEN_END
            
    else:
        # Randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]
        # 음차 만들어주기
        if interval and logscale and absolute:
            
            start_pitch = -1
            last_pitch = -1

            data_temp = numpy.array([])

            for token in data:
                token_cpu = token.cpu().detach().numpy()
                if token_cpu in range(128, 128+255):
                    if start_pitch == -1:
                        start_pitch = token_cpu - 127
                        last_pitch = token_cpu -127
                        token_cpu = 127
                        data_temp = numpy.append(start_pitch, data_temp)    # 앞에 절대음 토큰
                        
                    else:
                        token_cpu = (token_cpu-last_pitch)+127
                        last_pitch = last_pitch + token_cpu - 127
                        data_temp = numpy.append(data_temp, token_cpu)
                else:
                    data_temp = numpy.append(data_temp, token_cpu)
                
                data_temp = numpy.append(data_temp, token_cpu)
            data = torch.tensor(data_temp, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        # condition_token이 true면 label에 따라 조건코드를 추가해주자
        if condition_token:
            if label == 0:
                data = torch.tensor(CONDITION_CLASSIC) + raw_mid[start:end]
            elif label == 1:
                data = torch.tensor(CONDITION_POP) + raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]

    # print("x:",x)
    # print("tgt:",tgt)

    return x, tgt


# create_epiano_datasets
def create_epiano_datasets(dataset_root, max_seq, random_seq=False, condition_token=False, interval = False, logscale = False, octave = False, fusion=False, absolute=False, label = 0):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Creates train, evaluation, and test EPianoDataset objects for a pre-processed (preprocess_midi.py)
    root containing train, val, and test folders.
    ----------
    """

    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EPianoDataset(train_root, max_seq, random_seq,
                                  condition_token = condition_token, interval = interval, octave = octave, fusion=fusion, absolute=absolute, logscale = logscale, label = label)
    val_dataset = EPianoDataset(val_root, max_seq, random_seq,
                                condition_token = condition_token, interval = interval, octave = octave, fusion=fusion, absolute=absolute, logscale = logscale, label = label)
    test_dataset = EPianoDataset(test_root, max_seq, random_seq,
                                 condition_token = condition_token, interval = interval, octave = octave, fusion=fusion, absolute=absolute, logscale = logscale, label = label)

    return train_dataset, val_dataset, test_dataset

def create_pop909_datasets(dataset_root, max_seq, random_seq=False, condition_token=False, interval = False, octave = False, fusion=False, absolute=False, logscale = False, label = 1):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Creates train, evaluation, and test EPianoDataset objects for a pre-processed (preprocess_midi.py)
    root containing train, val, and test folders.
    ----------
    """


    pop909_dataset = EPianoDataset(dataset_root, max_seq, random_seq,
                                   condition_token = condition_token, interval = interval, octave = octave, fusion=fusion, absolute=absolute, logscale=logscale, label =  label)

    return pop909_dataset

# compute_epiano_accuracy
def compute_epiano_accuracy(out, tgt, interval = False, octave = False, fusion=False, absolute=False, logscale = False):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Computes the average accuracy for the given input and output batches. Accuracy uses softmax
    of the output.
    ----------
    """

    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    if interval and octave:
        mask = (tgt != TOKEN_PAD_OCTAVE_INTERVAL)
    elif not interval and octave:
        mask = (tgt != TOKEN_PAD_OCTAVE)
    elif octave and fusion and absolute:
        mask = (tgt != TOKEN_PAD_OCTAVE_FUSION_ABSOLUTE)
    elif octave and fusion:
        mask = (tgt != TOKEN_PAD_OCTAVE_FUSION)
    elif interval and not octave:
        mask = (tgt != TOKEN_PAD_INTERVAL)
    elif logscale:
        mask = (tgt != TOKEN_PAD_RELATIVE)
    else:
        mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    # Empty
    if(len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)

    return acc
