import torch

from third_party.midi_processor.processor import RANGE_NOTE_ON, RANGE_NOTE_OFF, RANGE_VEL, RANGE_TIME_SHIFT, DURATION, NOTE_CHANGE, START_PITCH, OCTAVE_RANGE, PITCH_RANGE, OCTAVE_RANGE_INTERVAL, PITCH_RANGE_INTERVAL, NOTE_ON_RELATIVE, DURATION_RELATIVE, TIME_SHIFT_RELATIVE

SEPERATOR               = "========================="

# Taken from the paper
ADAM_BETA_1             = 0.9
ADAM_BETA_2             = 0.98
ADAM_EPSILON            = 10e-9

LR_DEFAULT_START        = 1.0
SCHEDULER_WARMUP_STEPS  = 4000
# LABEL_SMOOTHING_E       = 0.1

# DROPOUT_P               = 0.1

TOKEN_END               = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD               = TOKEN_END + 1

CONDITION_CLASSIC       = TOKEN_PAD + 1
CONDITION_POP           = CONDITION_CLASSIC + 1

VOCAB_SIZE              = TOKEN_PAD + 1
CONDITION_VOCAB_SIZE    = CONDITION_POP + 1

# interval token
TOKEN_END_INTERVAL               = NOTE_CHANGE + DURATION + RANGE_TIME_SHIFT + RANGE_VEL
TOKEN_PAD_INTERVAL               = TOKEN_END_INTERVAL + 1

CONDITION_CLASSIC_INTERVAL       = TOKEN_PAD_INTERVAL + 1
CONDITION_POP_INTERVAL           = CONDITION_CLASSIC_INTERVAL + 1

VOCAB_SIZE_INTERVAL              = TOKEN_PAD_INTERVAL + 1
CONDITION_VOCAB_SIZE_INTERVAL    = CONDITION_POP_INTERVAL + 1

# octave_interavl token
TOKEN_END_OCTAVE_INTERVAL            = OCTAVE_RANGE_INTERVAL + PITCH_RANGE_INTERVAL + DURATION + RANGE_TIME_SHIFT + RANGE_VEL
TOKEN_PAD_OCTAVE_INTERVAL            = TOKEN_END_OCTAVE_INTERVAL + 1

CONDITION_CLASSIC_OCTAVE_INTERVAL    = TOKEN_PAD_OCTAVE_INTERVAL + 1
CONDITION_POP_OCTAVE_INTERVAL        = CONDITION_CLASSIC_OCTAVE_INTERVAL + 1

VOCAB_SIZE_OCTAVE_INTERVAL           = TOKEN_PAD_OCTAVE_INTERVAL + 1
CONDITION_VOCAB_SIZE_OCTAVE_INTERVAL = CONDITION_POP_OCTAVE_INTERVAL + 1

# octave token
TOKEN_END_OCTAVE            = OCTAVE_RANGE + PITCH_RANGE + DURATION + RANGE_TIME_SHIFT + RANGE_VEL
TOKEN_PAD_OCTAVE            = TOKEN_END_OCTAVE + 1

CONDITION_CLASSIC_OCTAVE    = TOKEN_PAD_OCTAVE + 1
CONDITION_POP_OCTAVE        = CONDITION_CLASSIC_OCTAVE + 1

VOCAB_SIZE_OCTAVE           = TOKEN_PAD_OCTAVE + 1
CONDITION_VOCAB_SIZE_OCTAVE = CONDITION_POP_OCTAVE + 1

# octave_fusion token
TOKEN_END_OCTAVE_FUSION            = OCTAVE_RANGE + PITCH_RANGE_INTERVAL + DURATION + RANGE_TIME_SHIFT + RANGE_VEL
TOKEN_PAD_OCTAVE_FUSION            = TOKEN_END_OCTAVE_FUSION + 1

CONDITION_CLASSIC_OCTAVE_FUSION    = TOKEN_PAD_OCTAVE_FUSION + 1
CONDITION_POP_OCTAVE_FUSION        = CONDITION_CLASSIC_OCTAVE_FUSION + 1

VOCAB_SIZE_OCTAVE_FUSION           = TOKEN_PAD_OCTAVE_FUSION + 1
CONDITION_VOCAB_SIZE_OCTAVE_FUSION = CONDITION_POP_OCTAVE_FUSION + 1

# octave_fusion_absolute token
TOKEN_END_OCTAVE_FUSION_ABSOLUTE            = OCTAVE_RANGE+PITCH_RANGE+OCTAVE_RANGE + PITCH_RANGE_INTERVAL + DURATION + RANGE_TIME_SHIFT + RANGE_VEL
TOKEN_PAD_OCTAVE_FUSION_ABSOLUTE            = TOKEN_END_OCTAVE_FUSION_ABSOLUTE + 1

CONDITION_CLASSIC_OCTAVE_FUSION_ABSOLUTE    = TOKEN_PAD_OCTAVE_FUSION_ABSOLUTE + 1
CONDITION_POP_OCTAVE_FUSION_ABSOLUTE        = CONDITION_CLASSIC_OCTAVE_FUSION_ABSOLUTE + 1

VOCAB_SIZE_OCTAVE_FUSION_ABSOLUTE           = TOKEN_PAD_OCTAVE_FUSION_ABSOLUTE + 1
CONDITION_VOCAB_SIZE_OCTAVE_FUSION_ABSOLUTE = CONDITION_POP_OCTAVE_FUSION_ABSOLUTE + 1

# absolute_relative
TOKEN_END_RELATIVE = RANGE_NOTE_ON + DURATION_RELATIVE + TIME_SHIFT_RELATIVE + RANGE_VEL    # logscale encoding
TOKEN_PAD_RELATIVE = TOKEN_END_RELATIVE + 1

VOCAB_SIZE_RELATIVE = TOKEN_PAD_RELATIVE + 1

# etc
TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long

PREPEND_ZEROS_WIDTH     = 4