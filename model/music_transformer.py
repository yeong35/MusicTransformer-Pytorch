import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR


# MusicTransformer
class MusicTransformer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Music Transformer reproduction from https://arxiv.org/abs/1809.04281. Arguments allow for
    tweaking the transformer architecture (https://arxiv.org/abs/1706.03762) and the rpr argument
    toggles Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class with dummy decoder to
    make a decoder-only transformer architecture

    For RPR support, there is modified Pytorch 1.2.0 code in rpr.py. Modified source will be
    kept up to date with Pytorch revisions only as necessary.
    ----------
    """

    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False, condition_token=False, interval = False, octave = False, fusion = False, absolute=False, logscale=False):
        super(MusicTransformer, self).__init__()

        self.dummy      = DummyDecoder()

        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq    = max_sequence
        self.rpr        = rpr
        self.condition_token = condition_token
        self.interval   = interval
        self.octave = octave
        self.fusion = fusion
        self.absolute = absolute
        self.logscale = logscale

        # Input embedding
        if not self.condition_token and not interval and not octave:
            self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        elif self.condition_token and not interval and not octave:
            self.embedding = nn.Embedding(CONDITION_VOCAB_SIZE, self.d_model)
        elif not self.condition_token and interval and not octave:
            self.embedding = nn.Embedding(VOCAB_SIZE_INTERVAL, self.d_model)
        elif self.condition_token and interval and not octave:
            self.embedding = nn.Embedding(CONDITION_VOCAB_SIZE_INTERVAL, self.d_model)
        elif condition_token and octave and fusion and absolute:
            self.embedding = nn.Embedding(CONDITION_VOCAB_SIZE_OCTAVE_FUSION_ABSOLUTE, self.d_model)
        elif not condition_token and octave and fusion and absolute:
            self.embedding = nn.Embedding(VOCAB_SIZE_OCTAVE_FUSION_ABSOLUTE, self.d_model)
        elif condition_token and octave and fusion:
            self.embedding = nn.Embedding(CONDITION_VOCAB_SIZE_OCTAVE_FUSION, self.d_model)
        elif not condition_token and octave and fusion:
            self.embedding = nn.Embedding(VOCAB_SIZE_OCTAVE_FUSION, self.d_model)
        elif not self.condition_token and not interval and octave:
            self.embedding = nn.Embedding(VOCAB_SIZE_OCTAVE, self.d_model)
        elif self.condition_token and not interval and octave:
            self.embedding = nn.Embedding(CONDITION_VOCAB_SIZE_OCTAVE, self.d_model)
        elif not condition_token and interval and octave:
            self.embedding = nn.Embedding(VOCAB_SIZE_OCTAVE_INTERVAL, self.d_model)
        elif logscale:
            self.embedding = nn.Embedding(VOCAB_SIZE_RELATIVE, self.d_model)
        else:
            self.embedding = nn.Embedding(CONDITION_VOCAB_SIZE_OCTAVE_INTERVAL, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        # Base transformer
        if(not self.rpr):
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy
            )
        # RPR Transformer
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder
            )

        # Final output is a softmaxed linear layer
        if interval and octave:
            self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE_OCTAVE_INTERVAL)
        elif interval and not octave:
            self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE_INTERVAL)
        elif octave and fusion and absolute:
            self.Wout = nn.Linear(self.d_model, VOCAB_SIZE_OCTAVE_FUSION_ABSOLUTE)
        elif octave and fusion:
            self.Wout = nn.Linear(self.d_model, VOCAB_SIZE_OCTAVE_FUSION)
        elif not interval and octave:
            self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE_OCTAVE)
        elif logscale:
            self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE_RELATIVE)
        else:
            self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)

    # forward
    def forward(self, x, mask=True):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """

        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None

        # 처음에 (batch, sequence 들어옴)
        x = self.embedding(x)               # (batch, sequence, d_model)

        # Input shape is (max_seq, batch_size, d_model)

        x = x.permute(1,0,2)                # (sequence, batch, d_model)
        x = self.positional_encoding(x)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1,0,2)        # (sequence, batch, vocab)

        y = self.Wout(x_out)                # (batch, sequence, vocab)
        # y = self.softmax(y)

        del mask

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y

    # generate
    def generate(self, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0,
                 condition_token=False, interval=False, octave=False, fusion=False, absolute=False, logscale=False, topp=0):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Generates midi given a primer sample. Music can be generated using a probability distribution over
        the softmax probabilities (recommended) or by using a beam search.
        ----------
        """

        assert (not self.training), "Cannot generate while in training mode"

        print("Generating sequence of max length:", target_seq_length)
        if interval and octave:
            gen_seq = torch.full((1, target_seq_length), TOKEN_PAD_OCTAVE_INTERVAL, dtype=TORCH_LABEL_TYPE, device=get_device())
        elif interval and not octave:
            gen_seq = torch.full((1, target_seq_length), TOKEN_PAD_INTERVAL, dtype=TORCH_LABEL_TYPE, device=get_device())
        elif octave and fusion and absolute:
            gen_seq = torch.full((1, target_seq_length), TOKEN_PAD_OCTAVE_FUSION_ABSOLUTE, dtype=TORCH_LABEL_TYPE, device=get_device())
        elif octave and fusion:
            gen_seq = torch.full((1, target_seq_length), TOKEN_PAD_OCTAVE_FUSION, dtype=TORCH_LABEL_TYPE, device=get_device())
        elif not interval and octave:
            gen_seq = torch.full((1, target_seq_length), TOKEN_PAD_OCTAVE, dtype=TORCH_LABEL_TYPE, device=get_device())
        elif logscale:
            gen_seq = torch.full((1, target_seq_length), TOKEN_PAD_RELATIVE, dtype=TORCH_LABEL_TYPE, device=get_device())
        else:
            gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())


        # print("primer:",primer)
        # print(gen_seq)
        cur_i = num_primer
        while(cur_i < target_seq_length):
            # gen_seq_batch     = gen_seq.clone()
            if interval and octave:
                y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END_OCTAVE_INTERVAL]
            elif interval and not octave:
                y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END_INTERVAL]
            elif octave and fusion and absolute:
                y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END_OCTAVE_FUSION_ABSOLUTE]
            elif octave and fusion:
                y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END_OCTAVE_FUSION]
            elif not interval and octave:
                y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END_OCTAVE]
            elif logscale:
                y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END_RELATIVE]
            else:
                y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END]

            token_probs = y[:, cur_i-1, :]

            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)

            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                if interval and octave:
                    beam_rows = top_i // VOCAB_SIZE_OCTAVE_INTERVAL
                    beam_cols = top_i % VOCAB_SIZE_OCTAVE_INTERVAL
                elif interval and not octave:
                    beam_rows = top_i // VOCAB_SIZE_INTERVAL
                    beam_cols = top_i % VOCAB_SIZE_INTERVAL
                elif octave and fusion and absolute:
                    beam_rows = top_i // VOCAB_SIZE_OCTAVE_FUSION_ABSOLUTE
                    beam_cols = top_i % VOCAB_SIZE_OCTAVE_FUSION_ABSOLUTE
                elif octave and fusion:
                    beam_rows = top_i // VOCAB_SIZE_OCTAVE_FUSION
                    beam_cols = top_i % VOCAB_SIZE_OCTAVE_FUSION
                elif not interval and octave:
                    beam_rows = top_i // VOCAB_SIZE_OCTAVE
                    beam_cols = top_i % VOCAB_SIZE_OCTAVE
                elif logscale:
                    beam_rows = top_i // VOCAB_SIZE_RELATIVE
                    beam_cols = top_i % VOCAB_SIZE_RELATIVE
                else:
                    beam_rows = top_i // VOCAB_SIZE
                    beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:

                distrib = torch.distributions.categorical.Categorical(probs=token_probs)

                if topp == 0:   # 기본 설정
                    next_token = distrib.sample()
                else:           # top_p의 p값을 설정
                    distrib, index_list = distrib.probs.sort(descending=True)   # 내림차순으로 정렬

                    p_value = torch.tensor(topp)
                    p_index_list = []
                    p_prob_list = []

                    for dis, ind in zip(distrib[0], index_list[0]):
                        p_prob_list.append(dis.detach().cpu().numpy().tolist())
                        p_index_list.append(ind.detach().cpu().numpy().tolist())

                        if sum(p_prob_list) > p_value:  # 누적확률값보다 커지면 하나씩만 빼주자
                            # p_index_list.pop()
                            # p_prob_list.pop()
                            break

                    # 위에서 만든 분포는 합이 1이 아니기 때문에, 아래와같이 하여 합하면 1인 분포를 만들어준다
                    new_prob_dist = torch.distributions.categorical.Categorical(probs=torch.tensor([p_prob_list]).to(get_device()))
                    next_token = p_index_list[new_prob_dist.sample()]

                    # 누적 확률 안의 index 중 하나를 선택함
                    next_token = torch.tensor([next_token]).to(get_device())

                # print("next token:",next_token)
                gen_seq[:, cur_i] = next_token


                # Let the transformer decide to end if it wants to
                if interval and octave:
                    if (next_token == TOKEN_END_OCTAVE_INTERVAL):
                        print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                        break
                elif interval and not octave:
                    if(next_token == TOKEN_END_INTERVAL):
                        print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                        break
                elif octave and fusion and absolute:
                    if (next_token == TOKEN_END_OCTAVE_FUSION_ABSOLUTE):
                        print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                        break

                elif octave and fusion:
                    if (next_token == TOKEN_END_OCTAVE_FUSION):
                        print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                        break
                elif not interval and octave:
                    if (next_token == TOKEN_END_OCTAVE):
                        print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                        break
                elif logscale:
                    if (next_token == TOKEN_END_RELATIVE):
                        print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                        break
                else:
                    if(next_token == TOKEN_END):
                        print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                        break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)
        if condition_token:
            # condition_token 제외하고 return
            return gen_seq[:, 1:cur_i]
        else:
            return gen_seq[:, :cur_i]

# Used as a dummy to nn.Transformer
# DummyDecoder
class DummyDecoder(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only
    architecture (stacked encoders with dummy decoder fits the bill)
    ----------
    """

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Returns the input (memory)
        ----------
        """

        return memory
