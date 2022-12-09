# Largely copied from https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from avcv.all import *
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .segment_utils import *
from .segment import *


def to_bbox(a,b):
    return torch.Tensor([a, 0, b, 1])[None]
def get_trellis(emission, tokens, blank_id=109):
    num_frame = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

# Merge words
def merge_words(segments, separator):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def get_w2v():
    w2v_processor = Wav2Vec2Processor.from_pretrained("./pretrained/processor/")
    w2vmodel = Wav2Vec2ForCTC.from_pretrained("./pretrained/w2vmodel/").requires_grad_(False)
    return dict(
        w2vmodel=w2vmodel,
        w2v_processor=w2v_processor,
        text_encode = w2v_processor.tokenizer.encode,
        text_decode = w2v_processor.tokenizer.decode,
        
    )


from .segment import Segment


def merge_repeats(path, transcript):
    
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


# audio_files['test']


def preproc_w2v_input(item_audio_label,
                     separator=None, lower=False):
    text = []
    for word, _, _ in item_audio_label.words:
            if lower: word=word.lower()
            text.append(word)
    
    transcript = separator.join(text)
    
    audio = item_audio_label.audio
    sample_rate = item_audio_label.sample_rate
    assert sample_rate == 16000
    input_values = torch.from_numpy(audio)
    
    return dict(
        audio=audio,
        sample_rate=sample_rate,
        transcript=transcript,
        input_values=input_values,
        file_name=get_name(item_audio_label.path),
    )




def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]






def forwad_w2v(w2vmodel, item, sep, text_encode, blank_id=109, device='cuda'):
    with torch.inference_mode():
        logits = emission = w2vmodel(item['input_values'][None].to(device)).logits[0]
        emission = torch.log_softmax(emission, dim=-1).cpu()
        
    tokens = text_encode(item['transcript'])#[dictionary[c] for c in transcript]
    trellis = get_trellis(emission, tokens, blank_id=blank_id)
    path = backtrack(trellis, emission, tokens, blank_id=blank_id)
    segments = merge_repeats(path, item['transcript'])
    word_segments = merge_words(segments, sep)
    ratio = item['input_values'].size(0) / (emission.size(0) - 1)
    return dict(word_segments=word_segments, ratio=ratio, trellis=trellis, 
        segments=segments, emission=emission, logits=logits, path=path, 
    )

def force_align(logits, transcript, sep, text_encode, blank_id=109, device='cuda', ratio=320):
    emission = torch.log_softmax(logits, dim=-1).cpu()
        
    tokens = text_encode(transcript)#[dictionary[c] for c in transcript]
    trellis = get_trellis(emission, tokens, blank_id=blank_id)
    path = backtrack(trellis, emission, tokens, blank_id=blank_id)
    segments = merge_repeats(path, transcript)
    word_segments = merge_words(segments, sep)

    return dict(word_segments=word_segments, ratio=ratio, trellis=trellis, 
        segments=segments, emission=emission, logits=logits, path=path, 
    )

def get_ctc_loss(item, res):
    target = torch.tensor(item['tokens']).unsqueeze(0).cpu()
    target[target==46] = 109
    input = torch.log_softmax(res['logits'], 1).cpu()
    input = input.unsqueeze(1)
    input_lengths = torch.full(size=(target.size(0),), fill_value=input.size(0), dtype=torch.long)
    target_lengths = torch.full(size=(target.size(0),), fill_value=target.size(1), dtype=torch.long)
    return F.ctc_loss(input, target, input_lengths, target_lengths, blank=109, )
    
    


class W2vForceAligner:
    def __init__(self, ckpt=None, device='cpu'):
        w2vmeta = get_w2v()
        self.devide = device
        self.w2vmodel = w2vmeta['w2vmodel'].to(self.devide)
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
            state_dict = {k[6:]:v for k,v in state_dict.items()}
            res = self.w2vmodel.load_state_dict(state_dict)
            logger.info('{} Loaded {}', res, ckpt)
        self.text_encode = w2vmeta['text_encode']
        self.text_decode = w2vmeta['text_decode']
        
    def __call__(self, item_audio_label,  blank_id=109, separator="|"):
        item_w2v = preproc_w2v_input(item_audio_label, separator=separator)
        res = forwad_w2v(self.w2vmodel, item_w2v, separator, text_encode=self.text_encode, device=self.devide, blank_id=blank_id)
        return res


def get_word_iou(seg1, seg2):
    assert isinstance(seg1, list) or isinstance(seg1, tuple)
    assert isinstance(seg2, list) or isinstance(seg2, tuple)
    pbox = to_bbox(*seg1[-2:])
    gbox = to_bbox(*seg2[-2:]) 
    giou = generalized_box_iou(pbox, gbox).item()
    return giou

