# from transformers.file_utils import cached_path, hf_bucket_url
import os
import os.path as osp
import zipfile
from dataclasses import dataclass
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as FA
import torchaudio.transforms as T
import whisper
from IPython.display import Audio, display
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .segment_utils import *


def get_name(path):
    """
    Get name of file without extension
    """
    return osp.splitext(osp.basename(path))[0]


w2v_processor = Wav2Vec2Processor.from_pretrained("./pretrained/processor/")
w2v_tokenizer = w2v_processor.tokenizer
wtokenizer = whisper.tokenizer.get_tokenizer(True,
                                             language="vi",
                                             task='transcribe')

from .segment import *
from .w2v_aligner import merge_words


def play_audio(waveform, sample_rate=16000):
    """
    Play audio in notebook
    """
    if waveform.shape[0] > 100:
        waveform = waveform[None]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError(
            "Waveform with more than 2 channels are not supported.")


# @memoize
# def get_audio_len(audio_file):
#     target_rate = 16000
#     _audio, current_sample_rate = torchaudio.load(audio_file)
#     _audio = FA.resample(_audio, current_sample_rate,
#                          target_rate)
#     return _audio.shape[-1] / target_rate

# @memoize
# def fast_get_audio_len(audio_file):
#     try:
#         return get_audio_len(audio_file)
#     except:
#         return -1


def display_segment(waveform,
                    word,
                    next_word=None,
                    sample_rate=16000,
                    ratio=320):
    """
        Display segment of audio
    """
    x0 = int(ratio * word.start)
    if next_word is not None:
        x1 = int(ratio * next_word.start)
    else:
        x1 = int(ratio * word.end)
    print(
        f"{word.label} ({word.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec"
    )
    segment = waveform[:, x0:x1]
    play_audio(segment)


def display_segment_with_time(waveform, start, end, sample_rate=16000):
    """
        Display segment of audio
    """
    x0 = int(sample_rate * start)
    x1 = int(sample_rate * end)
    segment = waveform[x0:x1]
    play_audio(segment, sample_rate=sample_rate)


# @imemoize
def load_audio(path):
    import torchaudio.functional as FA
    _audio, _sample_rate = torchaudio.load(path)
    _audio = FA.resample(_audio, _sample_rate, 16000)
    return _audio[0].numpy()


class ItemAudioLabel:
    """
    Pair audio and label, use for training/data augmentation/ sanity check/ statistic .....
    
    """
    def __init__(self,
                 path,
                 audio_file=None, # Assume audio file in the corresonding "songs" directory with mp3 or wav extension
                 model_type='detection',
                 word_score=None,
                 spliter=' ', # Use for split label into words in w2v forced alignment
                 is_training=True):
        self.path = path
        self.model_type = model_type
        self.spliter = spliter
        self.is_training = is_training

        if word_score is not None:
            self._word_score = word_score

        if audio_file is None:
            assert isinstance(path, str) and osp.exists(path), path
            audio_file = path.replace('/labels/',
                                      '/songs/').replace('.json', '.wav')
            if not osp.exists(audio_file):
                audio_file = path.replace('/labels/',
                                          '/songs/').replace('.json', '.mp3')

        assert osp.exists(audio_file), audio_file

        self.audio_file = audio_file
        self.name = get_name(audio_file)

    @property
    def data(self):
        if not hasattr(self, '_data'):
            path = self.path
            if isinstance(self.path, str):
                self._data = mmcv.load(path) if isinstance(path, str) else path
            else:
                self._data = self.path
        return self._data

    @property
    def audio(self):
        if not hasattr(self, '_audio'):
            self._sample_rate = 16000
            self._audio = load_audio(self.audio_file)
        return self._audio

    @property
    def giou(self):
        if not hasattr(self, '_giou'):
            path = self.audio_file.replace('/songs/',
                                           '/precomputed_giou/')[:-3] + 'pkl'
            if not osp.exists(path):
                raise Exception('Please run tools/precompute_giou.py first')
            
            w2v_segments, self._giou = mmcv.load(path)
            self.w2v_segments = [Segment(*_) for _ in w2v_segments]
        return np.array(self._giou)

    @property
    def transcript(self):
        """
            Join all words in transcript into a string
        """
        return self.spliter.join([_[0] for _ in self.words])

    @property
    def word_score(self):
        """ W2v score of each word in transcript

        Returns:
            List[float]: score of each word
        """
        if not hasattr(self, '_word_score'):
            self.giou
            self._word_score = [
                word.score for word in merge_words(self.w2v_segments, '|')
            ]
        return self._word_score

    def get_words_meta(self):
        text, start, end = list(zip(*self.words))
        # Example 
        # text:('Xin', 'có', 'một', 'lần', 'uống', 'chén', 'muộn', 'phiền', 'Nhà', 'im', 'đứng', 'cửa', 'cài', 'đóng', 'then', 'Vườn', 'mưa', 'xuống', 'hành', 'lang', 'tối', 'tăm', 'Về', 'thôi', 'nhé,', 'cổng', 'chào', 'cuối', 'sân')
        # start:(0.0, 0.421, 1.011, 1.36, 2.821, 3.561, 3.941, 4.631, 8.75, 9.411, 9.79, 10.98, 11.72, 12.0, 12.371, 14.5, 15.04, 15.49, 16.591, 17.35, 17.661, 18.161, 20.14, 20.47, 20.81, 22.18, 22.82, 23.13, 23.55)
        # end: (0.421, 1.011, 1.36, 1.571, 3.071, 3.941, 4.571, 4.731, 9.381, 9.73, 10.94, 11.72, 12.0, 12.301, 13.29, 15.04, 15.19, 16.171, 17.131, 17.651, 18.151, 18.361, 20.47, 20.81, 22.13, 22.74, 23.13, 23.55, 23.94)
        import ipdb; ipdb.set_trace()
        if self.is_training:
            ret = self.encode_for_detection(text,
                                       start,
                                       end,
                                       self.word_score,
                                       self.giou,
                                       mode_token=False)
        else:
            ret = self.encode_for_detection(text, start, end, mode_token=False)
        return ret

    @property
    def mel(self):
        try:
            return self._mel
        except:
            self._mel = whisper.log_mel_spectrogram(self.audio).numpy()
            return self._mel

    @property
    def sample_rate(self):
        if not hasattr(self, '_sample_rate'):
            self.audio

        self._sample_rate
        return self._sample_rate

    def play(self, by='line', with_giou=False):
        word_split_ids = [0]
        for i, line in enumerate(self.data):
            split_id = word_split_ids[-1] + len(line['l'])
            word_split_ids.append(split_id)
        if by == 'word':
            for i, (word, s, e) in enumerate(self.words):
                if e <= s:
                    e = s + 0.1
                print(word, s, e)
                display_segment_with_time(self.audio, s, e, self.sample_rate)
        else:
            for a, b in zip(word_split_ids[:-1], word_split_ids[1:]):
                ws = self.words[a:b]
                print(ws)
                s = ws[0][1]
                e = ws[-1][2]
                display_segment_with_time(self.audio, s, e, self.sample_rate)

    @property
    def words(self):
        if not hasattr(self, '_words'):
            out = ()
            words = []
            for i, line in enumerate(self.data):
                start = line['s'] / 1000
                end = line['e'] / 1000
                if end <= start:
                    end = start + 0.1
                words += [(word_meta['d'], word_meta['s'] / 1000,
                           word_meta['e'] / 1000) for word_meta in line['l']]
            self._words = words
        return self._words

    @property
    def audio_len(self):
        return fast_get_audio_len(self.audio_file)

    @property
    def w2v_tokens(self):
        if not hasattr(self, '_w2v_tokens'):
            self._w2v_tokens = w2v_tokenizer.encode(self.transcript)
        return self._w2v_tokens


    def encode_for_detection(self, texts,
                            starts,
                            ends,
                            word_scores=None,
                            gious=None,
                            mode_token=False):
        """ Custom input for detection

        Args:
            texts (str): Raw input text
            starts (_type_): _description_
            ends (_type_): _description_
            word_scores (_type_, optional): _description_. Defaults to None.
            gious (_type_, optional): _description_. Defaults to None.
            mode_token (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        tokens = []
        decode_position = []
        bboxes = []
        loss_scale = []
        
        if word_scores is None:
            word_scores = [1] * len(texts)
            gious = [1] * len(texts)

        for i, (s, t, e, word_score,
                giou) in enumerate(zip(starts, texts, ends, word_scores, gious)):

            tt = wtokenizer.encode(' ' + t)
            tokens += [*tt]
            if not mode_token:
                decode_position += [len(tokens)]
                bboxes.append(to_bbox_cxcywh(s, e))
                if len(t.split(' ')) > 1:
                    _loss_scale = 0
                elif word_score < 0.1 or giou < 0.1:
                    _loss_scale = 0
                else:
                    _loss_scale = 1

                loss_scale.append(_loss_scale)

        tokens = tokens + [wtokenizer.eot]
        if mode_token:
            return tokens
        return dict(
            dec_pos=decode_position,
            bboxes=bboxes,
            tokens=tokens,
            loss_scale=np.array(loss_scale),
        )
def get_json_paths(txt_split):
    """ Get path to json file from txt_split

    Args:
        txt_split: path to txt file

    Returns:
        list: list of path to json file
    """
    return open(txt_split).read().splitlines()
