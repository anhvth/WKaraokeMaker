# Author: AnhVo\
# This script is used to precompute w2v alignment for all training data, it is required to run before training,
# if you only use this project for inference, ignore this script

import argparse
import os
import os.path as osp

from avcv.all import *
from kmaker.data import *
from kmaker.w2v_aligner import *
from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor,
                          WhisperTokenizer)


def precompute_w2v(json_path, w2v_aligner, audio_file=None):
    """ Precompute w2v alignment for a single audio/label pair, caculate the word segments, giou and save to disk
        public/private test sample has no real start/end time value -> giou = -1
    Args:
        json_path (str): Path to json file
        w2v_aligner (W2vForceAligner): W2vForceAligner object
        audio_file (str, optional): Path to audio file. Defaults to None.
    Returns:
        None
    """
    name = get_name(json_path)
    item_audio_label = ItemAudioLabel(json_path, audio_file=audio_file)
    cache_path = item_audio_label.audio_file.replace('/songs/', '/precomputed_giou/')[:-3]+'pkl'
    if not osp.exists(cache_path):
        gious = []
        outputs = w2v_aligner(item_audio_label, separator='|')
        pred_segments = merge_words(outputs['segments'], '|')
        pred_words = [segment_to_word(segment) for segment in pred_segments]
        for pword, gword in zip(pred_words, item_audio_label.words):
            try:
                giou = get_word_iou(pword, gword)
            except:
                giou = -1
            gious.append(giou)

        def segment_to_list(segment):
            return segment.label, segment.start, segment.end, segment.score
        
        segments = [segment_to_list(segment) for segment in outputs['segments']]
        mmcv.dump([segments, gious], cache_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', default=False, help='Run with this option to get commands')
    parser.add_argument('--total',default=16, type=int, help='Num of splits, ')
    parser.add_argument('--task',default=0,type=int, help='Task id, when dry-run is on, this is untouch')
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7', help='List of gpus used, when dry-run is on')
    parser.add_argument('--max_parallel_process',default=16,type=int, help='Num of process runing in parallel, \
                                    each tmux will run a number of total/max_parallel_process sequentially')
    parser.add_argument('--name', default='multi_infer')
    args = parser.parse_args()
    
    w2v_aligner = W2vForceAligner('pretrained/epoch=43_val_loss=0.0000.ckpt', device='cuda')
    if args.dry_run:
        # Split task to different process runing on multi process
        cur_file = (osp.relpath(__file__))
        devices = [f'{gpu}' for gpu in args.gpus.split(',')]
        list_cmds = [[] for _ in range(args.max_parallel_process)]

        for process_id in range(args.total):
            gpu_id = devices[process_id%len(devices)]
            cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python {cur_file} --task {process_id} --total {args.total}'
            group_id = process_id%args.max_parallel_process
            list_cmds[group_id].append(cmd)

        for group_id, cmds in enumerate(list_cmds):
            cmd = '\n'.join(cmds)
            tmp_file = f'/tmp/{args.name}_{group_id}.sh'
            with open(tmp_file, 'w') as f:
                f.write(cmd)
            print(f"tmux new -s '{args.name}_{group_id:02d}' -d 'sh {tmp_file} && echo Done && sleep 10'")
            os.system(f"tmux new -s '{args.name}_{group_id:02d}' -d 'sh {tmp_file} && echo Done && sleep 10'")
    else:

        json_paths = mmcv.load('./data/training/training_data.pkl').path.tolist()
        tobe_run = []
        for inp in tqdm(json_paths[args.task::args.total]):
            precompute_w2v(inp, w2v_aligner)
