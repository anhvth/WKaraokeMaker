import argparse
from time import time

import torch

from kmaker.dataloader import *
from kmaker.model import *


def convert_result_to_competion_format(pred_word_segments, json_path, word_idx_to_milisec_ratio):
    """ Convert result to zac2002 competition format

    Args:
        pred_word_segments (_type_): list of Segment
        json_path (_type_): path to json file
        word_idx_to_milisec_ratio (_type_): convert word index to milisecond ratio

    Returns:
        _type_: dictionaly of result in competition format
    """
    pred_i = 0
    target = mmcv.load(json_path)
    for i, line in enumerate(target):
        for j, word in enumerate(line['l']):
            
            pred_word = pred_word_segments[pred_i]
            s = int(pred_word.start*word_idx_to_milisec_ratio)
            e = int(pred_word.end*word_idx_to_milisec_ratio)
            
            if j == 0:
                target[i]['s'] = s
            elif j == len(line['l'])-1:
                target[i]['e'] = e
                
            target[i]['l'][j]['s'] = s
            target[i]['l'][j]['e'] = e
            
            pred_i += 1
    return target


def preproc(json_path):
    """ Preprocess json file

    Args:
        path (str): path to json file
    Returns:
        tuple: (item, batch)
    """
    item = ItemAudioLabel(json_path, spliter='|', is_training=False) 
    rt =  dict(inputs=item.mel)
    rt.update(item.get_words_meta())
    rt['w2v_tokens'] = item.w2v_tokens
    rt['idx'] = None
    rt['transcript'] = item.transcript
    batch = collate_fn([rt], False)
    
    with torch.inference_mode():
        for k, v in batch.items():
            batch[k] = v.cuda() if hasattr(v, 'cuda') else v

    return item, batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', help='path to pretrained checkpoint')
    parser.add_argument('data', help='path to data directory')
    parser.add_argument('exp_name', help='give a name to make it easy to find the solution.zip file')
    parser.add_argument('--no-sot', dest='sot', action='store_false', help='disable sot, by default sot is enabled')
    parser.add_argument('--max_samples', '-m', default=None, type=int, help='max number of samples to evaluate, for debugging purpose')
    args = parser.parse_args()
    
    # Load model    
    st = torch.load(args.ckpt)
    if 'state_dict' in st:
        st = st['state_dict']
        
    model = get_whisper('base')
    modified_model = modify_whisper(model, args.sot)

    # model = get_whisper('base')
    # modified_model = modify_whisper(model)
    new_st = {k[6:]:v for k, v in st.items()}
    modified_model.load_state_dict(new_st)
    eval_model = modified_model.cuda().requires_grad_(False).eval()
    
    
    # Query data
    json_paths = glob(args.data+'/labels/*.json')
    # ds = AudioDataset([ItemAudioLabel(json_path)  for json_path in json_paths])

    collate_fn = collate_fn_with_sot if args.sot else collate_fn_without_sot
    all_predicted_time = []
    all_result = []
    if args.max_samples is not None:
        json_paths = json_paths[0::len(json_paths)//args.max_samples]

    for i, path in tqdm(enumerate(json_paths)):
        t1 = time()
        item, batch = preproc(path)
        with torch.inference_mode():

            outputs = eval_model.forward_both(
                        batch['inputs'],
                        labels=batch['labels'],
                        ctc_labels=batch['w2v_labels'],
                    )
            bboxes = outputs['bbox_pred'][batch['dec_pos']]

        
        xyxy = box_cxcywh_to_xyxy(bboxes)[:,[0,2]]
        words = [_[0] for _ in item.words]
        
        results = []
        xyxy = xyxy.clip(0, 1)
        for (x1,x2), word in zip((xyxy*30).tolist(), words):
            results.append((word, x1, x2))
        results = [Segment(*result, 1) for result in results]
        results = convert_result_to_competion_format(results, path, 1000)
        t2 = time()
        
        all_result.append(results)
        predicted_time = int(t2*1000 - t1*1000)
        
        all_predicted_time.append((item.name, predicted_time))
        print(f'{i} {item.name} {predicted_time}ms')
    
    names = [get_name(path) for path in json_paths]
    
    if 'public' in args.data:
        test_set = 'public'
    elif 'private' in args.data:
        test_set = 'private'
    else:
        test_set = 'training'

    for results, name in zip(all_result, names):
        mmcv.dump(results, f'outputs/{test_set}_{args.exp_name}/submission/{name}.json')

    os.system(f'cd outputs/{test_set}_{args.exp_name} && zip -r {test_set}_{args.exp_name}.zip submission')
    print('Output: {}'.format(osp.abspath(f'outputs/{test_set}_{args.exp_name}.zip')))
    print('Submit -> https://challenge.zalo.ai/portal/e2e-question-answering/submission')
    