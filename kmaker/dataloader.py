from kmaker.data import *

# Data utils/ augmentation
def stack_input(inputs, target_shape=[80, 3000]):
    batch_shape = [len(inputs), *target_shape]
    inps = torch.zeros(batch_shape)
    for i, input in enumerate(inputs):
        l = input.shape[1]
        inps[i,:, :l] = input
    return inps

def stack_bbox(list_bb, pad_val=-1):
    l = max([len(_) for _ in list_bb])
    out_shape = [len(list_bb), l, 4]
    z = torch.ones(out_shape)*pad_val
    for i, b in enumerate(list_bb):
        _l = len(b)
        z[i,:_l] = b
    return z

def stack_1d(tensors, pad_val=-100):
    l = max([len(t) for t in tensors])
    z = torch.ones([len(tensors), l])*-100
    for i, t in enumerate(tensors):
        _l = len(t)
        z[i, :_l] = t
    return z


def pad_left(item:ItemAudioLabel):
    """ Audio augmentation: pad left

    Args:
        item:AudioLabel

    Returns:
        item:AudioLabel
    """
    item = item.copy()

    current_len = item['inputs'].shape[1]
    if current_len>=3000:
        item['inputs'] = item['inputs'][:,:3000]
        return item
    
    max_pading_left = 3000-current_len
    assert max_pading_left>=0, item['item_idx']
    padding_left = np.random.choice(max_pading_left)
    out = np.zeros([80, 3000])
    out[:,padding_left:padding_left+current_len] = item['inputs']

    box_pad_x = padding_left/3000
    new_bboxes = []
    for bbox in item['bboxes']:
        bbox[:,0] = bbox[:,0]+box_pad_x
        new_bboxes.append(bbox)
    item['inputs'] = out
    item['bboxes'] = new_bboxes
    
    return item

music_token = wtokenizer.non_speech_tokens[43]
def mask_out(item):
    out_tokens = list(item['tokens'].copy())
    dec_ids = item['dec_pos']
    prev_is_mask = False
    for a, b in zip(dec_ids[:-1], dec_ids[1:]):
        if np.random.choice(2) == 0 and not prev_is_mask:
            out_tokens[a:b] = [music_token]*(b-a)
            prev_is_mask = True
        else:
            prev_is_mask = False
    item['tokens'] = out_tokens
    return item


def collate_fn_without_sot(items, is_training):
    """
        Simplified version of whisper input
    """
    if is_training:
        items = [pad_left(item) for item in items]
        items = [mask_out(item) for item in items] # mask out words with music token
    inputs = [torch.from_numpy(item['inputs']).float() for item in items]
    labels = [torch.tensor(item['tokens']).long() for item in items]
    w2v_labels = [torch.tensor(item['w2v_tokens']).long() for item in items]
    dec_pos = [torch.tensor([_ for _ in item['dec_pos']]) for item in items]
    bboxes = [torch.cat(item['bboxes']) for item in items]
    # loss_scale = [torch.from_numpy(np.ones(len(item['bboxes']))) for item in items]
    loss_scale = [torch.from_numpy(item['loss_scale']) for item in items]

    inputs = stack_input(inputs)
    labels = stack_1d(labels).long()
    w2v_labels = stack_1d(w2v_labels).long()
    dec_pos = stack_1d(dec_pos).int()    
    bboxes = torch.cat(bboxes)
    loss_scale = torch.cat(loss_scale)
    masks = dec_pos!=-100
    
    
    ids1 = torch.where(dec_pos>=0)[0]
    ids2 = dec_pos[np.where(dec_pos>=0)].long()
    # gt_conf = torch.cat([torch.from_numpy(item['gt_conf']) for item in items])
    
    batch= dict(
        inputs=inputs, labels=labels, dec_pos=(ids1, ids2), 
        bboxes=bboxes,masks=masks, loss_scale=loss_scale,w2v_labels=w2v_labels,
        ids=[item['idx'] for item in items],
        transcript=[item['transcript'] for item in items]
    )
    assert batch['w2v_labels'].max()<110, batch['ids']
    return batch



def collate_fn_with_sot(items, is_training):
    """
        Input starts with <sot> and ends with <eot> as in whisper
    """
    
    if is_training:
        items = [pad_left(item) for item in items]
        items = [mask_out(item) for item in items] # mask out words with music token
    inputs = [torch.from_numpy(item['inputs']).float() for item in items]
    labels = [torch.tensor(item['tokens']).long() for item in items]
    w2v_labels = [torch.tensor(item['w2v_tokens']).long() for item in items]
    dec_pos = [torch.tensor([_+3 for _ in item['dec_pos']]) for item in items]
    bboxes = [torch.cat(item['bboxes']) for item in items]
    # loss_scale = [torch.from_numpy(np.ones(len(item['bboxes']))) for item in items]
    loss_scale = [torch.from_numpy(item['loss_scale']) for item in items]

    inputs = stack_input(inputs)
    labels = stack_1d(labels).long()
    w2v_labels = stack_1d(w2v_labels).long()
    dec_pos = stack_1d(dec_pos).int()    
    bboxes = torch.cat(bboxes)
    loss_scale = torch.cat(loss_scale)
    masks = dec_pos!=-100
    
    
    ids1 = torch.where(dec_pos>=0)[0]
    ids2 = dec_pos[np.where(dec_pos>=0)].long()
    # gt_conf = torch.cat([torch.from_numpy(item['gt_conf']) for item in items])
    
    batch= dict(
        inputs=inputs, labels=labels, dec_pos=(ids1, ids2), 
        bboxes=bboxes,masks=masks, loss_scale=loss_scale,w2v_labels=w2v_labels,
        ids=[item['idx'] for item in items],
        transcript=[item['transcript'] for item in items]
    )
    assert batch['w2v_labels'].max()<110, batch['ids']
    sot = torch.tensor(wtokenizer.sot_sequence)
    sot = torch.stack([sot]*len(batch['labels']))
    batch['labels'] = torch.cat([sot, batch['labels']], 1)            
    return batch



class AudioDataset:
    """Create a dataset for audio data. Input is path_to_jsons, each item of this dataset is a dict of \
        'inputs', 'labels', 'dec_pos', 'bboxes', 'masks', 'loss_scale', 'w2v_labels', 'ids', 'transcript'"""
    def __init__(self, json_paths, ds_name=None):
        assert ds_name in ['train', 'val', 'test', None]
                           
        self.audio_labels = [ItemAudioLabel(_) for _ in json_paths]
        self.dsset = ds_name
        
    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        item = self.audio_labels[idx]
        rt =  dict(inputs=item.mel)
        rt.update(item.get_words_meta())
        rt['w2v_tokens'] = item.w2v_tokens
        assert max(rt['w2v_tokens'])<110, rt['w2v_tokens']
        rt['item_idx'] = idx
        rt['idx'] = idx
        rt['transcript'] = item.transcript
        return rt
