# Input pkl
import mmcv
valid_trainval = mmcv.load('data/training/training_data.pkl')
valid_trainval.sample(frac=1, random_state=0)

n_train = int(0.85*len(valid_trainval))

train_df = valid_trainval[:n_train]
val_df = valid_trainval[n_train:]


with open('data/training/train.txt', 'w') as f:
    for _path in train_df.path.values:
        f.write(_path + '\n')
        
with open('data/training/val.txt', 'w') as f:
    for _path in val_df.path.values:
        f.write(_path + '\n')
        
        
with open('data/training/train_debug.txt', 'w') as f:
    for _path in train_df.path.values[:100]:
        f.write(_path + '\n')
        

        
with open('data/training/val_debug.txt', 'w') as f:
    for _path in train_df.path.values[:100]:
        f.write(_path + '\n')
        
