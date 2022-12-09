#=========================HYPERPARAMETER
DEBUG=True

# TRAIN_PATH = './data/training/training_data_1k.pkl'
VAL_PATH = TRAIN_PATH= './data/training/training_data.pkl'
CTC_LOSS_SCALE = 0


LR = 1e-4
EPOCHS = 125
BZ = 10
GPUS = 6
STRATEGY = 'ddp'
NUM_WORKERS = 4            
OVERFIT_BATCHES = 0.0
CKPT_PRETRAIN = None
EXP_NAME = 'base_detection_no_ckpt_1k'


if DEBUG:
    print("DEBUG=1")
    GPUS=1
    OVERFIT_BATCHES=2
    EPOCHS=3
    STRATEGY = 'dp'
    BZ=2
    NUM_WORKERS = 0
    EXP_NAME += '/debug'
import os

#=======================================================================================================
from kmaker.data import *
from kmaker.dataloader import *
from kmaker.model import *

os.environ['TRANSFORMERS_OFFLINE'] = '1'






if __name__ == '__main__':
    train_df = get_data(TRAIN_PATH, 'train')
    val_df = get_data(VAL_PATH, 'val')
    len(train_df), len(val_df)

    val_ds = AudioDataset(val_df.it.tolist(), 'val')

    collate_fn_val = lambda x:collate_fn(x, is_training=False)
    dl_val = torch.utils.data.DataLoader(val_ds, BZ, num_workers=NUM_WORKERS, 
                                        shuffle=False, collate_fn=collate_fn_val)

    train_ds = AudioDataset(train_df.it.tolist(), 'train')
    dl_train = torch.utils.data.DataLoader(train_ds, BZ, num_workers=NUM_WORKERS, 
                                        shuffle=True, collate_fn=collate_fn)
    print(f'{len(dl_train)=} | {len(dl_val)=}')



    # ---- Lr scheduler
    sched = fn_schedule_cosine_with_warmpup_decay_timm(
        num_epochs=EPOCHS,
        num_steps_per_epoch=len(dl_train)//GPUS,
        num_epochs_per_cycle=EPOCHS//3,
        min_lr=1/100,
        cycle_decay=0.7,
    )
    # --- Optimizer
    optim = lambda params:torch.optim.Adam(params, lr=LR)

    model = get_whisper('base')
    modified_model = modify_whisper(model)

    lit = CustomModelTrainer(model=modified_model,create_optimizer_fn=optim,
                                create_lr_scheduler_fn=sched, loss_fn=nn.CrossEntropyLoss())    

    assert len(dl_val)
    trainer = get_trainer(EXP_NAME, EPOCHS, gpus=GPUS, overfit_batches= OVERFIT_BATCHES,
                        monitor={'metric': 'val/loss_giou', 'mode': 'min'}, strategy=STRATEGY, precision=32)

    trainer.fit(lit, dl_train, dl_val, ckpt_path=CKPT_PRETRAIN)