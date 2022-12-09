#=========================HYPERPARAMETER
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir')
parser.add_argument('--val_dir')
parser.add_argument('--arguments', '-a', nargs='+', default=None)
parser.add_argument('--ctc_loss_scale', '-c', default=0, type=float)
parser.add_argument('--epochs', '-e', default=125, type=int)
parser.add_argument('--batch_size', '-b', default=10, type=int)
parser.add_argument('--gpu', '-g', default=6, type=int)
parser.add_argument('--strategy', '-s', default='ddp', type=str, help='ddp, dp')
parser.add_argument('--debug', '-d', default=0, type=int)
parser.add_argument('--overfit_batches', '-o', default=0.0, type=float)
parser.add_argument('--exp_name', '-n', default='base_detection_no_ckpt_1k', type=str)
parser.add_argument('--num_workers', '-w', default=4, type=int)
parser.add_argument('--num_lr_cycles', '-l', default=3, type=int)
parser.add_argument('--lr', '-lr', default=1e-4, type=float)
parser.add_argument('--ckpt_pretrain', '-ckpt', default=None, type=str, help='path to pretrained checkpoint, lightning will load it automatically')

args = parser.parse_args()
if args.debug:
    print("DEBUG=1")
    args.gpus=1
    args.overfit_batches=2
    args.epochs=3
    args.strategy = 'dp'
    args.batch_size=2
    args.num_workers = 0
    args.exp_name += '/debug'



#=======================================================================================================
import os
from kmaker.data import *
from kmaker.dataloader import *
from kmaker.model import *
from kmaker.trainer import CustomModelTrainer
from ple.all import *

# os.environ['TRANSFORMERS_OFFLINE'] = '1'

if __name__ == '__main__':
    train_df = get_data(args.train_dir, ds_set_name='train')
    val_df = get_data(args.val_dir, ds_set_name='val')
    len(train_df), len(val_df)

    val_ds = AudioDataset(val_df.it.tolist(), 'val')
    
    collate_fn = collate_fn_with_sot if args.sot else collate_fn_without_sot
    
    collate_fn_val = lambda x:collate_fn(x, is_training=False)
    dl_val = torch.utils.data.DataLoader(val_ds, args.batch_size, num_workers=args.num_workers, 
                                        shuffle=False, collate_fn=collate_fn_val)

    train_ds = AudioDataset(train_df.it.tolist(), 'train')
    dl_train = torch.utils.data.DataLoader(train_ds, args.batch_size, num_workers=args.num_workers, 
                                        shuffle=True, collate_fn=collate_fn)
    print(f'{len(dl_train)=} | {len(dl_val)=}')



    # ---- Lr scheduler
    sched = fn_schedule_cosine_with_warmpup_decay_timm(
        num_epochs=args.epochs,
        num_steps_per_epoch=len(dl_train)//args.gpus,
        num_epochs_per_cycle=args.epochs//args.num_lr_cycle,
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
    trainer = get_trainer(EXP_NAME, args.epochs, gpus=args.gpus, overfit_batches= args.overfit_batches,
                        monitor={'metric': 'val/loss_giou', 'mode': 'min'}, strategy=args.strategy, precision=32)

    trainer.fit(lit, dl_train, dl_val, ckpt_path=args.ckpt_pretrain)
