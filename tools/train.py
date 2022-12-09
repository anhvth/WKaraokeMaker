#=========================Config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_txt')
parser.add_argument('--val_txt')
parser.add_argument('--ctc_loss_scale', '-c', default=0, type=float)
parser.add_argument('--epochs', '-e', default=125, type=int)
parser.add_argument('--batch_size', '-b', default=10, type=int)
parser.add_argument('--gpus', '-g', default=6, type=int)
parser.add_argument('--strategy', '-s', default='ddp', type=str, help='ddp, dp')
parser.add_argument('--debug', '-d', action='store_true', default=False)
parser.add_argument('--overfit_batches', '-o', default=0.0, type=float)
parser.add_argument('--exp_name', '-n', default='base_detection_no_ckpt_1k', type=str)
parser.add_argument('--num_workers', '-w', default=4, type=int)
parser.add_argument('--num_lr_cycle', '-l', default=3, type=int)
parser.add_argument('--lr', '-lr', default=1e-4, type=float)
parser.add_argument('--ckpt_pretrain', '-ckpt', default=None, type=str, help='path to pretrained checkpoint, lightning will load it automatically')
parser.add_argument('--no-sot', dest='sot', action='store_false', help='disable sot, by default sot is enabled')
parser.add_argument('--resume', action='store_true', help='resume training from last checkpoint')


args = parser.parse_args()
if args.debug: # debug mode
    print("DEBUG=1")
    args.gpus=1
    args.overfit_batches=2
    args.epochs=3
    args.strategy = 'dp'
    args.batch_size=2
    args.num_workers = 0
    args.exp_name += '/debug'



#====================== IMPORTS
from kmaker.data import *
from kmaker.dataloader import *
from kmaker.model import *
from kmaker.trainer import CustomModelTrainer
from ple.all import *

if __name__ == '__main__':
    # Boilerplate training code..............
    train_json_paths = get_json_paths(args.train_txt)
    val_json_paths = get_json_paths(args.val_txt)


    val_ds = AudioDataset(val_json_paths, 'val')
    
    collate_fn = collate_fn_with_sot if args.sot else collate_fn_without_sot
    
    dl_val = torch.utils.data.DataLoader(val_ds, args.batch_size, num_workers=args.num_workers, 
                                        shuffle=False, collate_fn=lambda x:collate_fn(x, is_training=False))

    train_ds = AudioDataset(train_json_paths, 'train')
    dl_train = torch.utils.data.DataLoader(train_ds, args.batch_size, num_workers=args.num_workers, 
                                        shuffle=True, collate_fn=lambda x:collate_fn(x, is_training=True))
    
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
    optim = lambda params:torch.optim.Adam(params, lr=args.lr)
    # TODO: Try WAdam

    model = get_whisper('base')
    modified_model = modify_whisper(model, args.sot)
    if args.ckpt_pretrain:
        lit = CustomModelTrainer.load_from_checkpoint(args.ckpt_pretrain, model=modified_model,create_optimizer_fn=optim, 
                                                      create_lr_scheduler_fn=sched, loss_fn=nn.CrossEntropyLoss())
    else:
        lit = CustomModelTrainer(model=modified_model,create_optimizer_fn=optim,
                                    create_lr_scheduler_fn=sched, loss_fn=nn.CrossEntropyLoss())   

    assert len(dl_val)
    trainer = get_trainer(args.exp_name, args.epochs, gpus=args.gpus, overfit_batches= args.overfit_batches,
                        monitor={'metric': 'val/loss_giou', 'mode': 'min'}, strategy=args.strategy, precision=32)
    if args.resume:
        trainer.fit(lit, dl_train, dl_val, ckpt_path=args.ckpt_pretrain)
    else:
        trainer.fit(lit, dl_train, dl_val)
