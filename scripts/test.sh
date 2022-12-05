PYTHONPATH=./ python tools/test.py lightning_logs/base_detection_no_ckpt_1k/08/ckpts/epoch\=116_val_loss_giou\=0.0000.ckpt data/training with_masked_text -m 200



PYTHONPATH=./ python tools/test.py lightning_logs/base_detection_no_ckpt_1k/08/ckpts/epoch\=116_val_loss_giou\=0.0000.ckpt data/public_test with_masked_text
PYTHONPATH=./ python tools/test.py lightning_logs/base_detection_no_ckpt_1k/08/ckpts/epoch\=116_val_loss_giou\=0.0000.ckpt data/private_test with_masked_text 