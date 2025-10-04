from types import SimpleNamespace
import torch
import socket

# General
cfg = SimpleNamespace(**{})
cfg.project= "rsna_aeurysm"
cfg.hostname = socket.gethostname()
cfg.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.weights_path= ""
cfg.fast_dev_run= False
cfg.save_weights= True
cfg.logger= None
cfg.train= True
cfg.val= True
cfg.seed= -1
cfg.fold= 0
cfg.resume_from_checkpoint = None
cfg.resume_epoch = None
cfg.resume_best_score = None

# Optimizer + Scheduler
cfg.scheduler = "Constant"
cfg.lr = 1e-4
cfg.lr_min= 1e-5
cfg.weight_decay = 1e-4
cfg.epochs= 50

# Other
cfg.ema= True
cfg.ema_decay= 0.99
cfg.mixed_precision= True
cfg.use_checkpoint= False
cfg.grad_accumulation= 1
cfg.track_grad_norm= False
cfg.grad_clip= 1.0
cfg.grad_norm_type= 2
cfg.logging_steps= 10
cfg.eval_epochs= 10

# Dataset/Dataloader
cfg.num_workers= 4
cfg.drop_last= True
cfg.pin_memory = True
cfg.batch_size= 8
cfg.batch_size_val= 1

# Augs
cfg.mixup_p= 1.0
cfg.mixup_beta= 1.0
cfg.cutmix_p = 0.15
cfg.rescale_p = 1.0
cfg.train_aug = None
cfg.val_aug = None
cfg.tta= True