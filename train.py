import pytorch_lightning as pl
import os
import shutil
import fnmatch
import torch
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

from infgen.utils.func import Logging, load_config_act
from infgen.datasets.scalable_dataset import MultiDataModule
from infgen.model.infgen import InfGen


def backup(source_dir, backup_dir):
    """
    Back up the source directory (code and configs) to a backup directory.
    """

    if os.path.exists(backup_dir):
        return
    os.makedirs(backup_dir, exist_ok=False)

    # Helper function to check if a path matches exclude patterns
    def should_exclude(path):
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
        return False

    # Iterate through the files and directories in source_dir
    for root, dirs, files in os.walk(source_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude(d)]

        # Determine the relative path and destination path
        rel_path = os.path.relpath(root, source_dir)
        dest_dir = os.path.join(backup_dir, rel_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy all relevant files
        for file in files:
            if any(fnmatch.fnmatch(file, pattern) for pattern in include_patterns):
                shutil.copy2(os.path.join(root, file), os.path.join(dest_dir, file))
    
    print(f"Backup completed. Files saved to: {backup_dir}")


if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)
    torch.set_printoptions(precision=3)

    parser = ArgumentParser()
    Predictor_hash = {'infgen': InfGen,}
    parser.add_argument('--config', type=str, default='configs/ours_long_term.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--save_ckpt_path', type=str, default="output/debug")
    parser.add_argument('--devices', type=int, default=1)
    args = parser.parse_args()

    # backup codes
    exclude_patterns = ['*output*', '*logs', 'wandb', 'data', '*debug*', '*backup*', 'interact_*', '*edge_map*', '__pycache__']
    include_patterns = ['*.py', '*.json', '*.yaml', '*.yml', '*.sh']
    backup(os.getcwd(), os.path.join(args.save_ckpt_path, 'backups'))

    logger = Logging().log(level='DEBUG')
    config = load_config_act(args.config)
    Predictor = Predictor_hash[config.Model.predictor]
    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
    Data_config = config.Dataset
    datamodule = MultiDataModule(**vars(Data_config), logger=logger)

    import os
    wandb_logger = None
    if int(os.getenv('WANDB', 0)) and not int(os.getenv('DEBUG', 0)):
        # squeue -O username,state,nodelist,gres,minmemory,numcpus,name
        wandb_logger = WandbLogger(project='simagent')

    trainer_config = config.Trainer
    max_epochs = trainer_config.max_epochs

    if args.pretrain_ckpt == '':
        model = Predictor(config.Model, save_path=args.save_ckpt_path, logger=logger, max_epochs=max_epochs)
    else:
        model = Predictor(config.Model, save_path=args.save_ckpt_path, logger=logger, max_epochs=max_epochs)
        model.load_params_from_file(filename=args.pretrain_ckpt)

    every_n_epochs = 1
    if int(os.getenv('OVERFIT', 0)):
        max_epochs = trainer_config.overfit_epochs
        every_n_epochs = 100

    if int(os.getenv('CHECK_INPUTS', 0)):
        max_epochs = 1

    check_val_every_n_epoch = 1  # save checkpoints for each epoch
    model_checkpoint = ModelCheckpoint(dirpath=args.save_ckpt_path,
                                       filename='{epoch:02d}',
                                       save_top_k=5,
                                       monitor='epoch',
                                       mode='max',
                                       save_last=True,
                                       every_n_train_steps=1000,
                                       save_on_train_epoch_end=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=trainer_config.accelerator, devices=args.devices if args.devices is not None else trainer_config.devices,
                         strategy=strategy, logger=wandb_logger,
                         accumulate_grad_batches=trainer_config.accumulate_grad_batches,
                         num_nodes=trainer_config.num_nodes,
                         callbacks=[model_checkpoint, lr_monitor],
                         max_epochs=max_epochs,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         log_every_n_steps=1,
                         gradient_clip_val=0.5)

    if args.ckpt_path == '':
        trainer.fit(model, datamodule)
    else:
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
