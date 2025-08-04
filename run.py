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

from infgen.utils.func import RankedLogger, load_config_act, CONSOLE
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
    
    logger.info(f"Backup completed. Files saved to: {backup_dir}")


if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)
    torch.set_printoptions(precision=3)

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ours_long_term.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, default=None,
                        help='Path to any pretrained model, will only load its parameters.'
    )
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to any trained model, will load all the states.'
    )
    parser.add_argument('--save_ckpt_path', type=str, default='output/debug',
                        help='Path to save the checkpoints in training mode'
    )
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the inference results in validation and test mode.'
    )
    parser.add_argument('--wandb', action='store_true',
                        help='Whether to use wandb logger in training.'
    )
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot_rollouts', action='store_true')
    parser.add_argument('--scenario_id', type=str, default=None)
    args = parser.parse_args()

    if not (args.train or args.validate or args.test or args.plot_rollouts):
        raise RuntimeError(f"Got invalid action, should be one of ['train', 'validate', 'test', 'plot_rollouts']")

    # ! setup logger
    logger = RankedLogger(__name__, rank_zero_only=True)

    # ! backup codes
    exclude_patterns = ['*output*', '*logs', 'wandb', 'data', '*debug*', '*backup*', 'interact_*', '*edge_map*', '__pycache__']
    include_patterns = ['*.py', '*.json', '*.yaml', '*.yml', '*.sh']
    backup(os.getcwd(), os.path.join(args.save_ckpt_path, 'backups'))

    config = load_config_act(args.config)

    wandb_logger = None
    if args.wandb and not int(os.getenv('DEBUG', 0)):
        # squeue -O username,state,nodelist,gres,minmemory,numcpus,name
        wandb_logger = WandbLogger(project='simagent')

    trainer_config = config.Trainer
    max_epochs = trainer_config.max_epochs

    # ! setup datamodule and model
    datamodule = MultiDataModule(**vars(config.Dataset), logger=logger, scenario_id=args.scenario_id)
    model = InfGen(config.Model, save_path=args.save_ckpt_path, logger=logger, max_epochs=max_epochs)
    if args.pretrain_ckpt:
        model.load_state_from_file(filename=args.pretrain_ckpt)
    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
    logger.info(f'Build model: {model.__class__.__name__} datamodule: {datamodule.__class__.__name__}')

    # ! checkpoint configuration
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

    # ! setup trainer
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
    logger.info(f'Build trainer: {trainer.__class__.__name__}')

    # ! run
    if args.train:

        logger.info(f'Start training ...')
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)

    # NOTE: here both validation and test process use validation split data
    # for validation, we enable the online metric calculation with results dumping
    # for test, we disable it and only dump the inference results.
    else:

        if args.save_path is not None:
            save_path = args.save_path
        else:
            assert args.ckpt_path is not None and os.path.exists(args.ckpt_path), \
                    f'Path {args.ckpt_path} not exists!'
            save_path = os.path.join(os.path.dirname(args.ckpt_path), 'validation')
        os.makedirs(save_path, exist_ok=True)
        CONSOLE.log(f'Results will be saved to [yellow]{save_path}[/]')

        model.save_path = save_path

        if not args.ckpt_path:
            CONSOLE.log(f'[yellow] Warning: no checkpoint will be loaded in validation! [/]')

        if args.validate:

            CONSOLE.log('[on blue] Start validating ... [/]')
            model.set(mode='validation')

        elif args.test:

            CONSOLE.log('[on blue] Sart testing ... [/]')
            model.set(mode='test')

        elif args.plot_rollouts:

            CONSOLE.log('[on blue] Sart generating ... [/]')
            model.set(mode='plot_rollouts')

        trainer.validate(model, datamodule, ckpt_path=args.ckpt_path)
