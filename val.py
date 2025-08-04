import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from rich.console import Console
from torch_geometric.loader import DataLoader

from infgen.datasets.scalable_dataset import MultiDataset, WaymoTargetBuilder
from infgen.utils.func import load_config_act, Logging
from infgen.model.infgen import InfGen

CONSOLE = Console(width=120)


if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--config', type=str, default='configs/ours_long_term.yaml')
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--insert_agent', action='store_true')
    parser.add_argument('--t', type=str, default=2)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    config = load_config_act(args.config)
    logger = Logging().log(level='DEBUG')

    data_config = config.Dataset
    val_dataset = MultiDataset(split='val',
                               raw_dir=data_config.val_raw_dir,
                               token_size=data_config.token_size,
                               transform=WaymoTargetBuilder(
                                   config.Model.num_historical_steps,
                                   config.Model.decoder.num_future_steps,
                                   max_num=data_config.max_num,
                                   training=False),
                               tfrecord_dir=data_config.val_tfrecords_splitted,
                               predict_motion=config.Model.predict_motion,
                               predict_state=config.Model.predict_state,
                               predict_map=config.Model.predict_map,
                               buffer_size=config.Model.buffer_size,
                               logger=logger,
                )
    dataloader = DataLoader(val_dataset,
                            shuffle=False,
                            num_workers=data_config.num_workers,
                            pin_memory=data_config.pin_memory,
                            persistent_workers=True if data_config.num_workers > 0 else False
                )

    if args.save_path is not None:
        save_path = args.save_path
    else:
        assert args.ckpt_path != "" and os.path.exists(args.ckpt_path), f"Path {args.ckpt_path} not exist!"
        save_path = os.path.join(os.path.dirname(args.ckpt_path), 'val')
    CONSOLE.log(f"Results will be saved to [yellow]{save_path}[/]")
    os.makedirs(save_path, exist_ok=True)

    model = InfGen(config.Model, save_path=save_path, logger=logger, insert_agent=args.insert_agent, t=args.t)
    CONSOLE.log(f"Loaded model from [yellow]{args.ckpt_path}[/]")

    trainer_config = config.Trainer
    trainer = pl.Trainer(accelerator=trainer_config.accelerator,
                         devices=trainer_config.devices,
                         strategy='ddp', num_sanity_val_steps=0)
    trainer.validate(model, dataloader, ckpt_path=args.ckpt_path)

    CONSOLE.log(f"Validation done!")
