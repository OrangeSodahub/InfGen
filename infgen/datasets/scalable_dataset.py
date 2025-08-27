import os
import pickle
import torch
import json
import pytorch_lightning as pl
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import DataLoader
from typing import Callable, Dict, List, Optional

from infgen.datasets.preprocess import TokenProcessor
from infgen.utils.func import CONSOLE


class MultiDataset(Dataset):
    def __init__(self,
                 split: str,
                 raw_dir: List[str] = None,
                 transform: Optional[Callable] = None,
                 tfrecord_dir: Optional[str] = None,
                 token_size=512,
                 predict_motion: bool=False,
                 predict_state: bool=False,
                 predict_map: bool=False,
                 #  state_token: Dict[str, int]=None,
                 #  pl2seed_radius: float=None,
                 buffer_size: int=128,
                 scenario_id: str=None,
                 logger=None) -> None:

        self.disable_invalid = not predict_state
        self.predict_motion = predict_motion
        self.predict_state = predict_state
        self.predict_map = predict_map
        self.logger = logger
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.training = split == 'train'
        self.buffer_size = buffer_size
        self._tfrecord_dir = tfrecord_dir
        self.logger.debug('Starting loading dataset')

        raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
        self._raw_files = sorted(os.listdir(raw_dir))

        # load meta infos and do filter
        json_path = '~/infgen/data/waymo_processed/meta_infos.json'
        label = 'training' if split == 'train' else ('validation' if split == 'val' else split)
        self.meta_infos = json.load(open(json_path, 'r', encoding='utf-8'))[label]
        self.logger.debug(f"Loaded meta infos from {json_path}")
        self.available_scenarios = list(self.meta_infos.keys())
        # self._raw_files = list(tqdm(filter(lambda fn: (
        #     scenario_id := fn.removesuffix('.pkl') in self.available_scenarios and
        #     8 <= self.meta_infos[scenario_id]['num_agents'] < self.buffer_size
        # ), self._raw_files), leave=False))
        df = pd.DataFrame.from_dict(self.meta_infos, orient='index')
        available_scenarios_set = set(self.available_scenarios)
        df_filtered = df[(df.index.isin(available_scenarios_set)) & (df['num_agents'] >= 8) & (df['num_agents'] < self.buffer_size)]
        valid_scenarios = set(df_filtered.index)
        if scenario_id is not None:
            CONSOLE.log(f'Set {scenario_id=}!')
            valid_scenarios = [scenario_id]
        self._raw_files = list(tqdm(filter(lambda fn: fn.removesuffix('.pkl') in valid_scenarios, self._raw_files), leave=False))
        if len(self._raw_files) <= 0:
            raise RuntimeError(f'Invalid number of data {len(self._raw_files)}!')
        self._raw_paths = list(map(lambda fn: os.path.join(raw_dir, fn), self._raw_files))

        self.logger.debug(f"The number of {split} dataset is {len(self._raw_paths)}")
        self.logger.debug(f"The buffer size is {self.buffer_size}")
        # self.token_processor = TokenProcessor(token_size,
        #                                       training=self.training,
        #                                       predict_motion=self.predict_motion,
        #                                       predict_state=self.predict_state,
        #                                       predict_map=self.predict_map,
        #                                       state_token=state_token,
        #                                       pl2seed_radius=pl2seed_radius) # 2048
        self.logger.debug(f"The used token size is {token_size}.")
        super().__init__(transform=transform, pre_transform=None, pre_filter=None)

    def len(self) -> int:
        return len(self._raw_paths)

    def get(self, idx: int):
        """
        Load pkl file (each represents a 91s scenario for waymo dataset)
        """
        with open(self._raw_paths[idx], 'rb') as handle:
            data = pickle.load(handle)

        if self._tfrecord_dir is not None:
            data['tfrecord_path'] = os.path.join(self._tfrecord_dir, f"{data['scenario_id']}.tfrecords")

        # data = self.token_processor.preprocess(data)
        return data


class WaymoTargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int,
                 max_num: int,
                 training: bool=False) -> None:

        self.max_num = max_num
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.step_current = num_historical_steps - 1
        self.training = training

    def _score_trained_agents(self, data):
        pos = data['agent']['position']
        av_index = torch.where(data['agent']['role'][:, 0])[0].item()
        distance = torch.norm(pos - pos[av_index], dim=-1)

        # we do not believe the perception out of range of 150 meters
        data['agent']['valid_mask'] &= distance < 150

        # we do not predict vehicle too far away from ego car
        role_train_mask = data['agent']['role'].any(-1)
        extra_train_mask = (distance[:, self.step_current] < 100) & (
            data['agent']['valid_mask'][:, self.step_current + 1 :].sum(-1) >= 5
        )

        train_mask = extra_train_mask | role_train_mask
        if train_mask.sum() > self.max_num:  # too many vehicle
            _indices = torch.where(extra_train_mask & ~role_train_mask)[0]
            selected_indices = _indices[
                torch.randperm(_indices.size(0))[: self.max_num - role_train_mask.sum()]
            ]
            data['agent']['train_mask'] = role_train_mask
            data['agent']['train_mask'][selected_indices] = True
        else:
            data['agent']['train_mask'] = train_mask  # [n_agent]

        return data

    def __call__(self, data) -> HeteroData:

        if self.training:
            self._score_trained_agents(data)

        data = TokenProcessor._tokenize_map(data)

        return HeteroData(data)


class MultiDataModule(pl.LightningDataModule):
    transforms = {
        'WaymoTargetBuilder': WaymoTargetBuilder,
    }

    dataset = {
        'scalable': MultiDataset,
    }

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 val_tfrecords_splitted: Optional[str] = None,
                 transform: Optional[str] = None,
                 dataset: Optional[str] = None,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 processor='ntp',
                 token_size=512,
                 predict_motion: bool=False,
                 predict_state: bool=False,
                 predict_map: bool=False,
                 state_token: Dict[str, int]=None,
                 pl2seed_radius: float=None,
                 max_num: int=32,
                 buffer_size: int=256,
                 logger=None,
                 scenario_id: Optional[str] = None,
                 **kwargs) -> None:

        super(MultiDataModule, self).__init__()
        self.root = root
        self.dataset_class = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.val_tfrecords_splitted = val_tfrecords_splitted
        self.processor = processor
        self.token_size = token_size
        self.predict_motion = predict_motion
        self.predict_state = predict_state
        self.predict_map = predict_map
        self.state_token = state_token
        self.pl2seed_radius = pl2seed_radius
        self.buffer_size = buffer_size
        self.logger = logger
        self.scenario_id = scenario_id

        self.train_transform = MultiDataModule.transforms[transform](num_historical_steps,
                                                                     num_future_steps,
                                                                     max_num=max_num,
                                                                     training=True)
        self.val_transform = MultiDataModule.transforms[transform](num_historical_steps,
                                                                   num_future_steps,
                                                                   max_num=max_num,
                                                                   training=False)

    def setup(self, stage: Optional[str] = None) -> None:
        general_params = dict(token_size=self.token_size,
                              predict_motion=self.predict_motion,
                              predict_state=self.predict_state,
                              predict_map=self.predict_map,
                              buffer_size=self.buffer_size,
                              scenario_id=self.scenario_id,
                              logger=self.logger)

        if stage == 'fit' or stage is None:
            self.train_dataset = MultiDataModule.dataset[self.dataset_class](split='train',
                                                                             raw_dir=self.train_raw_dir,
                                                                             transform=self.train_transform,
                                                                             **general_params)
            self.val_dataset = MultiDataModule.dataset[self.dataset_class](split='val',
                                                                           raw_dir=self.val_raw_dir,
                                                                           transform=self.val_transform,
                                                                           tfrecord_dir=self.val_tfrecords_splitted,
                                                                           **general_params)
        if stage == 'validate':
            self.val_dataset = MultiDataModule.dataset[self.dataset_class](split='val',
                                                                           raw_dir=self.val_raw_dir,
                                                                           transform=self.val_transform,
                                                                           tfrecord_dir=self.val_tfrecords_splitted,
                                                                           **general_params)
        if stage == 'test':
            self.test_dataset = MultiDataModule.dataset[self.dataset_class](split='test',
                                                                            raw_dir=self.test_raw_dir,
                                                                            transform=self.val_transform,
                                                                            tfrecord_dir=self.val_tfrecords_splitted,
                                                                            **general_params)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
