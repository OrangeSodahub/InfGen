from typing import Dict, Optional
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from infgen.modules.attr_tokenizer import Attr_Tokenizer
from infgen.modules.agent_decoder import InfGenAgentDecoder
from infgen.modules.occ_decoder import InfGenOccDecoder
from infgen.modules.map_decoder import InfGenMapDecoder


DECODER = {'agent_decoder': InfGenAgentDecoder,
           'occ_decoder': InfGenOccDecoder}


class InfGenDecoder(nn.Module):

    def __init__(self,
                 decoder_type: str,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 pl2seed_radius: float,
                 a2a_radius: float,
                 a2sa_radius: float,
                 pl2sa_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_token: Dict,
                 token_size=512,
                 attr_tokenizer: Attr_Tokenizer=None,
                 predict_motion: bool=False,
                 predict_state: bool=False,
                 predict_map: bool=False,
                 predict_occ: bool=False,
                 use_grid_token: bool=True,
                 use_head_token: bool=True,
                 use_state_token: bool=True,
                 disable_insertion: bool=False,
                 state_token: Dict[str, int]=None,
                 seed_size: int=5,
                 buffer_size: int=32,
                 num_recurrent_steps_val: int=-1,
                 loss_weight: dict=None,
                 logger=None) -> None:

        super(InfGenDecoder, self).__init__()

        self.map_encoder = InfGenMapDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            map_token=map_token,
        )

        assert decoder_type in list(DECODER.keys()), f"Unsupport decoder type: {decoder_type}"
        self.agent_encoder = DECODER[decoder_type](
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            pl2seed_radius=pl2seed_radius,
            a2a_radius=a2a_radius,
            a2sa_radius=a2sa_radius,
            pl2sa_radius=pl2sa_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            token_size=token_size,
            attr_tokenizer=attr_tokenizer,
            predict_motion=predict_motion,
            predict_state=predict_state,
            predict_map=predict_map,
            predict_occ=predict_occ,
            state_token=state_token,
            use_grid_token=use_grid_token,
            use_head_token=use_head_token,
            use_state_token=use_state_token,
            disable_insertion=disable_insertion,
            seed_size=seed_size,
            buffer_size=buffer_size,
            num_recurrent_steps_val=num_recurrent_steps_val,
            loss_weight=loss_weight,
            logger=logger,
        )
        self.map_enc = None
        self.predict_motion = predict_motion
        self.predict_state = predict_state
        self.predict_map = predict_map
        self.predict_occ = predict_occ
        self.data_keys = ["agent_valid_mask", "category", "valid_mask", "av_index", "scenario_id", "shape"]

    def get_agent_inputs(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        return self.agent_encoder.get_inputs(data)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)

        agent_enc = {}
        if self.predict_motion or self.predict_state or self.predict_occ:
            agent_enc = self.agent_encoder(data, map_enc)

        return {**map_enc, **agent_enc, **{k: data[k] for k in self.data_keys}}

    def inference(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)

        agent_enc = {}
        if self.predict_motion or self.predict_state or self.predict_occ:
            agent_enc = self.agent_encoder.inference(data, map_enc)

        return {**map_enc, **agent_enc, **{k: data[k] for k in self.data_keys}}

    def inference_no_map(self, data: HeteroData, map_enc) -> Dict[str, torch.Tensor]:
        agent_enc = self.agent_encoder.inference(data, map_enc)
        return {**map_enc, **agent_enc}

    def insert_agent(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder.insert(data, map_enc)
        return {**map_enc, **agent_enc, **{k: data[k] for k in self.data_keys}}

    def predict_nearest_pos(self, data: HeteroData, rank) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        self.agent_encoder.predict_nearest_pos(data, map_enc, rank)
