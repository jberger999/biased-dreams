import torch
from typing import Union

from uncertainty_aware_dreamer.ssm_mbrl.rssm.rssm import RSSM
from uncertainty_aware_dreamer.ssm_mbrl.rssm.transition.transition_factory import TransitionFactory
import uncertainty_aware_dreamer.ssm_mbrl.common.time_distribution as td

from uncertainty_aware_dreamer.ssm_mbrl.util.config_dict import ConfigDict

nn = torch.nn


class RSSMFactory:

    def __init__(self,
                 encoder_factories,
                 decoder_factories):
        self._encoder_factories = encoder_factories
        self._decoder_factories = decoder_factories

    @property
    def reward_decoder_idx(self) -> int:
        return len(self._decoder_factories) - 1

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:

        config = ConfigDict()

        for i, factory in enumerate(self._encoder_factories):
            config.add_subconf(name="encoder{}".format(i),
                               sub_conf=factory.get_default_config(finalize_adding=finalize_adding))

        for i, factory in enumerate(self._decoder_factories):
            config.add_subconf(name="decoder{}".format(i),
                               sub_conf=factory.get_default_config(finalize_adding=finalize_adding))

        config.add_subconf(name="transition",
                           sub_conf=TransitionFactory.get_default_config(finalize_adding=finalize_adding))

        if finalize_adding:
            config.finalize_adding()

        return config

    def _build(self,
               config: ConfigDict,
               input_sizes: list[Union[int, tuple[int, int, int]]],
               output_sizes: list[Union[int, tuple[int, int, int]]],
               action_dim: int):

        encoders, enc_out_sizes = [], []
        assert len(input_sizes) == len(self._encoder_factories)
        for i, (input_size, factory) in enumerate(zip(input_sizes, self._encoder_factories)):
            current_config = getattr(config, "encoder{}".format(i))
            enc, enc_out_size = factory.build(input_size=input_size,
                                              config=current_config)
            encoders.append(td.Jitted11TD(base_module=enc))
            enc_out_sizes.append(enc_out_size)

        transition_model = TransitionFactory().build(config.transition,
                                                     obs_sizes=enc_out_sizes,
                                                     action_dim=action_dim)

        decoders = []
        assert len(output_sizes) == len(self._decoder_factories)
        for i, (output_size, factory) in enumerate(zip(output_sizes, self._decoder_factories)):
            current_config = getattr(config, "decoder{}".format(i))
            dec = factory.build(input_size=transition_model.feature_size,
                                output_size=output_size,
                                config=current_config)
            decoders.append(td.Jitted12TD(base_module=dec))

        return encoders, transition_model, decoders

    def build(self,
              config: ConfigDict,
              input_sizes: list[Union[int, tuple[int, int, int]]],
              output_sizes: list[Union[int, tuple[int, int, int]]],
              action_dim: int):
        encoders, transition_model, decoders = self._build(config=config,
                                                           input_sizes=input_sizes,
                                                           output_sizes=output_sizes,
                                                           action_dim=action_dim)
        return RSSM(encoders=torch.nn.ModuleList(encoders),
                    transition_model=transition_model,
                    decoders=torch.nn.ModuleList(decoders))