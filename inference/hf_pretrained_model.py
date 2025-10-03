import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from transformers import PretrainedConfig, PreTrainedModel

from src.model import ConvResidualVQVAE_ResNet50Backbone


class ResVQVAEResNet50Config(PretrainedConfig):
    model_type = 'res-vqvae-net50'

    def __init__(self,
                 in_channels=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels


class ResVQVAEResNet50(PreTrainedModel):
    config_class = ResVQVAEResNet50Config

    def __init__(self, config):
        super().__init__(config)
        self.model = ConvResidualVQVAE_ResNet50Backbone(in_channels=config.in_channels)

    def forward(self, x):
        return self.model(x)
