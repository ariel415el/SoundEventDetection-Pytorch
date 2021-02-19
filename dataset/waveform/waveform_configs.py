from utils.common import human_format
from dataset.common_config import *

cfg_descriptor = f"WaveForm_SaR-{human_format(working_sample_rate)}_FrS-{human_format(frame_size)}" \
                 f"_HoS-{human_format(hop_size)}_Ch-{audio_channels}"