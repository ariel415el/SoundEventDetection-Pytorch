from utils.common import human_format
import numpy as np
from dataset.common_config import *

NFFT = 2**int(np.ceil(np.log2(frame_size)))  # The size of the padded frames on which fft will actualy work. Set this to a power of two for faster preprocessing
mel_bins = 64                                # How much frames to stretch over the
mel_min_freq = 20                            # Hz first mel bin (minimal possible value 0)
mel_max_freq = working_sample_rate // 2      # Hz last mel bin (maximal possible value sampling_rate / 2)

train_crop_size = frames_per_second * 10  # 10-second log mel spectrogram as input


cfg_descriptor = f"Spectogram_SaR-{human_format(working_sample_rate)}_FrS-{human_format(frame_size)}" \
                 f"_HoS-{human_format(hop_size)}_Mel-{mel_bins}_Ch-{audio_channels}"
