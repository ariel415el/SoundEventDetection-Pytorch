from utils.common import human_format

working_sample_rate = 48000  # resample all waveforms to this sampling rate
frame_size = 1000  # Size of frames on to extract spectogram form
hop_size = 500  # Gap between frames: there are (sample_rate / hop_size) frames per second
audio_channels = 1   # Restrict data to only the audio_channels first channels of the audio file

# frames_per_second = working_sample_rate // hop_size

cfg_descriptor = f"WaveForm_SaR-{human_format(working_sample_rate)}_FrS-{human_format(frame_size)}" \
                 f"_HoS-{human_format(hop_size)}_Ch-{audio_channels}"