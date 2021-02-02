from utils.common import human_format

time_margin= 0.1
working_sample_rate = 48000
frame_size = int(working_sample_rate * time_margin * 2)
hop_size = frame_size // 2
audio_channels = 1
min_event_percentage_in_positive_frame = 1/3

cfg_descriptor = f"WaveForm_SaR-{human_format(working_sample_rate)}_FrS-{human_format(frame_size)}" \
                 f"_HoS-{human_format(hop_size)}_Ch-{audio_channels}"