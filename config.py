working_sample_rate = 32000
window_size = 1024
hop_size = 500
# window_size = 2**12  # This better be a power of two for faster preprocessing
# hop_size = 2000      # Gap between frames windows: there are (sample_rate / sample_rate) frames per second
# window_size = 2**11  # This better be a power of two for faster preprocessing
# hop_size = 1000      # Gap between frames windows: there are (sample_rate / sample_rate) frames per second
mel_bins = 64
mel_min_freq = 50       # Hz
mel_max_freq = 14000    # Hz
audio_channels = 1

frames_per_second = working_sample_rate // hop_size
train_crop_size = frames_per_second * 10     # 10-second log mel spectrogram as input

# Clap details:
time_margin = 0.1

# Tau-SED details:
# The label configuration is the same as https://github.com/sharathadavanne/seld-dcase2019
# tau_sed_labels = ['knock', 'drawer', 'clearthroat', 'phone', 'keysDrop', 'speech',
#           'keyboard', 'pageturn', 'cough', 'doorslam', 'laughter']

# tau_sed_labels = ['knock', 'keysDrop', 'doorslam']
tau_sed_labels = ['doorslam']
classes_num = len(tau_sed_labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(tau_sed_labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(tau_sed_labels)}
