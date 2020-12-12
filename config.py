sample_rate = 32000
window_size = 1024
hop_size = 500      # So that there are 64 frames per second
mel_bins = 64
fmin = 50       # Hz
fmax = 14000    # Hz
audio_channels = 1

frames_per_second = sample_rate // hop_size
time_steps = frames_per_second * 10     # 10-second log mel spectrogram as input


# Tau-SED details
# The label configuration is the same as https://github.com/sharathadavanne/seld-dcase2019
# tau_sed_labels = ['knock', 'drawer', 'clearthroat', 'phone', 'keysDrop', 'speech',
#           'keyboard', 'pageturn', 'cough', 'doorslam', 'laughter']

# tau_sed_labels = ['knock', 'keysDrop', 'doorslam']
tau_sed_labels = ['doorslam']

classes_num = len(tau_sed_labels)

lb_to_idx = {lb: idx for idx, lb in enumerate(tau_sed_labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(tau_sed_labels)}