working_sample_rate = 48000  # resample all waveforms to this sampling rate
NFFT = 2 ** 10  # The size of the padded frames on which fft will actualy work. Set this to a power of two for faster preprocessing
frame_size = 1000  # Size of frames on to extract spectogram form
hop_size = 500  # Gap between frames: there are (sample_rate / hop_size) frames per second
mel_bins = 64    # How much frames to stretch over the
mel_min_freq = 20  # Hz first mel bin (minimal possible value 0)
mel_max_freq = working_sample_rate // 2  # Hz last mel bin (maximal possible value sampling_rate / 2)
audio_channels = 1   # Restrict data to only the audio_channels first channels of the audio file

frames_per_second = working_sample_rate // hop_size
train_crop_size = frames_per_second * 10  # 10-second log mel spectrogram as input

# Clap details:
time_margin = 0.125  # Time gap around the sepecified event point to be considered as a True s

# Tau-SED details:
# The label configuration is the same as https://github.com/sharathadavanne/seld-dcase2019
# tau_sed_labels = ['knock', 'drawer', 'clearthroat', 'phone', 'keysDrop', 'speech',
#           'keyboard', 'pageturn', 'cough', 'doorslam', 'laughter']

# tau_sed_labels = ['knock', 'keysDrop', 'doorslam']
tau_sed_labels = ['doorslam']
classes_num = len(tau_sed_labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(tau_sed_labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(tau_sed_labels)}
