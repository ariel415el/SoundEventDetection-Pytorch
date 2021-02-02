from dataset.spectogram_features.preprocess import read_multichannel_audio, \
    multichannel_stft, multichannel_complex_to_log_mel
from dataset.spectogram_features import spectogram_configs as cfg
import matplotlib.pyplot as plt
import numpy as np
import soundfile

if __name__ == '__main__':
    # audio_path = '/home/ariel/projects/sound/data/Film_take_clap/original/Meron/S05A-S07AT2.WAV'
    # audio_path = '/home/ariel/projects/sound/data/Film_take_clap/raw/JackRinger-05/161019_1233.wav'
    # audio_path = '/home/ariel/projects/sound/data/Film_take_clap/raw/DyingWithYou/1A-T002.WAV'
    audio_path = '/home/ariel/projects/sound/data/FilmClap/original/Meron/S015-S001T2.WAV'
    # audio_path = 'samples/StillJames_2B-T002.WAV'
    sec_start = 24
    sec_end = 26

    multichannel_waveform = read_multichannel_audio(audio_path=audio_path, target_fs=cfg.working_sample_rate)


    multichannel_waveform = multichannel_waveform[cfg.working_sample_rate * sec_start: cfg.working_sample_rate * sec_end]
    soundfile.write("tmp_file.WAV", multichannel_waveform, cfg.working_sample_rate)
    feature = multichannel_stft(multichannel_waveform)
    feature = multichannel_complex_to_log_mel(feature)

    frames_num = feature.shape[1]
    tick_hop = frames_num // 20
    xticks = np.concatenate((np.arange(0, frames_num - tick_hop, tick_hop), [frames_num]))
    xlabels = [f"{x / cfg.frames_per_second:.3f}s" for x in xticks]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(feature[0].T, origin='lower', cmap='jet')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation='vertical')
    ax.xaxis.set_ticks_position('bottom')
    plt.show()