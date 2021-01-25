from dataset.spectogram_features.preprocess import LogMelExtractor, read_audio
from dataset.spectogram_features import spectogram_configs as cfg
import matplotlib.pyplot as plt
import numpy as np
import soundfile

if __name__ == '__main__':
    # audio_path = '/home/ariel/projects/sound/data/Film_take_clap/original/Meron/S05A-S07AT2.WAV'
    audio_path = '/home/ariel/projects/sound/data/Film_take_clap/raw/JackRinger-05/161019_1233.wav'
    audio_path = '/home/ariel/projects/sound/data/Film_take_clap/raw/DyingWithYou/1A-T002.WAV'
    # audio_path = 'samples/StillJames_2B-T002.WAV'
    sec_start = 4
    sec_end = 13
    feature_extractor = LogMelExtractor(
        sample_rate=cfg.working_sample_rate,
        nfft=cfg.NFFT,
        window_size=cfg.frame_size,
        hop_size=cfg.hop_size,
        mel_bins=cfg.mel_bins,
        fmin=cfg.mel_min_freq,
        fmax=cfg.mel_max_freq)

    multichannel_audio = read_audio(audio_path=audio_path, target_fs=cfg.working_sample_rate)
    multichannel_audio = multichannel_audio[cfg.working_sample_rate * sec_start: cfg.working_sample_rate * sec_end]
    soundfile.write("tmp_file.WAV", multichannel_audio, cfg.working_sample_rate)
    mel_features = feature_extractor.transform_multichannel(multichannel_audio)[0].T

    frames_num = mel_features.shape[1]
    tick_hop = frames_num // 20
    xticks = np.concatenate((np.arange(0, frames_num - tick_hop, tick_hop), [frames_num]))
    xlabels = [f"{x / cfg.frames_per_second:.3f}s" for x in xticks]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(mel_features, origin='lower', cmap='jet')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation='vertical')
    ax.xaxis.set_ticks_position('bottom')
    plt.show()