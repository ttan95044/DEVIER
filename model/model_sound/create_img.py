import numpy as np
import librosa.display
import os
import matplotlib.pyplot as plt



def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)


def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)


# wav->spectrograms 10s
create_pngs_from_wavs('D:/luan-van-data/sound_cut/cham', 'D:/luan-van-data/sound_img/cham')
create_pngs_from_wavs('D:/luan-van-data/sound_cut/hoa', 'D:/luan-van-data/sound_img/hoa')
create_pngs_from_wavs('D:/luan-van-data/sound_cut/khac', 'D:/luan-van-data/sound_img/khac')
create_pngs_from_wavs('D:/luan-van-data/sound_cut/khmer', 'D:/luan-van-data/sound_img/khmer')
create_pngs_from_wavs('D:/luan-van-data/sound_cut/kinh', 'D:/luan-van-data/sound_img/kinh')


