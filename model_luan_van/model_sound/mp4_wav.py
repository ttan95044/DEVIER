from moviepy.editor import VideoFileClip


def convert_mp4_to_wav(input_path, output_path):
    video = VideoFileClip(input_path)
    audio = video.audio
    audio.write_audiofile(output_path)


# Đường dẫn tệp MP4 đầu vào và tệp WAV đầu ra
input_file = 'D:/luan-van-data/video/khac/khac (27).mp4'
output_file = 'D:/luan-van-data/sound_wav/khac/khac (27).wav'

# Gọi hàm chuyển đổi
convert_mp4_to_wav(input_file, output_file)
