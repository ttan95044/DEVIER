import soundfile as sf


def cut_wav_file(input_file, output_directory, duration):
    # Đọc file âm thanh WAV đầu vào
    data, samplerate = sf.read(input_file)

    # Tính toán số lượng phần cần cắt
    num_parts = int(len(data) / (samplerate * duration)) + 1

    count = 1

    # Cắt file âm thanh thành các phần nhỏ
    for i in range(num_parts):
        start = i * samplerate * duration
        end = min((i + 1) * samplerate * duration, len(data))
        part = data[start:end]

        # Tạo tên tệp đầu ra cho từng phần
        output_file = output_directory + '/khac (27) {}.wav'.format(count)
        count += 1

        # Ghi phần âm thanh nhỏ vào tệp đầu ra
        sf.write(output_file, part, samplerate)

        print("Created part {}: {}".format(count, output_file))


# Sử dụng hàm cut_wav_file
# Đường dẫn đến tệp âm thanh WAV đầu vào
input_file = 'D:/luan-van-data/sound_wav/khac/khac (27).wav'
output_directory = 'D:/luan-van-data/sound_cut/khac'  # Thư mục đầu ra để lưu các tệp nhỏ
duration = 10  # Độ dài mỗi phần cắt (giây)

cut_wav_file(input_file, output_directory, duration)
