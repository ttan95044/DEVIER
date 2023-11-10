import 'dart:async';
import 'dart:convert';
import 'package:audioplayers/audioplayers.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:luanvanapp/ethnic/cham.dart';
import 'package:luanvanapp/ethnic/hoa.dart';
import 'package:luanvanapp/ethnic/khac.dart';
import 'package:luanvanapp/ethnic/khmer.dart';
import 'dart:io';

import 'package:luanvanapp/ethnic/kinh.dart';

class SoundPrediction {
  final String topLabel;

  SoundPrediction(this.topLabel);

  factory SoundPrediction.fromJson(Map<String, dynamic> json) {
    return SoundPrediction(
      json['top_label'] as String,
    );
  }
}

class TfliteSoundScreen extends StatefulWidget {
  const TfliteSoundScreen({Key? key}) : super(key: key);

  @override
  // ignore: library_private_types_in_public_api
  _TfliteSoundScreenState createState() => _TfliteSoundScreenState();
}

class _TfliteSoundScreenState extends State<TfliteSoundScreen> {
  File? audioFile;
  List<SoundPrediction>? predictions;
  Image? convertedImage;
  final audioPlayer = AudioPlayer();
  bool isPlaying = false;
  Duration duration = Duration.zero;
  Duration position = Duration.zero;

  String formatTime(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final hours = twoDigits(duration.inHours);
    final minutes = twoDigits(duration.inMinutes.remainder(60));
    final seconds = twoDigits(duration.inSeconds.remainder(60));

    return [
      if (duration.inHours > 0) hours,
      minutes,
      seconds,
    ].join(':');
  }

  void selectAndUploadAudio() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['mp3', 'wav', 'aac'],
    );

    if (result != null) {
      audioFile = File(result.files.single.path!);

      await uploadAudio(audioFile!);
      await fetchPredictions();
      getImage();
      setAudio();
    }
  }

  @override
  void initState() {
    super.initState();

    audioPlayer.onPlayerStateChanged.listen(
      (state) {
        setState(() {
          isPlaying = state == PlayerState.PLAYING;
        });
      },
    );

    audioPlayer.onDurationChanged.listen(
      (newDuration) {
        setState(() {
          duration = newDuration;
        });
      },
    );

    audioPlayer.onAudioPositionChanged.listen(
      (newPosition) {
        setState(() {
          position = newPosition;
        });
      },
    );
    convertedImage = null;
  }

  Future setAudio() async {
    audioPlayer.setReleaseMode(ReleaseMode.LOOP);
    // ignore: avoid_print
    print(audioFile);

    if (audioFile != null) {
      final file = audioFile;
      audioPlayer.setUrl(file!.path, isLocal: true);
    }
  }

  @override
  void dispose() {
    audioPlayer.dispose();

    super.dispose();
  }

  Future<void> uploadAudio(File audioFile) async {
    final url = Uri.parse('http://192.168.1.3:5000/process_audio');
    final request = http.MultipartRequest('POST', url);
    request.files
        .add(await http.MultipartFile.fromPath('audio', audioFile.path));

    final response = await request.send();
    if (response.statusCode == 200) {
      final responseText = await response.stream.bytesToString();
      // ignore: avoid_print
      print(responseText);
    } else {
      // ignore: avoid_print
      print('Lỗi khi tải lên video: ${response.statusCode}');
    }
  }

  Future<void> getImage() async {
    const imageUrl =
        'http://192.168.1.3:5000/static/spectrogram_part_1.wav.png';
    // ignore: avoid_print
    print(imageUrl);
    setState(() {
      convertedImage = Image.network(imageUrl);
    });
  }

  Future<void> fetchPredictions() async {
    final url = Uri.parse('http://192.168.1.3:5000/get_prediction');
    final response = await http.get(url);

    if (response.statusCode == 200) {
      final responseText = response.body;
      // ignore: avoid_print
      print(responseText);

      if (responseText.isNotEmpty) {
        final jsonResponse = json.decode(responseText);
        final topLabel = jsonResponse['top_label'] as String;
        // ignore: avoid_print
        print('Top Label: $topLabel');

        // Cập nhật predictions và thông báo cập nhật UI
        setState(() {
          predictions = [SoundPrediction(topLabel)];
        });
      }
    } else {
      // ignore: avoid_print
      print('Lỗi khi lấy dự đoán: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          "Tìm kiếm bằng âm thanh",
          style: TextStyle(fontWeight: FontWeight.bold, color: Color.fromARGB(255, 0, 119, 255)),
        ),
        backgroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back,color: Color.fromARGB(255, 0, 119, 255),),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
      ),
      body: ListView(
        children: [
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 300,
                  height: 300,
                  child: convertedImage,
                ),
                Slider(
                  min: 0,
                  max: duration.inSeconds.toDouble(),
                  value: position.inSeconds.toDouble(),
                  onChanged: (value) async {
                    final position = Duration(seconds: value.toInt());
                    await audioPlayer.seek(position);
                    await audioPlayer.resume();
                  },
                ),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        formatTime(position),
                      ),
                      Text(
                        formatTime(duration - position),
                      )
                    ],
                  ),
                ),
                CircleAvatar(
                  radius: 35,
                  child: IconButton(
                    icon: Icon(
                      isPlaying ? Icons.pause : Icons.play_arrow,
                    ),
                    iconSize: 30,
                    onPressed: () async {
                      if (isPlaying) {
                        await audioPlayer.pause();
                      } else {
                        await audioPlayer.resume();
                      }
                    },
                  ),
                ),
                ElevatedButton(
                  onPressed: selectAndUploadAudio,
                  child: const Text('Chọn đoạn âm thanh ( mp3, wav )'),
                ),
                if (predictions != null)
                  Column(
                    children: predictions!.map((prediction) {
                      return Text(
                        prediction.topLabel == "hoa"
                            ? "Âm thanh của dân tộc: Hoa"
                            : prediction.topLabel == "cham"
                                ? "Âm thanh của dân tộc: Champa"
                                : prediction.topLabel == "kinh"
                                    ? "Âm thanh của dân tộc: Kinh"
                                    : prediction.topLabel == "khmer"
                                        ? "Âm thanh của dân tộc: Khmer"
                                        : prediction.topLabel == "khac"
                                            ? "Âm thanh của dân tộc: Dân tộc Khác"
                                            : prediction.topLabel,
                        style: const TextStyle(fontSize: 16, color: Colors.red),
                      );
                    }).toList(),
                  ),
                ElevatedButton(
                  onPressed: () {
                    if (predictions != null && predictions!.isNotEmpty) {
                      final topLabel = predictions![0].topLabel;
                      switch (topLabel) {
                        case "kinh":
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const Kinh()),
                          );
                          break;
                        case "cham":
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const Cham()),
                          );
                          break;
                        case "hoa":
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const Hoa()),
                          );
                          break;
                        case "khmer":
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const Khmer()),
                          );
                          break;
                        case "khac":
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const Khac()),
                          );
                          break;
                      }
                    }
                  },
                  child: const Text('Xem dân tộc đã dữ đoán'),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}
