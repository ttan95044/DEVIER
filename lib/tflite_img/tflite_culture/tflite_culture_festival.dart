import 'dart:io';

import 'package:flutter/material.dart';
import 'package:luanvanapp/ethnic/cham.dart';
import 'package:luanvanapp/ethnic/hoa.dart';
import 'package:luanvanapp/ethnic/khac.dart';
import 'package:luanvanapp/ethnic/khmer.dart';
import 'package:luanvanapp/ethnic/kinh.dart';
import 'package:tflite/tflite.dart';

class TfliteCultureFestival extends StatefulWidget {
  final File image;
  const TfliteCultureFestival({Key? key, required this.image})
      : super(key: key);

  @override
  _TfliteCultureFestivalState createState() => _TfliteCultureFestivalState();
}

class _TfliteCultureFestivalState extends State<TfliteCultureFestival> {
  late List _results = [];
  int numResults = 5;
  double threshold = 0.05;
  double imageMean = 128;
  double imageStd = 128;
  bool predictionDone = false;
  @override
  void initState() {
    super.initState();
    loadModel(); // Load mô hình TFLite khi trang được khởi tạo
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  Future loadModel() async {
    Tflite.close();
    await Tflite.loadModel(
      model: "assets/model/mobilenet_festival/mobile_net_festival.tflite",
      labels: "assets/model/mobilenet_festival/labels_festival.txt",
    );
    print("Models loaded successfully");
  }

  Future predict() async {
    if (!predictionDone) {
      final List? recognitions = await Tflite.runModelOnImage(
        path: widget.image.path,
        numResults: numResults,
        threshold: threshold,
        imageMean: imageMean,
        imageStd: imageStd,
      );
      setState(() {
        _results = recognitions!;
      });
      predictionDone = true;
    }
  }

  void navigateToDetailScreen(String label) {
    if (label == "kinh") {
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => Kinh(),
        ),
      );
    } else if (label == "hoa") {
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => Hoa(),
        ),
      );
    } else if (label == "cham") {
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => Cham(),
        ),
      );
    } else if (label == "khmer") {
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => Khmer(),
        ),
      );
    } else if (label == "khac") {
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => Khac(),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    predict();

    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          "Lễ hội",
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
      body: Center(
        child: Column(
          children: <Widget>[
            Image.file(
              widget.image,
              width: 300,
              height: 300,
            ),
            ListView.builder(
              shrinkWrap: true,
              itemCount: _results.length,
              itemBuilder: (context, index) {
                final result = _results[index];
                final label = result['label'];
                final confidence = result['confidence'] * 100;
                return Card(
                  child: Column(
                    children: [
                      InkWell(
                        onTap: () {
                          navigateToDetailScreen(label);
                        },
                        child: Container(
                          margin: const EdgeInsets.all(10),
                          child: Text(
                            "$label - ${confidence.toStringAsFixed(2)}%",
                            style: const TextStyle(
                              color: Colors.red,
                              fontSize: 20,
                            ),
                          ),
                        ),
                      ),
                      if (label == "kinh")
                        ElevatedButton(
                          onPressed: () {
                            navigateToDetailScreen(label);
                          },
                          child: const Text("Xem dân chi tiết dân tộc Kinh"),
                        ),
                      if (label == "hoa")
                        ElevatedButton(
                          onPressed: () {
                            navigateToDetailScreen(label);
                          },
                          child: const Text("Xem dân chi tiết dân tộc Hoa"),
                        ),
                      if (label == "cham")
                        ElevatedButton(
                          onPressed: () {
                            navigateToDetailScreen(label);
                          },
                          child: const Text("Xem dân chi tiết dân tộc Champa"),
                        ),
                      if (label == "khmer")
                        ElevatedButton(
                          onPressed: () {
                            navigateToDetailScreen(label);
                          },
                          child: const Text("Xem dân chi tiết dân tộc Khmer"),
                        ),
                      if (label == "khac")
                        ElevatedButton(
                          onPressed: () {
                            navigateToDetailScreen(label);
                          },
                          child:
                              const Text("Xem dân chi tiết những dân tộc khác"),
                        ),
                    ],
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}
