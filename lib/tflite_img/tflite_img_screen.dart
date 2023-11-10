import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:luanvanapp/tflite_img/tflite_culture/tflite_culture_festival.dart';
import 'package:luanvanapp/tflite_img/tflite_culture/tflite_culture_marriage.dart';
import 'package:luanvanapp/tflite_img/tflite_culture/tflite_culture_temple.dart';
import 'package:luanvanapp/tflite_img/tflite_culture/tflite_culture_dance.dart';
import 'package:luanvanapp/tflite_img/tflite_culture/tflite_culture_floating_market.dart';
import 'package:tflite/tflite.dart';

class TfliteImgScreen extends StatefulWidget {
  const TfliteImgScreen({Key? key}) : super(key: key);

  @override
  // ignore: library_private_types_in_public_api
  _TfliteImgScreenState createState() => _TfliteImgScreenState();
}

class _TfliteImgScreenState extends State<TfliteImgScreen> {
  late File _image;
  late List _results = [];
  bool imageSelect = false;
  int numResults = 5;
  double threshold = 0.05;
  double imageMean = 128;
  double imageStd = 128;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  Future loadModel() async {
    Tflite.close();
    String res = (await Tflite.loadModel(
      model: "assets/model/mobilenet_img/mobile_net_img.tflite",
      labels: "assets/model/mobilenet_img/labels_img.txt",
    ))!;
    // ignore: avoid_print
    print("Models loading status: $res");
  }

  Future imageClassification(File image) async {
    final List? recognitions = await Tflite.runModelOnImage(
      path: image.path,
      numResults: numResults,
      threshold: threshold,
      imageMean: imageMean,
      imageStd: imageStd,
    );
    setState(() {
      _results = recognitions!;
      _image = image;
      imageSelect = true;
    });
  }

  Future pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? pickedFile = await picker.pickImage(
      source: ImageSource.gallery,
    );
    File image = File(pickedFile!.path);
    imageClassification(image);
  }

  Future captureImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? pickedFile = await picker.pickImage(
      source: ImageSource.camera,
    );
    File image = File(pickedFile!.path);
    imageClassification(image);
  }

  void navigateToDetailScreen(File image, String label) {
    if (label == "mua") {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (context) => TfliteCultureDance(image: image),
        ),
      );
    } else if (label == "cuoi") {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (context) => TfliteCultureMarriage(image: image),
        ),
      );
    } else if (label == "chua") {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (context) => TfliteCultureTemple(image: image),
        ),
      );
    } else if (label == "le") {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (context) => TfliteCultureFestival(image: image),
        ),
      );
    } else if (label == "cho_noi") {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (context) => TfliteCultureFloatingMarket(image: image),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          "Tìm kiếm bằng hình ảnh",
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
          if (imageSelect)
            Container(
              width: 300,
              height: 300,
              margin: const EdgeInsets.all(10),
              child: Image.file(_image),
            )
          else
            Container(
              margin: const EdgeInsets.all(10),
              child: const Opacity(
                opacity: 0.8,
                child: Center(
                  child: Text("Hãy chọn hình ảnh hoặc chụp"),
                ),
              ),
            ),
          const SizedBox(height: 10,),
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
                        navigateToDetailScreen(_image, label);
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
                    if (label == "mua")
                      ElevatedButton(
                        onPressed: () {
                          navigateToDetailScreen(_image, label);
                        },
                        child: const Text("Xem dân chi tiết dân tộc của ảnh"),
                      ),
                    if (label == "cuoi")
                      ElevatedButton(
                        onPressed: () {
                          navigateToDetailScreen(_image, label);
                        },
                        child: const Text("Xem dân chi tiết dân tộc của ảnh"),
                      ),
                    if (label == "le")
                      ElevatedButton(
                        onPressed: () {
                          navigateToDetailScreen(_image, label);
                        },
                        child: const Text("Xem dân chi tiết dân tộc của ảnh"),
                      ),
                    if (label == "chua")
                      ElevatedButton(
                        onPressed: () {
                          navigateToDetailScreen(_image, label);
                        },
                        child: const Text("Xem dân chi tiết dân tộc của ảnh"),
                      ),
                    if (label == "cho_noi")
                      ElevatedButton(
                        onPressed: () {
                          navigateToDetailScreen(_image, label);
                        },
                        child: const Text("Xem dân chi tiết dân tộc của ảnh"),
                      ),
                  ],
                ),
              );
            },
          ),
          Column(
            children: [
              const Text("Số lượng kết quả hiển thị:"),
              Padding(
                padding: const EdgeInsets.only(left: 40, right: 40),
                child: TextFormField(
                  initialValue: numResults.toString(),
                  onChanged: (value) {
                    if (int.tryParse(value) != null) {
                      setState(() {
                        numResults = int.parse(value);
                      });
                    }
                  },
                  keyboardType: TextInputType.number,
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
          Column(
            children: [
              const Text("Ngưỡng lọc kết quả (số X * 100 = X%):"),
              Padding(
                padding: const EdgeInsets.only(left: 40, right: 40),
                child: TextFormField(
                  initialValue: threshold.toString(),
                  onChanged: (value) {
                    if (int.tryParse(value) != null) {
                      setState(() {
                        threshold = double.parse(value);
                      });
                    }
                  },
                  keyboardType: TextInputType.number,
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
          Column(
            children: [
              const Text("chuẩn hóa hình ảnh:"),
              Padding(
                padding: const EdgeInsets.only(left: 40, right: 40),
                child: TextFormField(
                  initialValue: imageMean.toString(),
                  onChanged: (value) {
                    if (int.tryParse(value) != null) {
                      setState(() {
                        imageMean = double.parse(value);
                      });
                    }
                  },
                  keyboardType: TextInputType.number,
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
          Column(
            children: [
              const Text("chuẩn hóa hình ảnh:"),
              Padding(
                padding: const EdgeInsets.only(left: 40, right: 40),
                child: TextFormField(
                  initialValue: imageStd.toString(),
                  onChanged: (value) {
                    if (int.tryParse(value) != null) {
                      setState(() {
                        imageStd = double.parse(value);
                      });
                    }
                  },
                  keyboardType: TextInputType.number,
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
          const SizedBox(
            height: 30,
          ),
        ],
      ),
      bottomNavigationBar: BottomAppBar(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            ElevatedButton(
              onPressed: pickImage,
              child: const Text("Upload File"),
            ),
            ElevatedButton(
              onPressed: captureImage,
              child: const Text("Capture Image"),
            ),
          ],
        ),
      ),
    );
  }
}
