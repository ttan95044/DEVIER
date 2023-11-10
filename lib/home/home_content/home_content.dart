import 'package:flutter/material.dart';
import 'package:luanvanapp/bard_ai/bard_ai.dart';
import 'package:luanvanapp/tflite_img/tflite_img_screen.dart';
import 'package:luanvanapp/tflite_sound/tflite_sound_screen.dart';

class HomeContent extends StatelessWidget {
  const HomeContent({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(
        top: 10,
        bottom: 10,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          const SizedBox(
            height: 120,
          ),
          SizedBox(
            width: 380, // Set the desired width for the button
            child: MaterialButton(
              color: Colors.white,
              shape: const RoundedRectangleBorder(
                borderRadius: BorderRadius.all(Radius.circular(40.0)),
              ),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const BardAi(),
                  ),
                );
              },
              height: 90,
              child: const Padding(
                padding: EdgeInsets.symmetric(
                  vertical: 10.0,
                  horizontal: 10.0,
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.search,
                      color: Color.fromARGB(255, 0, 119, 255),
                      size: 36,
                    ),
                    SizedBox(
                      width: 10,
                    ),
                    Text(
                      "Tìm kiếm bằng Google Bard",
                      style: TextStyle(
                        color: Color.fromARGB(255, 0, 119, 255),
                        fontSize: 23,
                        fontWeight: FontWeight.w600,
                        fontFamily: 'pv',
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(
            height: 30,
          ),
          SizedBox(
            width: 380, // Set the desired width for the button
            child: MaterialButton(
              color: Colors.white,
              shape: const RoundedRectangleBorder(
                borderRadius: BorderRadius.all(Radius.circular(40.0)),
              ),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const TfliteImgScreen(),
                  ),
                );
              },
              height: 90,
              child: const Padding(
                padding: EdgeInsets.symmetric(
                  vertical: 10.0,
                  horizontal: 10.0,
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.image,
                      color: Color.fromARGB(255, 0, 119, 255),
                      size: 36,
                    ),
                    SizedBox(
                      width: 10,
                    ),
                    Text(
                      "Tìm kiếm bằng hình ảnh",
                      style: TextStyle(
                        color: Color.fromARGB(255, 0, 119, 255),
                        fontSize: 26,
                        fontWeight: FontWeight.w600,
                        fontFamily: 'pv',
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(
            height: 20,
          ),
          SizedBox(
            width: 380, // Set the desired width for the button
            child: MaterialButton(
              color: Colors.white,
              shape: const RoundedRectangleBorder(
                borderRadius: BorderRadius.all(Radius.circular(40.0)),
              ),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const TfliteSoundScreen(),
                  ),
                );
              },
              height: 90,
              child: const Padding(
                padding: EdgeInsets.symmetric(
                  vertical: 10.0,
                  horizontal: 10.0,
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.music_note,
                      color: Color.fromARGB(255, 0, 119, 255),
                      size: 36,
                    ),
                    SizedBox(
                      width: 10,
                    ),
                    Text(
                      "Tìm kiếm bằng âm thanh",
                      style: TextStyle(
                        color: Color.fromARGB(255, 0, 119, 255),
                        fontSize: 26,
                        fontWeight: FontWeight.w600,
                        fontFamily: 'pv',
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
