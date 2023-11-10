import 'dart:io';

import 'package:flutter/material.dart';
import 'package:luanvanapp/ethnic/kinh.dart';

class TfliteCultureFloatingMarket extends StatefulWidget {
  final File image;
  const TfliteCultureFloatingMarket({Key? key, required this.image})
      : super(key: key);

  @override
  // ignore: library_private_types_in_public_api
  _TfliteCultureFloatingMarketState createState() =>
      _TfliteCultureFloatingMarketState();
}

class _TfliteCultureFloatingMarketState
    extends State<TfliteCultureFloatingMarket> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          "Chợ nổi",
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
          children: [
            Image.file(
              widget.image,
              width: 300,
              height: 300,
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pushReplacement(
                  MaterialPageRoute(
                    builder: (context) => const Kinh(),
                  ),
                );
              },
              child: const Text('Chợ nổi của dân tộc Kinh xem thêm thông tin'),
            ),
          ],
        ),
      ),
    );
  }
}
