import 'package:flutter/material.dart';

class Navbar extends StatelessWidget {
  const Navbar({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(
        top: 20,
        bottom: 10,
      ),
      child: Center(
        child: SizedBox(
          width: 600,
          height: 100, // Đặt chiều cao của Container
          child: Card(
            color: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(50.0),
            ),
            child: const Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                Text(
                  "Tìm kiếm dân tộc",
                  style: TextStyle(
                    fontWeight: FontWeight.w800,
                    color: Color.fromARGB(255, 0, 119, 255),
                    fontSize: 50,
                    fontFamily: 'GR',
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
