import 'package:flutter/material.dart';
import 'package:luanvanapp/home/home_content/home_content.dart';
import 'package:luanvanapp/navbar/navbar.dart';

class HomeScreen extends StatelessWidget {
  // ignore: use_key_in_widget_constructors
  const HomeScreen({Key? key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/images/nen.jpg'),
            fit: BoxFit.cover,
          ),
        ),
        child: ListView(
          children: const [
            Navbar(),
            HomeContent()
          ],
        ),
      ),
    );
  }
}
