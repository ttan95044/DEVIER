import 'package:flutter/material.dart';
import 'package:luanvanapp/bard_ai/bard_ai_screen/ethnic_cham/ai_ethnic_cham.dart';
import 'package:luanvanapp/bard_ai/bard_ai_screen/ethnic_hoa/ai_ethnic_hoa.dart';
import 'package:luanvanapp/bard_ai/bard_ai_screen/ethnic_khmer/ai_ethnic_khmer.dart';
import 'package:luanvanapp/bard_ai/bard_ai_screen/ethnic_kinh/ai_ethnic_kinh.dart';

class BardAi extends StatelessWidget {
  const BardAi({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          "Select the ethnicity",
          style: TextStyle(
              fontWeight: FontWeight.bold,
              color: Color.fromARGB(255, 0, 119, 255)),
        ),
        backgroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(
            Icons.arrow_back,
            color: Color.fromARGB(255, 0, 119, 255),
          ),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              "Ask questions about enthnic",
              style: TextStyle(fontSize: 20),
            ),
            const SizedBox(
              height: 20,
            ),
            SizedBox(
              width: 200,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const AiEthnicKinh()),
                  );
                },
                child: const Text(
                  'Kinh',
                  style: TextStyle(fontSize: 20),
                ),
              ),
            ),
            SizedBox(
              width: 200,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const AiEthnicCham()),
                  );
                },
                child: const Text(
                  'Cham',
                  style: TextStyle(fontSize: 20),
                ),
              ),
            ),
            SizedBox(
              width: 200,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const AiEthnicHoa()),
                  );
                },
                child: const Text(
                  'Hoa',
                  style: TextStyle(fontSize: 20),
                ),
              ),
            ),
            SizedBox(
              width: 200,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const AiEthnicKhmer()),
                  );
                },
                child: const Text(
                  'Khmer',
                  style: TextStyle(fontSize: 20),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
