import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:luanvanapp/bard_ai/bard_ai_screen/ethnic_khmer/bard_ai_controller_khmer.dart';

class AiEthnicKhmer extends StatelessWidget {
const AiEthnicKhmer({ Key? key }) : super(key: key);

@override
  Widget build(BuildContext context) {
    BardAIControllerKhmer controller = Get.put(BardAIControllerKhmer());

    TextEditingController textField = TextEditingController();

    List<String> permissionWord = [
      'khmer',
      'kho-me',
      'chol chnăm thmay',
      'chôl chnăm thmây',
      'ok om bok',
      'sene dolta',
      'ooc oom bok',
      'sene boht',
      'mon an',
      'dam cuoi',
      'marriage',
      'truyen thong',
      'traditional',
      'culture',
      'dish',
    ];

    void showMaterialBanner(BuildContext context) {
      ScaffoldMessenger.of(context).showMaterialBanner(
        MaterialBanner(
          content: const Text(
              'System supports for Khmer ethnic groups. \nPlease ask your questions related to ethnic group'),
          backgroundColor: Colors.yellow,
          actions: [
            TextButton(
                onPressed: () {
                  ScaffoldMessenger.of(context).hideCurrentMaterialBanner();
                },
                child: const Text('CLOSE')),
          ],
        ),
      );
    }

    return Scaffold(
      backgroundColor: const Color(0xfff2f1f9),
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          "AI Chatbot",
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
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            Expanded(
              child: ListView(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                            vertical: 10, horizontal: 20),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: const Text("Please give me your questions \nabout Khmer ethnic group!"),
                      ),
                    ],
                  ),
                  const SizedBox(height: 15),
                  Obx(
                    () => Column(
                      children: controller.historyList
                          .map(
                            (e) => Container(
                              margin: const EdgeInsets.symmetric(vertical: 10),
                              padding: const EdgeInsets.symmetric(
                                  vertical: 10, horizontal: 20),
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Row(
                                children: [
                                  Text(e.system == "user" ? "👨‍💻" : "🤖"),
                                  const SizedBox(width: 10),
                                  Flexible(child: Text(e.message)),
                                ],
                              ),
                            ),
                          )
                          .toList(),
                    ),
                  )
                ],
              ),
            ),
            Container(
              decoration: BoxDecoration(
                color: Colors.blueAccent.withOpacity(0.5),
                borderRadius: BorderRadius.circular(10),
              ),
              height: 60,
              child: Row(
                children: [
                  Expanded(
                    child: TextFormField(
                      controller: textField,
                      decoration: const InputDecoration(
                          hintText: "You: ",
                          border: OutlineInputBorder(
                            borderSide: BorderSide.none,
                          )),
                    ),
                  ),
                  Obx(
                    () => controller.isLoading.value
                        ? const CircularProgressIndicator()
                        : IconButton(
                            onPressed: () {
                              String inputText =
                                  textField.text.trim().toLowerCase();
                              if (inputText.isNotEmpty) {
                                // Kiểm tra xem văn bản đầu vào
                                // ignore: no_leading_underscores_for_local_identifiers
                                bool _permissionWord = permissionWord
                                    .any((word) => inputText.contains(word));

                                if (_permissionWord) {
                                  // Gửi câu hỏi nếu có từ cho phép
                                  controller.sendPrompt(inputText);
                                  textField.clear();
                                } else {
                                  // Hiển thị thông báo
                                  showMaterialBanner(context);
                                }
                              }
                            },
                            icon: const Icon(
                              Icons.send,
                              color: Colors.white,
                            ),
                          ),
                  ),
                  const SizedBox(width: 10)
                ],
              ),
            ),
            const SizedBox(height: 10),
          ],
        ),
      ),
    );
  }
}