import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:luanvanapp/bard_ai/bard_ai_screen/bard_ai_controller.dart';

class BardAiScreen extends StatelessWidget {
  const BardAiScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    BardAIController controller = Get.put(BardAIController());
    TextEditingController textField = TextEditingController();
    return Scaffold(
      backgroundColor: const Color(0xfff2f1f9),
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          "AI Chatbot",
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
                      padding:
                          const EdgeInsets.symmetric(vertical: 10, horizontal: 20),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child:
                          const Text("Please give me your questions!"),
                    ),
                  ],
                ),
                const SizedBox(height: 15),
  
                Obx(() => Column(
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
                    ))
              ],
            )),
            Container(
              decoration: BoxDecoration(
                color: Colors.blueAccent.withOpacity(0.5),
                borderRadius: BorderRadius.circular(10),
              ),
              height: 60,
              child: Row(children: [
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
                            if (textField.text != "") {
                              controller.sendPrompt(textField.text);
                              textField.clear();
                            }
                          },
                          icon: const Icon(
                            Icons.send,
                            color: Colors.white,
                          )),
                ),
                const SizedBox(width: 10)
              ]),
            ),
            const SizedBox(height: 10),
          ],
        ),
      ),
    );
  }
}
