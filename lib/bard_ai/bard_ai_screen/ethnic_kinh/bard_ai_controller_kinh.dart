import 'dart:convert';
import 'package:get/get.dart';
import 'package:http/http.dart' as http;
import 'package:luanvanapp/bard_ai/bard_ai_screen/ethnic_kinh/bard_model_kinh.dart';
import 'package:luanvanapp/bard_ai/bard_ai_screen/data.dart';

class BardAIControllerKinh extends GetxController {
  RxList historyList = RxList<BardModelKinh>();

  RxBool isLoading = false.obs;

  void sendPrompt(String prompt) async {
    isLoading.value = true;
    var newHistory = BardModelKinh(system: "user", message: prompt);
    historyList.add(newHistory);
    try {
      final body = {
        'prompt': {
          'text': prompt,
        },
      };
      final url =
          'https://generativelanguage.googleapis.com/v1beta3/models/text-bison-001:generateText?key=$APIKEY';

      final response = await http.post(
        Uri.parse(url),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data.containsKey("candidates") && data["candidates"].isNotEmpty) {
          final bardReplay = data["candidates"][0]["output"];
          var newHistory2 = BardModelKinh(system: "bard", message: bardReplay);
          historyList.add(newHistory2);
          // ignore: avoid_print
          print(bardReplay.toString());
        } else {
          // ignore: avoid_print
          print('Invalid response format');
        }
      } else {
        // ignore: avoid_print
        print('Request failed with status: ${response.statusCode}');
      }
    } catch (error) {
      // ignore: avoid_print
      print('Error: $error');
    }

    isLoading.value = false;
  }
}
