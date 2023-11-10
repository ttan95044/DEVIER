import 'dart:convert';
import 'package:get/get.dart';
import 'package:http/http.dart' as http;
import 'package:luanvanapp/bard_ai/bard_ai_screen/bard_model.dart';
import 'package:luanvanapp/bard_ai/bard_ai_screen/data.dart';

class BardAIController extends GetxController {
  RxList historyList = RxList<BardModel>();

  RxBool isLoading = false.obs;

  void sendPrompt(String prompt) async {
    isLoading.value = true;
    var newHistory = BardModel(system: "user", message: prompt);
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
          var newHistory2 = BardModel(system: "bard", message: bardReplay);
          historyList.add(newHistory2);
          print(bardReplay.toString());
        } else {
          print('Invalid response format');
        }
      } else {
        print('Request failed with status: ${response.statusCode}');
      }
    } catch (error) {
      print('Error: $error');
    }

    isLoading.value = false;
  }
}
