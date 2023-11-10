import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

class Hoa extends StatefulWidget {
  const Hoa({Key? key}) : super(key: key);

  @override
  // ignore: library_private_types_in_public_api
  _HoaState createState() => _HoaState();
}

class _HoaState extends State<Hoa> {
  _launchURL() async {
    final url = Uri.parse('https://nhandan.vn/dan-toc-hoa-post723900.html');
    if (await canLaunchUrl(url)) {
      await launchUrl(url);
    } else {
      throw 'Could not launch $url';
    }
  }

  _launchURL1() async {
    final url = Uri.parse(
        'https://vi.wikipedia.org/wiki/Ng%C6%B0%E1%BB%9Di_Hoa_(Vi%E1%BB%87t_Nam)');
    if (await canLaunchUrl(url)) {
      await launchUrl(url);
    } else {
      throw 'Could not launch $url';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          "Dân tộc Hoa",
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
          Center(
            child: Column(
              children: <Widget>[
                Image.asset(
                  'assets/images/hoa.jpg',
                ),
                const Padding(
                  padding: EdgeInsets.all(6.0),
                  child: Column(
                    children: [
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Người Hoa di cư đến Việt Nam vào những thời điểm khác nhau từ thế kỷ XVI, và sau này vào cuối thời Minh, đầu thời Thanh, kéo dài cho đến nửa đầu thế kỷ XX, tên gọi khác: Khách, Hán, Tàu.',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                      SizedBox(
                        height: 15,
                      ),
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Theo số liệu Điều tra 53 dân tộc thiểu số 1/4/2019: Tổng dân số: 749.466 người, trong đó, có 389.651 nam, 359.815 nữ. Số hộ dân cư gồm 241.822 hộ. Tỷ lệ dân số sống ở khu vực nông thôn chiếm 30,3%.',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                      SizedBox(
                        height: 15,
                      ),
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Tiếng nói thuộc nhóm ngôn ngữ Hán (Ngữ hệ Hán-Tạng)',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                      SizedBox(
                        height: 15,
                      ),
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Người Hoa thiên di vào cả miền bắc và miền nam Việt Nam từ nhiều địa phương, bằng nhiều con đường, trong những thời gian khác nhau, kéo dài suốt từ thời kỳ bắc thuộc cho đến năm 1954.',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(
                  height: 20,
                ),
                InkWell(
                  onTap: _launchURL,
                  child: const Text(
                    'Xem thêm dân tộc Hoa ở nhandan.vn',
                    style: TextStyle(
                      color: Colors.blue,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                ),
                const SizedBox(
                  height: 20,
                ),
                InkWell(
                  onTap: _launchURL1,
                  child: const Text(
                    'Xem thêm dân tộc Hoa ở Wikipedia',
                    style: TextStyle(
                      color: Colors.blue,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                ),
                const SizedBox(
                  height: 50,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
