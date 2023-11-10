import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

class Cham extends StatefulWidget {
  const Cham({ Key? key }) : super(key: key);

  @override
  _ChamState createState() => _ChamState();
}

class _ChamState extends State<Cham> {
  _launchURL() async {
    final url = Uri.parse('https://nhandan.vn/dan-toc-cham-post723909.html');
    if (await canLaunchUrl(url)) {
      await launchUrl(url);
    } else {
      throw 'Could not launch $url';
    }
  }

  _launchURL1() async {
    final url = Uri.parse(
        'https://vi.wikipedia.org/wiki/Ng%C6%B0%E1%BB%9Di_Ch%C4%83m');
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
          "Dân tộc Chăm",
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
                  'assets/images/cham.jpg',
                ),
                const Padding(
                  padding: EdgeInsets.all(6.0),
                  child: Column(
                    children: [
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Dân tộc Chăm vốn sinh tụ ở duyên hải miền trung Việt Nam từ rất lâu đời, đã từng kiến tạo nên một nền văn hóa rực rỡ với ảnh hưởng sâu sắc của văn hóa Ấn Ðộ. Ngay từ những thế kỷ thứ XVII, người Chăm đã từng xây dựng nên vương quốc Chăm-pa. Tên gọi khác: Chàm, Chiêm, Chiêm Thành, Chăm Pa, Hời.',
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
                            'Theo số liệu Điều tra 53 dân tộc thiểu số 1/4/2019: Tổng dân số: 178.948 người. Trong đó, nam: 87.838 người; nữ: 91.110 người. Tỷ lệ dân số sống ở khu vực nông thôn: 83,8%.',
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
                            'Tiếng nói thuộc nhóm ngôn ngữ Malayô-Polynéxia (ngữ hệ Nam Ðảo).',
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
                            'Người Chăm cư trú tại Ninh Thuận, Bình Thuận.',
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
                    'Xem thêm dân tộc Chăm ở nhandan.vn',
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
                    'Xem thêm dân tộc Chăm ở Wikipedia',
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
