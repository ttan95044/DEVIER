import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

class Khac extends StatefulWidget {
  const Khac({Key? key}) : super(key: key);

  @override
  _KhacState createState() => _KhacState();
}

class _KhacState extends State<Khac> {
  _launchURL() async {
    final url = Uri.parse('https://nhandan.vn/cong-dong-54-dan-toc.html');
    if (await canLaunchUrl(url)) {
      await launchUrl(url);
    } else {
      throw 'Could not launch $url';
    }
  }

  _launchURL1() async {
    final url = Uri.parse(
        'https://vi.wikipedia.org/wiki/Danh_m%E1%BB%A5c_c%C3%A1c_d%C3%A2n_t%E1%BB%99c_Vi%E1%BB%87t_Nam');
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
          "Dân tộc Việt Nam",
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
                  'assets/images/khac.jpg',
                ),
                const Padding(
                  padding: EdgeInsets.all(6.0),
                  child: Column(
                    children: [
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Việt Nam là một quốc gia đa dân tộc. Cộng đồng các dân tộc thiểu số Việt Nam được hình thành và phát triển cùng với tiến trình lịch sử hàng nghìn năm dựng nước và giữ nước của cả dân tộc. Bởi vậy, có thể khẳng định rằng, lịch sử hình thành, phát triển các dân tộc gắn với lịch sử hình thành, phát triển đất nước Việt Nam. Nghiên cứu lịch sử hình thành, phát triển các dân tộc không thể tách rời lịch sử hình thành, phát triển cộng đồng dân tộc Việt Nam, trong đó bao gồm cả dân tộc đa số và các dân tộc thiểu số.',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                      SizedBox(height: 15,),
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Khái niệm văn hóa có rất nhiều định nghĩa, sau Thế chiến thứ hai - một thời kỳ ồn ào, xáo động của các nhà văn hóa học, nhiều người đã nhận ra rằng cách tiếp cận hệ thống về văn hóa có nhiều ưu điểm hơn cả. Văn hóa là cả một hệ thống tổng thể quy định con đường sống của một dân tộc. Hệ thống này bao gồm toàn bộ những gì thuộc về tư duy, triết học, tín ngưỡng, phong tục tập quán, nghệ thuật văn học...',
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
                    'Xem thêm 54 dân tộc Việt Nam ở nhandan.vn',
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
                    'Xem thêm 54 dân tộc Việt Nam ở Wikipedia',
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
