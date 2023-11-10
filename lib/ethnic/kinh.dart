import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

class Kinh extends StatelessWidget {
  const Kinh({Key? key}) : super(key: key);

  _launchURL() async {
    final url = Uri.parse('https://nhandan.vn/dan-toc-kinh-post723893.html');
    if (await canLaunchUrl(url)) {
      await launchUrl(url);
    } else {
      throw 'Could not launch $url';
    }
  }

  _launchURL1() async {
    final url = Uri.parse(
        'https://vi.wikipedia.org/wiki/Ng%C6%B0%E1%BB%9Di_Vi%E1%BB%87t');
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
          "Dân tộc Kinh (Việt)",
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
                  'assets/images/kinh.jpg',
                ),
                const Padding(
                  padding: EdgeInsets.all(6.0),
                  child: Column(
                    children: [
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Người Kinh (cũng như người Mường, Thổ, Chứt) là tộc người bản địa, sở tại, sinh sống từ rất lâu đời trên dải đất Việt Nam, không phải có nguồn gốc từ lãnh thổ bên ngoài. Từ cái nôi ban đầu là vùng Bắc Bộ và Bắc Trung Bộ, tập trung ở đồng bằng, người Kinh chuyển cư đến các vùng khác, trở thành tộc người đông đảo, có mặt trên mọi địa bàn, địa hình của Tổ quốc Việt Nam, Người Kinh còn có tên gọi khác là: Người Việt.',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                      SizedBox(height: 15,),
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Theo số liệu thống kê của Tổng điều tra dân số và nhà ở Việt Nam năm 2019, dân tộc Kinh hiện có 82.085.826 người, chiếm 86,83% tổng dân số cả nước. Số dân sống ở thành thị là 31.168.839 người, sống ở nông thôn là 50.916.987 người.',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                      SizedBox(height: 15,),
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Tiếng Việt: nằm trong nhóm ngôn ngữ Việt-Mường, ngữ hệ Nam Á.',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                      SizedBox(height: 15,),
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Người Kinh là tộc người duy nhất trong 54 tộc người của quốc gia Việt Nam cư trú thành các cộng đồng ở tất cả tỉnh thành của Việt Nam.',
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
                    'Xem thêm dân tộc Kinh ở nhandan.vn',
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
                    'Xem thêm dân tộc Kinh ở Wikipedia',
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
