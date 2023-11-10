import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

class Khmer extends StatefulWidget {
  const Khmer({Key? key}) : super(key: key);

  @override
  // ignore: library_private_types_in_public_api
  _KhmerState createState() => _KhmerState();
}

class _KhmerState extends State<Khmer> {
  _launchURL() async {
    final url = Uri.parse('https://nhandan.vn/dan-toc-kho-me-post723889.html');
    if (await canLaunchUrl(url)) {
      await launchUrl(url);
    } else {
      throw 'Could not launch $url';
    }
  }

  _launchURL1() async {
    final url =
        Uri.parse('https://vi.wikipedia.org/wiki/Ng%C6%B0%E1%BB%9Di_Khmer');
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
          "Dân tộc Khmer",
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
                  'assets/images/khmer.jpg',
                ),
                const Padding(
                  padding: EdgeInsets.all(6.0),
                  child: Column(
                    children: [
                      SizedBox(
                        child: Align(
                          alignment: Alignment.centerRight,
                          child: Text(
                            'Nhiều nghiên cứu đã khẳng định người Khơ-me ở nước ta là hậu duệ của các di dân từ Lục Chân Lạp - tiền thân của nhà nước Campuchia ngày nay. Họ di cư sang khu vực này theo nhiều đợt và do nhiều nguyên nhân khác nhau. Cùng với người Việt và người Hoa, người Khơ-me là một trong những nhóm cư dân có mặt sớm nhất ở đồng bằng sông Cửu Long.',
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
                            'Ở Việt Nam, người Khơ-me còn có một số tên gọi khác như Khơ-me Crôm, Khơ-me Hạ, Khơ-me Dưới, người Việt gốc Miên.',
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
                            'Theo số liệu điều tra 53 dân tộc thiểu số 1/4/ 2019, tổng dân số người Khơ-me: 1.319.652 người; dân số nam: 650.238 người; dân số nữ: 669.414 người; quy mô hộ: 3.7 người/hộ; tỷ lệ dân số sống ở khu vực nông thôn: 76.5%.',
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
                            'Người Khơ-me nói tiếng Khơ-me - một ngôn ngữ thuộc nhóm Môn-Khơ-me trong ngữ hệ Nam Á.',
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
                            'Căn cứ vào đặc điểm sinh thái và môi trường tự nhiên, chia sự phân bố của người Khơ-me ở đồng bằng sông Cửu Long theo ba vùng chính. Vùng nội địa: đây là vùng cư trú lâu đời nhất của người Khơ-me ở đồng bằng sông Cửu Long. Vùng ven biển: kéo dài từ Trà Vinh, qua Sóc Trăng tới Bạc Liêu.m Vùng núi biên giới Tây Nam: thuộc các tỉnh An Giang và Kiên Giang, nơi có dãy núi Thất Sơn, núi Sập, núi Vọng Thê.',
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
                    'Xem thêm dân tộc Khmer ở nhandan.vn',
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
                    'Xem thêm dân tộc Khmer ở Wikipedia',
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
