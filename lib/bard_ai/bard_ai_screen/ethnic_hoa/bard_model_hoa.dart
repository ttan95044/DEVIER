class BardModelHoa {
  String? system;
  String? message;

  BardModelHoa({this.system, this.message});

  BardModelHoa.fromJson(Map<String, dynamic> json) {
    if(json["system"] is String) {
      system = json["system"];
    }
    if(json["message"] is String) {
      message = json["message"];
    }
  }

  Map<String, dynamic> toJson() {
    // ignore: no_leading_underscores_for_local_identifiers
    final Map<String, dynamic> _data = <String, dynamic>{};
    _data["system"] = system;
    _data["message"] = message;
    return _data;
  }
}