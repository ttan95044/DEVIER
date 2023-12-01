class BardModelCham {
  String? system;
  String? message;

  BardModelCham({this.system, this.message});

  BardModelCham.fromJson(Map<String, dynamic> json) {
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