import 'package:flutter/services.dart';

class BluetoothDeviceSummary {
  const BluetoothDeviceSummary({
    required this.name,
    required this.address,
  });

  final String name;
  final String address;

  factory BluetoothDeviceSummary.fromMap(Map<Object?, Object?> map) {
    final rawName = (map['name'] ?? '').toString().trim();
    final address = (map['address'] ?? '').toString().trim().toUpperCase();
    return BluetoothDeviceSummary(
      name: rawName.isEmpty ? address : rawName,
      address: address,
    );
  }
}

class BluetoothTransport {
  static const MethodChannel _channel = MethodChannel('aura_nav/bluetooth');

  Future<bool> isSupported() async {
    return (await _channel.invokeMethod<bool>('isSupported')) ?? false;
  }

  Future<bool> isEnabled() async {
    return (await _channel.invokeMethod<bool>('isEnabled')) ?? false;
  }

  Future<List<BluetoothDeviceSummary>> getBondedDevices() async {
    final rawDevices =
        await _channel.invokeMethod<List<dynamic>>('getBondedDevices') ??
        const <dynamic>[];

    return rawDevices
        .whereType<Map>()
        .map(
          (device) => BluetoothDeviceSummary.fromMap(
            Map<Object?, Object?>.from(device),
          ),
        )
        .toList();
  }

  Future<void> connect(String address) {
    return _channel.invokeMethod<void>('connect', <String, dynamic>{
      'address': normalizeAddress(address),
    });
  }

  Future<void> disconnect() {
    return _channel.invokeMethod<void>('disconnect');
  }

  Future<void> sendPayload(String payload) {
    return _channel.invokeMethod<void>('sendPayload', <String, dynamic>{
      'payload': payload,
    });
  }

  static String normalizeAddress(String address) {
    return address.trim().replaceAll('-', ':').toUpperCase();
  }
}
