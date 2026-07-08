import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:amap_flutter_base/amap_flutter_base.dart';
import 'package:amap_flutter_map/amap_flutter_map.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';

import 'bluetooth_transport.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AuraARCompanionApp());
}

class CoordConverter {
  static const double pi = 3.14159265358979324;
  static const double a = 6378137.0;
  static const double ee = 0.00669342162296594323;

  static List<double> wgs84ToGcj02(double lng, double lat) {
    double dlat = _transformLat(lng - 105.0, lat - 35.0);
    double dlng = _transformLng(lng - 105.0, lat - 35.0);
    double radlat = lat / 180.0 * pi;
    double magic = sin(radlat);
    magic = 1 - ee * magic * magic;
    double sqrtmagic = sqrt(magic);
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi);
    dlng = (dlng * 180.0) / (a / sqrtmagic * cos(radlat) * pi);
    return <double>[lng + dlng, lat + dlat];
  }

  static double _transformLat(double x, double y) {
    double ret =
        -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * sqrt(x.abs());
    ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0;
    ret += (20.0 * sin(y * pi) + 40.0 * sin(y / 3.0 * pi)) * 2.0 / 3.0;
    ret += (160.0 * sin(y / 12.0 * pi) + 320 * sin(y * pi / 30.0)) * 2.0 / 3.0;
    return ret;
  }

  static double _transformLng(double x, double y) {
    double ret =
        300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * sqrt(x.abs());
    ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0;
    ret += (20.0 * sin(x * pi) + 40.0 * sin(x / 3.0 * pi)) * 2.0 / 3.0;
    ret += (150.0 * sin(x / 12.0 * pi) + 300.0 * sin(x / 30.0 * pi)) * 2.0 / 3.0;
    return ret;
  }
}

class AuraARCompanionApp extends StatelessWidget {
  const AuraARCompanionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: const Color(0xFFF2F2F7),
        primaryColor: const Color(0xFF007AFF),
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF007AFF)),
        fontFamily: 'Roboto',
      ),
      home: const NavControlCenter(),
    );
  }
}

class _BluetoothDialogResult {
  const _BluetoothDialogResult.connect({
    required this.address,
    required this.name,
  }) : disconnect = false;

  const _BluetoothDialogResult.disconnect()
      : disconnect = true,
        address = '',
        name = '';

  final bool disconnect;
  final String address;
  final String name;
}

class _CachedRouteStep {
  const _CachedRouteStep({
    required this.action,
    required this.startMeters,
    required this.endMeters,
    required this.distanceMeters,
  });

  final String action;
  final double startMeters;
  final double endMeters;
  final double distanceMeters;
}

class _CachedRoute {
  const _CachedRoute({
    required this.points,
    required this.cumulativeMeters,
    required this.steps,
    required this.totalDistanceMeters,
    required this.fullPolyline,
  });

  final List<LatLng> points;
  final List<double> cumulativeMeters;
  final List<_CachedRouteStep> steps;
  final double totalDistanceMeters;
  final String fullPolyline;

  _CachedRouteStep stepForProgress(double progressMeters) {
    if (steps.isEmpty) {
      return const _CachedRouteStep(
        action: '继续前进',
        startMeters: 0,
        endMeters: 0,
        distanceMeters: 0,
      );
    }

    for (final step in steps) {
      if (progressMeters <= step.endMeters) {
        return step;
      }
    }
    return steps.last;
  }
}

class _RouteProjection {
  const _RouteProjection({
    required this.segmentIndex,
    required this.snappedPoint,
    required this.progressMeters,
    required this.distanceToRouteMeters,
  });

  final int segmentIndex;
  final LatLng snappedPoint;
  final double progressMeters;
  final double distanceToRouteMeters;
}

class NavControlCenter extends StatefulWidget {
  const NavControlCenter({super.key});

  @override
  State<NavControlCenter> createState() => _NavControlCenterState();
}

class _NavControlCenterState extends State<NavControlCenter> {
  static const Duration _periodicRefreshInterval = Duration(seconds: 12);

  final String gaodeKey = '587aca5fd88fc2b1aaa6382af28e80e8';
  final String androidMapKey = '9feb19394862e808b4eae14330270ddb';
  final BluetoothTransport _bluetoothTransport = BluetoothTransport();
  final TextEditingController _destController = TextEditingController();

  List<dynamic> _suggestions = <dynamic>[];
  Timer? _debounce;
  Timer? _navLoopTimer;
  Timer? _locationRefreshTimer;
  StreamSubscription<Position>? _positionStreamSubscription;

  AMapController? _mapController;
  LatLng _mapCenter = const LatLng(30.59, 114.30);
  Set<Polyline> _mapPolylines = <Polyline>{};

  String _currentPosStr = '正在获取位置...';
  String _targetPosStr = '请在上方搜索或拖动地图选择';
  String? _targetCoords;
  double? _initialTotalDistance;
  int _acceptedLocationCount = 0;
  int _activeRefreshCount = 0;
  double? _latestAccuracy;
  DateTime? _latestPositionTimestamp;

  String _statusText = '系统已就绪，等待选择目的地';
  String _subStatusText = '当前蓝牙: 未连接';

  String _targetBluetoothAddress = '';
  String _targetBluetoothName = '';
  bool _bluetoothConnected = false;
  bool _bluetoothSupported = true;
  bool _navRequestInFlight = false;
  bool isComputing = false;
  Map<String, double>? _latestLocation;
  Position? _latestRawPosition;
  Future<Map<String, double>?>? _refreshLocationFuture;
  DateTime? _latestLocationAt;
  DateTime? _lastNavDispatchAt;
  _CachedRoute? _cachedRoute;
  _RouteProjection? _latestProjection;
  int _offRouteHits = 0;
  int _routeRevision = 0;
  int _lastHelmetRouteRevision = -1;
  String? _lastBluetoothPayload;

  @override
  void initState() {
    super.initState();
    unawaited(_checkPermissions());
  }

  String get _currentBluetoothLabel {
    if (_targetBluetoothAddress.isEmpty) {
      return '未选择设备';
    }
    if (_targetBluetoothName.isEmpty || _targetBluetoothName == _targetBluetoothAddress) {
      return _targetBluetoothAddress;
    }
    return '$_targetBluetoothName ($_targetBluetoothAddress)';
  }

  Future<void> _checkPermissions() async {
    final permissions = <Permission>[Permission.location];
    if (Platform.isAndroid) {
      permissions.add(Permission.bluetoothScan);
      permissions.add(Permission.bluetoothConnect);
    }

    await permissions.request();
    await _prepareBluetooth();
    await _startLocationTracking();
    _ensurePeriodicLocationRefresh();
    unawaited(_refreshCurrentLocation(trigger: 'startup'));
  }

  LocationSettings _buildLocationSettings() {
    if (Platform.isAndroid) {
      return AndroidSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        distanceFilter: 2,
        forceLocationManager: true,
        intervalDuration: const Duration(seconds: 1),
      );
    }

    return const LocationSettings(
      accuracy: LocationAccuracy.bestForNavigation,
      distanceFilter: 2,
    );
  }

  Future<void> _startLocationTracking() async {
    if (_positionStreamSubscription != null) {
      debugPrint('[GPS] position stream already active');
      return;
    }

    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      debugPrint('[GPS] location service disabled');
      if (mounted) {
        setState(() {
          _currentPosStr = '定位服务未开启';
        });
      }
      return;
    }

    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }

    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      debugPrint('[GPS] location permission unavailable: $permission');
      if (mounted) {
        setState(() {
          _currentPosStr = '定位权限未授予';
        });
      }
      return;
    }

    debugPrint(
      '[GPS] start position stream settings='
      'accuracy=bestForNavigation distanceFilter=2 interval=1s forceLocationManager=true',
    );

    _positionStreamSubscription = Geolocator.getPositionStream(
      locationSettings: _buildLocationSettings(),
    ).listen(
      _handlePositionUpdate,
      onError: (Object error) {
        debugPrint('[GPS] stream error: $error');
        if (!mounted) {
          return;
        }
        setState(() {
          _currentPosStr = '实时定位异常';
        });
        if (isComputing) {
          _updateStatus('实时定位异常', error.toString());
        }
      },
    );
  }

  String _formatDateTime(DateTime? value) {
    if (value == null) {
      return 'null';
    }
    return value.toIso8601String();
  }

  String _formatPositionTimestamp(DateTime? value) {
    if (value == null) {
      return 'null';
    }
    return value.toIso8601String();
  }

  String _formatDebugClock(DateTime? value) {
    if (value == null) {
      return '--';
    }
    final local = value.toLocal();
    final hh = local.hour.toString().padLeft(2, '0');
    final mm = local.minute.toString().padLeft(2, '0');
    final ss = local.second.toString().padLeft(2, '0');
    return '$hh:$mm:$ss';
  }

  String get _locationDebugSummary {
    final accuracyText = _latestAccuracy == null
        ? '--'
        : '${_latestAccuracy!.toStringAsFixed(1)}m';
    final timeText = _formatDebugClock(_latestPositionTimestamp);
    return '新点: $_acceptedLocationCount | 强刷: $_activeRefreshCount | 精度: $accuracyText | 时间: $timeText';
  }

  Duration? get _latestLocationAge {
    if (_latestLocationAt == null) {
      return null;
    }
    return DateTime.now().difference(_latestLocationAt!);
  }

  bool _hasRecentStreamLocation(Duration maxAge) {
    final age = _latestLocationAge;
    return _latestLocation != null && age != null && age <= maxAge;
  }

  bool _isDuplicatePosition(Position pos) {
    final previous = _latestRawPosition;
    if (previous == null) {
      return false;
    }

    final sameTimestamp =
        previous.timestamp != null &&
        pos.timestamp != null &&
        previous.timestamp!.isAtSameMomentAs(pos.timestamp!);
    final sameCoordinates =
        (previous.latitude - pos.latitude).abs() < 0.0000001 &&
        (previous.longitude - pos.longitude).abs() < 0.0000001;

    return sameTimestamp && sameCoordinates;
  }

  void _ensurePeriodicLocationRefresh() {
    if (_locationRefreshTimer != null) {
      return;
    }

    debugPrint(
      '[GPS][timer] start periodic refresh interval='
      '${_periodicRefreshInterval.inSeconds}s',
    );
    _locationRefreshTimer = Timer.periodic(_periodicRefreshInterval, (_) {
      unawaited(_performPeriodicLocationRefresh());
    });
  }

  Future<void> _performPeriodicLocationRefresh() async {
    debugPrint(
      '[GPS][timer] periodic refresh tick '
      'isComputing=$isComputing '
      'latestAgeMs=${_latestLocationAge?.inMilliseconds}',
    );
    final location = await _refreshCurrentLocation(
      showError: false,
      trigger: 'timer',
    );
    if (location != null && isComputing && !_navRequestInFlight) {
      debugPrint('[GPS][timer] periodic refresh produced location for nav update');
      unawaited(_executeNavLoop(originOverride: location));
    }
  }

  void _logPosition(String source, Position pos, Map<String, double> location) {
    final now = DateTime.now();
    final ageMs = pos.timestamp == null
        ? 'unknown'
        : now.difference(pos.timestamp!).inMilliseconds.toString();
    final cacheGapMs = _latestLocationAt == null
        ? 'first'
        : now.difference(_latestLocationAt!).inMilliseconds.toString();

    debugPrint(
      '[GPS][$source] raw='
      'lat=${pos.latitude.toStringAsFixed(6)},lng=${pos.longitude.toStringAsFixed(6)} '
      'acc=${pos.accuracy.toStringAsFixed(1)}m '
      'speed=${pos.speed.toStringAsFixed(2)}m/s '
      'heading=${pos.heading.toStringAsFixed(1)} '
      'time=${_formatPositionTimestamp(pos.timestamp)} '
      'ageMs=$ageMs '
      'cacheGapMs=$cacheGapMs '
      'mapped=lng=${location['lng']!.toStringAsFixed(6)},lat=${location['lat']!.toStringAsFixed(6)}',
    );
  }

  Map<String, double> _applyPosition(Position pos) {
    final gcj02 = CoordConverter.wgs84ToGcj02(pos.longitude, pos.latitude);
    final location = <String, double>{
      'lng': gcj02[0],
      'lat': gcj02[1],
    };

    if (_isDuplicatePosition(pos)) {
      _logPosition('duplicate', pos, location);
      return _latestLocation ?? location;
    }

    _logPosition('accept', pos, location);
    _latestLocation = location;
    _latestRawPosition = pos;
    _latestLocationAt = DateTime.now();
    _acceptedLocationCount += 1;
    _latestAccuracy = pos.accuracy;
    _latestPositionTimestamp = pos.timestamp?.toLocal() ?? DateTime.now();

    if (mounted) {
      setState(() {
        _currentPosStr =
            '已定位 ${gcj02[0].toStringAsFixed(4)}, ${gcj02[1].toStringAsFixed(4)}';
      });
    }

    if (isComputing &&
        !_navRequestInFlight &&
        (_lastNavDispatchAt == null ||
            DateTime.now().difference(_lastNavDispatchAt!) >=
                const Duration(seconds: 2))) {
      debugPrint(
        '[NAV][trigger] location stream triggered nav refresh '
        'lastDispatch=${_formatDateTime(_lastNavDispatchAt)}',
      );
      unawaited(_executeNavLoop(originOverride: location));
    }

    return location;
  }

  void _handlePositionUpdate(Position pos) {
    debugPrint('[GPS][stream] received position update');
    _applyPosition(pos);
  }

  Future<Map<String, double>?> _refreshCurrentLocation({
    bool showError = true,
    String trigger = 'manual',
  }) {
    final inFlight = _refreshLocationFuture;
    if (inFlight != null) {
      debugPrint('[GPS][refresh] join in-flight request trigger=$trigger');
      return inFlight;
    }

    final future = _runRefreshCurrentLocation(
      showError: showError,
      trigger: trigger,
    );
    _refreshLocationFuture = future;
    future.whenComplete(() {
      if (identical(_refreshLocationFuture, future)) {
        _refreshLocationFuture = null;
      }
    });
    return future;
  }

  Future<Map<String, double>?> _runRefreshCurrentLocation({
    bool showError = true,
    String trigger = 'manual',
  }) async {
    try {
      final serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        debugPrint('[GPS][refresh] location service disabled trigger=$trigger');
        if (mounted) {
          setState(() {
            _currentPosStr = '定位服务未开启';
          });
        }
        return null;
      }

      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        debugPrint(
          '[GPS][refresh] location permission unavailable: $permission trigger=$trigger',
        );
        if (mounted) {
          setState(() {
            _currentPosStr = '定位权限未授予';
          });
        }
        return null;
      }

      debugPrint('[GPS][refresh] requesting current position trigger=$trigger');
      if (mounted) {
        setState(() {
          _activeRefreshCount += 1;
        });
      }
      final pos = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.bestForNavigation,
        timeLimit: const Duration(seconds: 15),
      );
      debugPrint('[GPS][refresh] current position resolved trigger=$trigger');
      return _applyPosition(pos);
    } catch (e) {
      debugPrint('[GPS][refresh] failed trigger=$trigger: $e');
      final fallbackAge = _latestLocationAge;
      if (_latestLocation != null && fallbackAge != null) {
        debugPrint(
          '[GPS][refresh] keep latest stream location '
          'trigger=$trigger '
          'ageMs=${fallbackAge.inMilliseconds} '
          'rawTime=${_formatPositionTimestamp(_latestRawPosition?.timestamp)}',
        );
        return _latestLocation;
      }
      if (mounted && showError) {
        setState(() {
          _currentPosStr = '定位失败，请检查手机 GPS';
        });
      }
      if (showError && isComputing) {
        _updateStatus('定位失败', e.toString());
      }
      return null;
    } finally {
      if (mounted) {
        setState(() {
          _activeRefreshCount =
              ((_activeRefreshCount - 1).clamp(0, 1 << 20) as num).toInt();
        });
      }
    }
  }

  Future<Map<String, double>?> _currentLocation({
    bool requireFresh = false,
  }) async {
    final age = _latestLocationAge;
    final recentThreshold = requireFresh
        ? const Duration(seconds: 3)
        : const Duration(seconds: 6);
    final hasRecentLocation = _hasRecentStreamLocation(recentThreshold);

    if (hasRecentLocation) {
      debugPrint(
        '[GPS][current] use recent stream location '
        'requireFresh=$requireFresh '
        'ageMs=${age!.inMilliseconds} '
        'rawTime=${_formatPositionTimestamp(_latestRawPosition?.timestamp)} '
        'storedAt=${_formatDateTime(_latestLocationAt)}',
      );
      return _latestLocation;
    }

    debugPrint(
      '[GPS][current] refresh needed requireFresh=$requireFresh '
      'hasRecentLocation=$hasRecentLocation '
      'latestStoredAt=${_formatDateTime(_latestLocationAt)} '
      'rawTime=${_formatPositionTimestamp(_latestRawPosition?.timestamp)}',
    );

    final refreshed = await _refreshCurrentLocation(
      showError: requireFresh || _latestLocation == null,
      trigger: requireFresh ? 'current:fresh' : 'current',
    );
    if (refreshed == null && _latestLocation != null) {
      debugPrint(
        '[GPS][current] refresh failed, fallback to cached location '
        'storedAt=${_formatDateTime(_latestLocationAt)} '
        'rawTime=${_formatPositionTimestamp(_latestRawPosition?.timestamp)}',
      );
    }
    return refreshed ?? _latestLocation;
  }

  Future<void> _prepareBluetooth() async {
    if (!Platform.isAndroid) {
      _bluetoothSupported = false;
      _updateStatus('当前平台不支持', '蓝牙导航仅实现了 Android 端连接');
      return;
    }

    try {
      final supported = await _bluetoothTransport.isSupported();
      final enabled = await _bluetoothTransport.isEnabled();
      if (!mounted) return;

      setState(() {
        _bluetoothSupported = supported;
      });

      if (!supported) {
        _updateStatus('蓝牙不可用', '当前手机不支持蓝牙连接');
        return;
      }

      if (!enabled) {
        _updateStatus('请先开启手机蓝牙', '然后在右上角选择 ELF2 设备');
      }
    } catch (e) {
      _updateStatus('蓝牙初始化失败', e.toString());
    }
  }

  Future<void> _showBluetoothSettingsDialog() async {
    final addressController = TextEditingController(text: _targetBluetoothAddress);
    var selectedName = _targetBluetoothName;
    List<BluetoothDeviceSummary> bondedDevices = const <BluetoothDeviceSummary>[];

    try {
      bondedDevices = await _bluetoothTransport.getBondedDevices();
    } catch (_) {}

    if (!mounted) return;

    final result = await showDialog<_BluetoothDialogResult>(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              title: const Text(
                '设置蓝牙设备',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              content: SizedBox(
                width: 360,
                child: SingleChildScrollView(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        '请先让手机与 ELF2 开发板完成系统配对，然后在这里选择或输入开发板蓝牙地址。',
                        style: TextStyle(fontSize: 13, color: Colors.black54),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: addressController,
                        textCapitalization: TextCapitalization.characters,
                        decoration: InputDecoration(
                          hintText: '例如: 00:11:22:33:44:55',
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(12),
                            borderSide: BorderSide.none,
                          ),
                          filled: true,
                          fillColor: Colors.grey.shade200,
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 14,
                          ),
                        ),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        '已配对设备',
                        style: TextStyle(
                          color: Colors.grey.shade700,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 8),
                      if (bondedDevices.isEmpty)
                        const Text(
                          '没有读取到已配对设备，可以直接手动输入 ELF2 的蓝牙 MAC 地址。',
                          style: TextStyle(fontSize: 13, color: Colors.black54),
                        )
                      else
                        ConstrainedBox(
                          constraints: const BoxConstraints(maxHeight: 220),
                          child: ListView.builder(
                            shrinkWrap: true,
                            itemCount: bondedDevices.length,
                            itemBuilder: (context, index) {
                              final device = bondedDevices[index];
                              final currentAddress =
                                  BluetoothTransport.normalizeAddress(
                                    addressController.text,
                                  );
                              return ListTile(
                                dense: true,
                                contentPadding: EdgeInsets.zero,
                                title: Text(device.name),
                                subtitle: Text(device.address),
                                trailing: currentAddress == device.address
                                    ? const Icon(
                                        Icons.check,
                                        color: Color(0xFF007AFF),
                                      )
                                    : null,
                                onTap: () {
                                  setDialogState(() {
                                    addressController.text = device.address;
                                    selectedName = device.name;
                                  });
                                },
                              );
                            },
                          ),
                        ),
                    ],
                  ),
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text(
                    '取消',
                    style: TextStyle(color: Colors.black54),
                  ),
                ),
                TextButton(
                  onPressed: _bluetoothConnected
                      ? () => Navigator.pop(
                            context,
                            const _BluetoothDialogResult.disconnect(),
                          )
                      : null,
                  child: const Text('断开'),
                ),
                TextButton(
                  onPressed: () {
                    final address =
                        BluetoothTransport.normalizeAddress(addressController.text);
                    if (address.isEmpty) {
                      return;
                    }

                    Navigator.pop(
                      context,
                      _BluetoothDialogResult.connect(
                        address: address,
                        name: selectedName,
                      ),
                    );
                  },
                  child: const Text(
                    '连接',
                    style: TextStyle(
                      color: Color(0xFF007AFF),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            );
          },
        );
      },
    );

    if (result == null) {
      return;
    }
    if (result.disconnect) {
      await _disconnectBluetooth();
      return;
    }

    setState(() {
      _targetBluetoothAddress = result.address;
      _targetBluetoothName =
          result.name.isEmpty ? result.address : result.name;
      _bluetoothConnected = false;
    });

    await _connectBluetooth();
  }

  Future<bool> _connectBluetooth({String? address}) async {
    final normalizedAddress = BluetoothTransport.normalizeAddress(
      address ?? _targetBluetoothAddress,
    );
    if (normalizedAddress.isEmpty) {
      _updateStatus('未配置蓝牙设备', '请先在右上角选择或输入 ELF2 的蓝牙地址');
      return false;
    }

    try {
      final enabled = await _bluetoothTransport.isEnabled();
      if (!enabled) {
        _updateStatus('蓝牙未开启', '请先在手机系统里开启蓝牙');
        return false;
      }

      await _bluetoothTransport.connect(normalizedAddress);
      if (!mounted) return true;

      setState(() {
        _targetBluetoothAddress = normalizedAddress;
        if (_targetBluetoothName.isEmpty) {
          _targetBluetoothName = normalizedAddress;
        }
        _bluetoothConnected = true;
      });
      _updateStatus('蓝牙已连接', '当前设备: $_currentBluetoothLabel');
      return true;
    } on PlatformException catch (e) {
      if (mounted) {
        setState(() {
          _bluetoothConnected = false;
        });
      }
      _updateStatus('蓝牙连接失败', e.message ?? e.code);
      return false;
    } catch (e) {
      if (mounted) {
        setState(() {
          _bluetoothConnected = false;
        });
      }
      _updateStatus('蓝牙连接失败', e.toString());
      return false;
    }
  }

  Future<void> _disconnectBluetooth({bool updateStatus = true}) async {
    try {
      await _bluetoothTransport.disconnect();
    } catch (_) {}

    if (!mounted) return;
    setState(() {
      _bluetoothConnected = false;
    });

    if (updateStatus) {
      _updateStatus('蓝牙已断开', '当前设备: $_currentBluetoothLabel');
    }
  }

  Future<bool> _ensureBluetoothConnected() async {
    if (!_bluetoothSupported) {
      _updateStatus('蓝牙不可用', '当前手机不支持蓝牙连接');
      return false;
    }
    if (_bluetoothConnected) {
      return true;
    }
    return _connectBluetooth();
  }

  void _onSearchChanged(String query) {
    if (_debounce?.isActive ?? false) {
      _debounce!.cancel();
    }

    _debounce = Timer(const Duration(milliseconds: 500), () async {
      if (query.isEmpty) {
        if (!mounted) return;
        setState(() {
          _suggestions = <dynamic>[];
        });
        return;
      }

      final url =
          'https://restapi.amap.com/v3/assistant/inputtips?keywords=$query&key=$gaodeKey&datatype=all';
      try {
        final response = await http.get(Uri.parse(url));
        final data = json.decode(response.body);
        if (!mounted) return;
        if (data['status'] == '1') {
          setState(() {
            _suggestions = data['tips'] as List<dynamic>;
          });
        }
      } catch (_) {}
    });
  }

  Future<void> _reverseGeocode(LatLng point) async {
    if (isComputing) {
      return;
    }

    final url =
        'https://restapi.amap.com/v3/geocode/regeo?location=${point.longitude},${point.latitude}&key=$gaodeKey';
    try {
      final response = await http.get(Uri.parse(url));
      final data = json.decode(response.body);
      if (!mounted) return;

      if (data['status'] == '1') {
        final address = data['regeocode']['formatted_address'].toString();
        setState(() {
          _destController.text = address;
          _targetCoords = '${point.longitude},${point.latitude}';
          _targetPosStr = address;
        });
      }
    } catch (_) {}
  }

  Future<void> _sendToELF2(Map<String, dynamic> data) async {
    if (!await _ensureBluetoothConnected()) {
      return;
    }

    try {
      final payload = json.encode(data);
      if (_lastBluetoothPayload == payload) {
        debugPrint('[BT] skip duplicate payload');
        return;
      }

      await _bluetoothTransport.sendPayload(payload);
      _lastBluetoothPayload = payload;
    } on PlatformException catch (e) {
      if (mounted) {
        setState(() {
          _bluetoothConnected = false;
        });
      }
      _updateStatus('蓝牙发送失败', e.message ?? e.code);
    } catch (e) {
      if (mounted) {
        setState(() {
          _bluetoothConnected = false;
        });
      }
      _updateStatus('蓝牙发送失败', e.toString());
    }
  }

  Future<void> _startTacticalNav() async {
    FocusScope.of(context).unfocus();
    debugPrint(
      '[NAV][start] begin target=$_targetCoords name=${_destController.text}',
    );
    if (_targetCoords == null) {
      _updateStatus('无法开始导航', '请先选择一个有效的目的地');
      return;
    }
    if (!await _ensureBluetoothConnected()) {
      return;
    }

    await _startLocationTracking();
    final currentLocation = await _currentLocation(requireFresh: true);
    if (currentLocation == null) {
      debugPrint('[NAV][start] abort: no fresh location');
      _updateStatus('无法开始导航', '未获取到实时定位，请检查手机定位服务');
      return;
    }

    debugPrint(
      '[NAV][start] initial location '
      'lng=${currentLocation['lng']},lat=${currentLocation['lat']} '
      'storedAt=${_formatDateTime(_latestLocationAt)} '
      'rawTime=${_formatPositionTimestamp(_latestRawPosition?.timestamp)}',
    );

    _navLoopTimer?.cancel();
    if (mounted) {
      setState(() {
        isComputing = true;
        _initialTotalDistance = null;
        _lastNavDispatchAt = null;
      });
    }
    _cachedRoute = null;
    _latestProjection = null;
    _offRouteHits = 0;
    _lastHelmetRouteRevision = -1;
    _lastBluetoothPayload = null;
    _updateStatus('路线计算中...', '正在通过蓝牙同步到 $_currentBluetoothLabel');

    await _executeNavLoop(
      originOverride: currentLocation,
      forceReroute: true,
    );

    _navLoopTimer = Timer.periodic(const Duration(seconds: 3), (_) async {
      await _executeNavLoop();
    });
  }

  void _fitMapToBounds(List<LatLng> points) {
    if (points.isEmpty || _mapController == null) {
      return;
    }

    double minLat = points[0].latitude;
    double maxLat = points[0].latitude;
    double minLng = points[0].longitude;
    double maxLng = points[0].longitude;

    for (final point in points) {
      if (point.latitude < minLat) minLat = point.latitude;
      if (point.latitude > maxLat) maxLat = point.latitude;
      if (point.longitude < minLng) minLng = point.longitude;
      if (point.longitude > maxLng) maxLng = point.longitude;
    }

    if (minLat == maxLat) {
      minLat -= 0.001;
      maxLat += 0.001;
    }
    if (minLng == maxLng) {
      minLng -= 0.001;
      maxLng += 0.001;
    }

    final bounds = LatLngBounds(
      southwest: LatLng(minLat, minLng),
      northeast: LatLng(maxLat, maxLng),
    );
    _mapController?.moveCamera(CameraUpdate.newLatLngBounds(bounds, 80.0));
  }

  double _distanceMeters(LatLng a, LatLng b) {
    return Geolocator.distanceBetween(
      a.latitude,
      a.longitude,
      b.latitude,
      b.longitude,
    );
  }

  List<LatLng> _decodePolyline(String polyline) {
    final points = <LatLng>[];
    for (final point in polyline.split(';')) {
      if (point.isEmpty || !point.contains(',')) {
        continue;
      }
      final coords = point.split(',');
      final lng = double.tryParse(coords[0]) ?? 0;
      final lat = double.tryParse(coords[1]) ?? 0;
      if (lng != 0 && lat != 0) {
        points.add(LatLng(lat, lng));
      }
    }
    return points;
  }

  _CachedRoute? _buildCachedRoute(
    Map<String, dynamic> path,
    Map<String, double> origin,
  ) {
    final steps = path['steps'];
    if (steps is! List || steps.isEmpty) {
      return null;
    }

    final routePoints = <LatLng>[LatLng(origin['lat']!, origin['lng']!)];
    final routeSteps = <_CachedRouteStep>[];
    final fullPolylineParts = <String>[];
    double progressMeters = 0;

    for (final rawStep in steps) {
      final step = rawStep as Map<String, dynamic>;
      final polyline = step['polyline']?.toString() ?? '';
      if (polyline.isEmpty) {
        continue;
      }

      final decoded = _decodePolyline(polyline);
      if (decoded.isEmpty) {
        continue;
      }

      if (routePoints.isNotEmpty && decoded.isNotEmpty) {
        final first = decoded.first;
        final last = routePoints.last;
        final samePoint =
            (first.latitude - last.latitude).abs() < 0.0000001 &&
            (first.longitude - last.longitude).abs() < 0.0000001;
        if (samePoint) {
          decoded.removeAt(0);
        }
      }

      routePoints.addAll(decoded);
      fullPolylineParts.add(polyline);

      final stepDistance =
          double.tryParse(step['distance']?.toString() ?? '') ?? 0;
      routeSteps.add(
        _CachedRouteStep(
          action: step['action']?.toString().trim().isNotEmpty == true
              ? step['action'].toString()
              : '继续前进',
          startMeters: progressMeters,
          endMeters: progressMeters + stepDistance,
          distanceMeters: stepDistance,
        ),
      );
      progressMeters += stepDistance;
    }

    if (routePoints.length < 2) {
      return null;
    }

    final cumulative = <double>[0];
    for (var i = 1; i < routePoints.length; i += 1) {
      cumulative.add(
        cumulative.last + _distanceMeters(routePoints[i - 1], routePoints[i]),
      );
    }

    final fallbackTotal = cumulative.isNotEmpty ? cumulative.last : 0.0;
    final double totalDistance =
        double.tryParse(path['distance']?.toString() ?? '') ?? fallbackTotal;

    return _CachedRoute(
      points: routePoints,
      cumulativeMeters: cumulative,
      steps: routeSteps,
      totalDistanceMeters: totalDistance,
      fullPolyline: fullPolylineParts.join(';'),
    );
  }

  _RouteProjection? _projectOntoRoute(
    _CachedRoute route,
    Map<String, double> location,
  ) {
    if (route.points.length < 2) {
      return null;
    }

    final x0 = location['lng']!;
    final y0 = location['lat']!;
    double? bestDistance;
    _RouteProjection? bestProjection;

    for (var i = 0; i < route.points.length - 1; i += 1) {
      final start = route.points[i];
      final end = route.points[i + 1];
      final x1 = start.longitude;
      final y1 = start.latitude;
      final x2 = end.longitude;
      final y2 = end.latitude;
      final dx = x2 - x1;
      final dy = y2 - y1;
      final len2 = dx * dx + dy * dy;
      if (len2 == 0) {
        continue;
      }

      final rawT = ((x0 - x1) * dx + (y0 - y1) * dy) / len2;
      final t = rawT.clamp(0.0, 1.0);
      final snappedLng = x1 + dx * t;
      final snappedLat = y1 + dy * t;
      final snapped = LatLng(snappedLat, snappedLng);
      final distance = Geolocator.distanceBetween(
        location['lat']!,
        location['lng']!,
        snappedLat,
        snappedLng,
      );

      if (bestDistance == null || distance < bestDistance) {
        final startMeters = route.cumulativeMeters[i];
        final endMeters = route.cumulativeMeters[i + 1];
        bestDistance = distance;
        bestProjection = _RouteProjection(
          segmentIndex: i,
          snappedPoint: snapped,
          progressMeters: startMeters + (endMeters - startMeters) * t,
          distanceToRouteMeters: distance,
        );
      }
    }

    return bestProjection;
  }

  bool _shouldReroute(_RouteProjection projection) {
    if (projection.distanceToRouteMeters > 45) {
      _offRouteHits += 1;
    } else {
      _offRouteHits = 0;
    }
    return _offRouteHits >= 2;
  }

  void _applyRouteToMap(_CachedRoute route, {bool fitBounds = false}) {
    if (mounted) {
      setState(() {
        _mapPolylines = <Polyline>{
          Polyline(
            points: route.points,
            color: const Color(0xFF007AFF),
            width: 8,
          ),
        };
      });
    }

    if (fitBounds) {
      _fitMapToBounds(route.points);
    }
  }

  Future<void> _requestAndCacheRoute(
    Map<String, double> origin,
    {required String reason}
  ) async {
    if (_targetCoords == null) {
      return;
    }

    final originStr = '${origin['lng']},${origin['lat']}';
    debugPrint(
      '[NAV][reroute] request reason=$reason origin=$originStr dest=$_targetCoords',
    );
    final url =
        'https://restapi.amap.com/v4/direction/bicycling?origin=$originStr&destination=$_targetCoords&key=$gaodeKey&strategy=0';

    final response = await http.get(Uri.parse(url)).timeout(
          const Duration(seconds: 15),
        );
    if (response.statusCode != 200) {
      debugPrint('[NAV][reroute] route request failed status=${response.statusCode}');
      _updateStatus('导航更新失败', '高德接口响应异常: ${response.statusCode}');
      _cachedRoute = null;
      return;
    }

    final data = json.decode(response.body);
    if (data['errcode'].toString() != '0') {
      debugPrint(
        '[NAV][reroute] response error errcode=${data['errcode']} errmsg=${data['errmsg']}',
      );
      _updateStatus(
        '路线规划失败',
        data['errmsg']?.toString() ?? '高德未返回可用骑行路径',
      );
      _cachedRoute = null;
      return;
    }

    final paths = data['data']?['paths'];
    if (paths is! List || paths.isEmpty) {
      _updateStatus('路线规划失败', '高德未返回可用路径');
      _cachedRoute = null;
      return;
    }

    final path = Map<String, dynamic>.from(paths[0] as Map);
    final route = _buildCachedRoute(path, origin);
    if (route == null) {
      _updateStatus('路线规划失败', '当前路径没有可执行的导航步骤');
      _cachedRoute = null;
      return;
    }

    _cachedRoute = route;
    _latestProjection = null;
    _offRouteHits = 0;
    _routeRevision += 1;
    _initialTotalDistance ??= route.totalDistanceMeters;
    _applyRouteToMap(route, fitBounds: reason == 'init' || reason == 'forced');
    debugPrint(
      '[NAV][reroute] cached route reason=$reason '
      'points=${route.points.length} total=${route.totalDistanceMeters.toStringAsFixed(0)}',
    );
  }

  Future<void> _executeNavLoop({
    Map<String, double>? originOverride,
    bool forceReroute = false,
  }) async {
    if (!isComputing || _navRequestInFlight) {
      debugPrint(
        '[NAV][loop] skip isComputing=$isComputing inFlight=$_navRequestInFlight',
      );
      return;
    }

    _navRequestInFlight = true;
    _lastNavDispatchAt = DateTime.now();
    try {
      debugPrint(
        '[NAV][loop] start originOverride=${originOverride != null} '
        'latestStoredAt=${_formatDateTime(_latestLocationAt)} '
        'rawTime=${_formatPositionTimestamp(_latestRawPosition?.timestamp)}',
      );
      final myLoc = originOverride ?? await _currentLocation();
      if (myLoc == null || _targetCoords == null) {
        debugPrint('[NAV][loop] waiting for latest location target=$_targetCoords');
        _updateStatus('等待定位更新', '正在获取最新位置...');
        return;
      }

      if (_cachedRoute == null || forceReroute) {
        final rerouteReason = _cachedRoute == null ? 'init' : 'forced';
        await _requestAndCacheRoute(myLoc, reason: rerouteReason);
      }

      final route = _cachedRoute;
      if (route == null) {
        return;
      }

      final projection = _projectOntoRoute(route, myLoc);
      if (projection == null) {
        debugPrint('[NAV][loop] projection failed, fallback reroute');
        await _requestAndCacheRoute(myLoc, reason: 'projection_failed');
        return;
      }

      _latestProjection = projection;
      debugPrint(
        '[NAV][loop] projection progress=${projection.progressMeters.toStringAsFixed(1)} '
        'distToRoute=${projection.distanceToRouteMeters.toStringAsFixed(1)}',
      );

      if (_shouldReroute(projection)) {
        debugPrint(
          '[NAV][loop] off route detected hits=$_offRouteHits '
          'dist=${projection.distanceToRouteMeters.toStringAsFixed(1)}',
        );
        await _requestAndCacheRoute(myLoc, reason: 'off_route');
        return;
      }

      final step = route.stepForProgress(projection.progressMeters);
      final remainMeters = (route.totalDistanceMeters - projection.progressMeters)
          .clamp(0, route.totalDistanceMeters);
      final nextStepRemain = (step.endMeters - projection.progressMeters)
          .clamp(0, step.distanceMeters);
      final shouldSendFullPath = _lastHelmetRouteRevision != _routeRevision;

      _initialTotalDistance ??= route.totalDistanceMeters;
      _updateStatus(
        '导航中: ${step.action}',
        '距终点 ${remainMeters.toStringAsFixed(0)} 米 | 蓝牙: $_currentBluetoothLabel',
      );

      await _sendToELF2(<String, dynamic>{
        'status': 'navigating',
        'cur_lng': myLoc['lng'],
        'cur_lat': myLoc['lat'],
        'dest_lng': double.parse(_targetCoords!.split(',')[0]),
        'dest_lat': double.parse(_targetCoords!.split(',')[1]),
        'dest_name': _destController.text,
        'next_action': step.action,
        'next_dist': nextStepRemain.toStringAsFixed(0),
        'remain_dist': remainMeters.toStringAsFixed(0),
        'total_dist': _initialTotalDistance?.toStringAsFixed(0) ?? '0',
        'full_path': shouldSendFullPath ? route.fullPolyline : '',
      });
      if (shouldSendFullPath) {
        _lastHelmetRouteRevision = _routeRevision;
      }
      debugPrint(
        '[NAV][helmet] sent cur=${myLoc['lng']},${myLoc['lat']} '
        'next=${step.action} nextDist=${nextStepRemain.toStringAsFixed(0)} '
        'remain=${remainMeters.toStringAsFixed(0)} sendPath=$shouldSendFullPath',
      );
    } catch (e) {
      debugPrint('[NAV][loop] exception: $e');
      _updateStatus('导航更新失败', e.toString());
    } finally {
      _navRequestInFlight = false;
      debugPrint('[NAV][loop] finish');
    }
  }

  void _stopNav() {
    _navLoopTimer?.cancel();
    _cachedRoute = null;
    _latestProjection = null;
    _offRouteHits = 0;
    _lastHelmetRouteRevision = -1;
    _lastBluetoothPayload = null;
    if (mounted) {
      setState(() {
        isComputing = false;
        _initialTotalDistance = null;
        _lastNavDispatchAt = null;
        _mapPolylines.clear();
      });
    }
    _updateStatus('导航已停止', '你可以重新选择目的地');
    unawaited(_sendToELF2(<String, dynamic>{'status': 'standby'}));
  }

  void _updateStatus(String main, String sub) {
    if (!mounted) {
      return;
    }
    setState(() {
      _statusText = main;
      _subStatusText = sub;
    });
  }

  @override
  void dispose() {
    _debounce?.cancel();
    _navLoopTimer?.cancel();
    _locationRefreshTimer?.cancel();
    unawaited(_positionStreamSubscription?.cancel());
    _destController.dispose();
    unawaited(_bluetoothTransport.disconnect());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.transparent,
        title: const Text(
          'AR 视觉伴侣',
          style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: Icon(
              _bluetoothConnected ? Icons.bluetooth_connected : Icons.bluetooth,
              color: const Color(0xFF007AFF),
            ),
            tooltip: '设置蓝牙设备',
            onPressed: _showBluetoothSettingsDialog,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Column(
              children: [
                Container(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.05),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: TextField(
                    controller: _destController,
                    onChanged: _onSearchChanged,
                    style: const TextStyle(color: Colors.black87),
                    decoration: const InputDecoration(
                      hintText: '搜索目的地或在地图上选择...',
                      hintStyle: TextStyle(color: Colors.black38),
                      prefixIcon: Icon(Icons.search, color: Color(0xFF007AFF)),
                      border: InputBorder.none,
                      contentPadding: EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 15,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(18),
                    ),
                    child: Text(
                      '当前位置: $_currentPosStr',
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.black54,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 8),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xFFEFF6FF),
                      borderRadius: BorderRadius.circular(18),
                    ),
                    child: Text(
                      _locationDebugSummary,
                      style: const TextStyle(
                        fontSize: 12,
                        color: Color(0xFF1D4ED8),
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                Expanded(
                  flex: 5,
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(24),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.08),
                          blurRadius: 15,
                          offset: const Offset(0, 8),
                        ),
                      ],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(24),
                      child: Stack(
                        children: [
                          AMapWidget(
                            privacyStatement: const AMapPrivacyStatement(
                              hasContains: true,
                              hasShow: true,
                              hasAgree: true,
                            ),
                            apiKey: AMapApiKey(androidKey: androidMapKey),
                            initialCameraPosition: CameraPosition(
                              target: _mapCenter,
                              zoom: 15,
                            ),
                            onMapCreated: (controller) {
                              _mapController = controller;
                            },
                            onCameraMove: (pos) {
                              _mapCenter = pos.target;
                            },
                            onCameraMoveEnd: (pos) {
                              _reverseGeocode(pos.target);
                            },
                            polylines: _mapPolylines,
                          ),
                          if (!isComputing)
                            const Center(
                              child: Icon(
                                Icons.location_on,
                                color: Color(0xFFFF3B30),
                                size: 48,
                              ),
                            ),
                          Positioned(
                            top: 16,
                            left: 16,
                            right: 16,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 16,
                                vertical: 10,
                              ),
                              decoration: BoxDecoration(
                                color: Colors.white.withValues(alpha: 0.9),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Text(
                                '目标: ${_targetPosStr.length > 15 ? '${_targetPosStr.substring(0, 15)}...' : _targetPosStr}',
                                style: const TextStyle(
                                  fontSize: 13,
                                  color: Colors.black87,
                                  fontWeight: FontWeight.w500,
                                ),
                                textAlign: TextAlign.center,
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(24),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.05),
                        blurRadius: 10,
                        offset: const Offset(0, 5),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      Text(
                        _statusText,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.black87,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        _subStatusText,
                        style: const TextStyle(
                          fontSize: 14,
                          color: Colors.black54,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(
                      child: _iosBtn(
                        '开始导航',
                        isComputing ? null : () => _startTacticalNav(),
                        const Color(0xFF007AFF),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: _iosBtn(
                        '停止',
                        isComputing ? _stopNav : null,
                        const Color(0xFFFF3B30),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
              ],
            ),
          ),
          if (_suggestions.isNotEmpty) _buildSuggestionOverlay(),
        ],
      ),
    );
  }

  Widget _buildSuggestionOverlay() {
    return Positioned(
      top: 80,
      left: 16,
      right: 16,
      child: Material(
        elevation: 15,
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          constraints: const BoxConstraints(maxHeight: 300),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.black.withValues(alpha: 0.05)),
          ),
          child: ListView.separated(
            shrinkWrap: true,
            itemCount: _suggestions.length,
            separatorBuilder: (context, index) =>
                const Divider(height: 1, color: Colors.black12),
            itemBuilder: (context, index) {
              final tip = _suggestions[index];
              return ListTile(
                leading: const Icon(Icons.place, color: Colors.black38),
                title: Text(
                  tip['name'],
                  style: const TextStyle(
                    color: Colors.black87,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                subtitle: Text(
                  tip['address'] is String ? tip['address'] : '',
                  style: const TextStyle(fontSize: 12, color: Colors.black54),
                ),
                onTap: () {
                  final locData = tip['location'];
                  final locStr = locData is String ? locData : '';
                  if (locStr.isEmpty || !locStr.contains(',')) {
                    return;
                  }

                  final lng = double.parse(locStr.split(',')[0]);
                  final lat = double.parse(locStr.split(',')[1]);
                  final newTarget = LatLng(lat, lng);

                  setState(() {
                    _destController.text = tip['name'];
                    _targetCoords = locStr;
                    _targetPosStr = tip['name'];
                    _suggestions = <dynamic>[];
                    _mapCenter = newTarget;
                  });

                  _mapController?.moveCamera(
                    CameraUpdate.newCameraPosition(
                      CameraPosition(target: newTarget, zoom: 16),
                    ),
                  );
                  FocusScope.of(context).unfocus();
                },
              );
            },
          ),
        ),
      ),
    );
  }

  Widget _iosBtn(String txt, VoidCallback? fn, Color bgColor) {
    return ElevatedButton(
      onPressed: fn,
      style: ElevatedButton.styleFrom(
        backgroundColor: bgColor,
        disabledBackgroundColor: Colors.black12,
        elevation: 0,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        padding: const EdgeInsets.symmetric(vertical: 16),
      ),
      child: Text(
        txt,
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.bold,
          fontSize: 16,
        ),
      ),
    );
  }
}
