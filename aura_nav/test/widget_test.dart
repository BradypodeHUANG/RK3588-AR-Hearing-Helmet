import 'package:flutter_test/flutter_test.dart';

import 'package:aura_nav/main.dart';

void main() {
  test('CoordConverter returns a coordinate pair', () {
    final result = CoordConverter.wgs84ToGcj02(114.30, 30.59);

    expect(result, hasLength(2));
    expect(result[0], isNonZero);
    expect(result[1], isNonZero);
  });
}
