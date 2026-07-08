# aura_nav

A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Learn Flutter](https://docs.flutter.dev/get-started/learn-flutter)
- [Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Flutter learning resources](https://docs.flutter.dev/reference/learning-resources)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

## Android APK build on China mainland networks

This project already points Gradle at domestic mirrors for most Android
dependencies. To avoid Flutter and pub still reaching slow overseas sources,
use the helper script below when building an APK:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_apk_cn.ps1 -Mode arm64
```

Useful modes:

- `arm64`: fastest for most modern Android phones.
- `universal`: one APK for mixed device architectures, but slower and larger.
- `split-per-abi`: generates multiple smaller APK files, one per ABI.

If you only want to warm caches first:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_apk_cn.ps1 -PrepareOnly
```

APK output directory:

```text
build\app\outputs\flutter-apk
```

Notes:

- The first release build can still take a long time because Flutter release
  builds do AOT compilation, resource processing, and native packaging.
- The current Android release config still uses the debug signing key, so the
  APK can be installed directly on other phones, but a custom release keystore
  should be configured before formal distribution.
