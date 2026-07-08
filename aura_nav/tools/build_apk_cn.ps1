param(
    [ValidateSet("arm64", "universal", "split-per-abi")]
    [string]$Mode = "arm64",
    [switch]$Offline,
    [switch]$SkipPrecache,
    [switch]$SkipPubGet,
    [switch]$ForcePrecache,
    [switch]$ForcePubGet,
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localPropertiesPath = Join-Path $projectRoot "android\local.properties"
$packageConfigPath = Join-Path $projectRoot ".dart_tool\package_config.json"

$flutterSdk = $null
$androidSdk = $null

if (Test-Path $localPropertiesPath) {
    foreach ($line in Get-Content $localPropertiesPath) {
        if ($line -match '^flutter\.sdk=(.+)$') {
            $flutterSdk = $Matches[1] -replace '\\\\', '\'
        }

        if ($line -match '^sdk\.dir=(.+)$') {
            $androidSdk = $Matches[1] -replace '\\\\', '\'
        }
    }
}

$flutterCmd = if ($flutterSdk) {
    $candidate = Join-Path $flutterSdk "bin\flutter.bat"
    if (Test-Path $candidate) { $candidate } else { "flutter" }
} else {
    "flutter"
}

if (-not $env:PUB_HOSTED_URL) {
    $env:PUB_HOSTED_URL = "https://pub.flutter-io.cn"
}

if (-not $env:FLUTTER_STORAGE_BASE_URL) {
    $env:FLUTTER_STORAGE_BASE_URL = "https://storage.flutter-io.cn"
}

if (-not $env:ANDROID_SDK_ROOT -and $androidSdk) {
    $env:ANDROID_SDK_ROOT = $androidSdk
}

if (-not $env:ANDROID_HOME -and $androidSdk) {
    $env:ANDROID_HOME = $androidSdk
}

Write-Host "Project root: $projectRoot"
Write-Host "Flutter SDK: $flutterCmd"
Write-Host "PUB_HOSTED_URL: $env:PUB_HOSTED_URL"
Write-Host "FLUTTER_STORAGE_BASE_URL: $env:FLUTTER_STORAGE_BASE_URL"
if ($androidSdk) {
    Write-Host "Android SDK: $androidSdk"
}

if ($Offline) {
    if ($env:GRADLE_OPTS) {
        $env:GRADLE_OPTS = "$($env:GRADLE_OPTS) -Dorg.gradle.offline=true"
    }
    else {
        $env:GRADLE_OPTS = "-Dorg.gradle.offline=true"
    }

    Write-Host "Gradle offline: enabled"
}

$engineReleaseCache = @(
    "android-arm64-release",
    "android-arm-release",
    "android-x64-release"
) | ForEach-Object {
    Join-Path (Split-Path $flutterCmd -Parent | Join-Path -ChildPath "cache\artifacts\engine") $_
}

$hasEngineCache = $engineReleaseCache | Where-Object { Test-Path $_ } | Measure-Object | Select-Object -ExpandProperty Count
$needPrecache = $ForcePrecache -or ((-not $SkipPrecache) -and ($hasEngineCache -eq 0))

$needPubGet = $ForcePubGet -or ((-not $SkipPubGet) -and (-not (Test-Path $packageConfigPath)))

if ($PrepareOnly) {
    Write-Host "PrepareOnly mode:"
    Write-Host ("- Engine cache present: " + ($(if ($hasEngineCache -gt 0) { "yes" } else { "no" })))
    Write-Host ("- package_config.json present: " + ($(if (Test-Path $packageConfigPath) { "yes" } else { "no" })))
    Write-Host ("- Will run precache: " + ($(if ($needPrecache) { "yes" } else { "no" })))
    Write-Host ("- Will run pub get: " + ($(if ($needPubGet) { "yes" } else { "no" })))
}

Push-Location $projectRoot
try {
    if ($needPrecache) {
        & $flutterCmd precache --android
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    elseif (-not $SkipPrecache) {
        Write-Host "Skipping precache because Android engine artifacts are already cached."
    }

    if ($needPubGet) {
        & $flutterCmd pub get
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    elseif (-not $SkipPubGet) {
        Write-Host "Skipping pub get because .dart_tool/package_config.json already exists."
    }

    if ($PrepareOnly) {
        Write-Host "Environment prepared. Skipping APK build."
        exit 0
    }

    $buildArgs = @("build", "apk", "--release")

    switch ($Mode) {
        "arm64" {
            $buildArgs += @("--target-platform", "android-arm64")
        }
        "split-per-abi" {
            $buildArgs += "--split-per-abi"
        }
        "universal" {
        }
    }

    if (-not $needPubGet) {
        $buildArgs += "--no-pub"
    }

    Write-Host ("Running: flutter " + ($buildArgs -join " "))
    & $flutterCmd @buildArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $outputRoot = Join-Path $projectRoot "build\app\outputs\flutter-apk"
    if (Test-Path $outputRoot) {
        Write-Host ""
        Write-Host "Generated APK files:"
        Get-ChildItem $outputRoot -Filter "*.apk" |
            Sort-Object LastWriteTime -Descending |
            Select-Object FullName, Length, LastWriteTime |
            Format-Table -AutoSize
    }
}
finally {
    Pop-Location
}
