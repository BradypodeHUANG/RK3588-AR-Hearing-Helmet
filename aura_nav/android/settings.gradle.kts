pluginManagement {
    val flutterSdkPath =
        run {
            val properties = java.util.Properties()
            file("local.properties").inputStream().use { properties.load(it) }
            val flutterSdkPath = properties.getProperty("flutter.sdk")
            require(flutterSdkPath != null) { "flutter.sdk not set in local.properties" }
            flutterSdkPath
        }

    includeBuild("$flutterSdkPath/packages/flutter_tools/gradle")

    repositories {
        // ！！！精准制导：只允许 io.flutter 引擎走这个通道！！！
        maven { 
            url = uri("https://storage.flutter-io.cn/download.flutter.io") 
            content {
                includeGroup("io.flutter")
            }
        }
        
        // --- 阿里云镜像：加速其他所有常规组件 ---
        maven { url = uri("https://maven.aliyun.com/repository/google") }
        maven { url = uri("https://maven.aliyun.com/repository/central") }
        maven { url = uri("https://maven.aliyun.com/repository/public") }
        maven { url = uri("https://maven.aliyun.com/repository/gradle-plugin") }
        
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("dev.flutter.flutter-plugin-loader") version "1.0.0"
    id("com.android.application") version "8.11.1" apply false
    id("org.jetbrains.kotlin.android") version "2.2.20" apply false
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_PROJECT)
    repositories {
        // ！！！精准制导：只允许 io.flutter 引擎走这个通道！！！
        maven { 
            url = uri("https://storage.flutter-io.cn/download.flutter.io") 
            content {
                includeGroup("io.flutter")
            }
        }
        
        // --- 阿里云镜像：加速其他所有常规组件 ---
        maven { url = uri("https://maven.aliyun.com/repository/google") }
        maven { url = uri("https://maven.aliyun.com/repository/central") }
        maven { url = uri("https://maven.aliyun.com/repository/public") }
        
        google()
        mavenCentral()
    }
}

include(":app")