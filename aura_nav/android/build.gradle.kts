allprojects {
    repositories {
        // Prefer the Flutter China mirror for engine artifacts so Gradle
        // does not fall back to storage.googleapis.com on restricted networks.
        maven {
            url = uri("https://storage.flutter-io.cn/download.flutter.io")
            content {
                includeGroup("io.flutter")
            }
        }
        maven { url = uri("https://maven.aliyun.com/repository/google") }
        maven { url = uri("https://maven.aliyun.com/repository/central") }
        maven { url = uri("https://maven.aliyun.com/repository/public") }
        google()
        mavenCentral()
    }
}

fun Project.inferAndroidNamespace(): String? {
    val manifestPackage =
        file("src/main/AndroidManifest.xml")
            .takeIf { it.exists() }
            ?.readText()
            ?.let { Regex("""package\s*=\s*"([^"]+)"""").find(it)?.groupValues?.getOrNull(1) }

    return when (name) {
        "amap_flutter_map" -> "com.amap.flutter.map"
        else -> manifestPackage?.takeIf { it.isNotBlank() }
    } ?: group.toString().takeIf { it.isNotBlank() && it != "unspecified" }
}

fun Project.applyMissingAndroidNamespace() {
    val androidExtension = extensions.findByName("android") ?: return
    val getNamespace =
        androidExtension.javaClass.methods.find {
            it.name == "getNamespace" && it.parameterCount == 0
        }
    val setNamespace =
        androidExtension.javaClass.methods.find {
            it.name == "setNamespace" && it.parameterCount == 1
        }
            ?: return

    val currentNamespace = getNamespace?.invoke(androidExtension) as? String
    if (!currentNamespace.isNullOrBlank()) {
        return
    }

    val inferredNamespace = inferAndroidNamespace() ?: return
    setNamespace.invoke(androidExtension, inferredNamespace)
}

subprojects {
    pluginManager.withPlugin("com.android.library") {
        applyMissingAndroidNamespace()
    }

    afterEvaluate {
        applyMissingAndroidNamespace()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
