#include <xf86drm.h>
#include <xf86drmMode.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <sstream>
#include <string>
#include <cstdio>
#include <array>
#include <deque>
#include <iomanip>
#include <filesystem>
#include <csignal>
#include <cstdlib>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <termios.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/rfcomm.h>
#include <bluetooth/sdp.h>
#include <bluetooth/sdp_lib.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

// --- NanoVG ---
#define NANOVG_GLES2_IMPLEMENTATION
#include "nanovg/src/nanovg.h"
#include "nanovg/src/nanovg_gl.h"

using json = nlohmann::json;
struct VertexData { float x, y, z, u, v; };
struct EyeMeshData { std::vector<VertexData> vertices; std::vector<unsigned short> indices; };
struct Point2D { float x, y; };

// ==========================================
// 操作系统架构：状态机与全局变量
// ==========================================
enum class SysState { STANDBY, BOOT_PLAY, BOOT_FADE, BOOT_WAIT, DESKTOP, APP_RADAR, APP_SIGN_TRANSLATE, APP_REAR_CAMERA, APP_NAVIGATION, APP_SPEECH_TO_TEXT, APP_HEARING_ASSIST, APP_PLACEHOLDER, APP_RIDING_MODE };

std::atomic<SysState> g_target_state(SysState::STANDBY);
std::atomic<int> g_selected_app(0);
const int TOTAL_APPS = 7;
constexpr float kDefaultSymmetricShear = -0.14600f;
constexpr float kSymmetricShearAdjustStep = 0.001f;
constexpr const char* kArHudPidFile = "/tmp/ar_app1_hud.pid";
std::atomic<bool> g_desktop_tuning_mode(false);
std::atomic<float> g_symmetric_shear(kDefaultSymmetricShear);
std::atomic<unsigned long long> g_symmetric_shear_revision(0);

struct SoundAlert { float angle; float distance; float start_time; float life_time; };
std::vector<SoundAlert> g_active_alerts;
float g_last_spawn_time = 0.0f;

enum class RadarSoundType { HORN, DOG };

struct RadarDetection {
    RadarSoundType type = RadarSoundType::HORN;
    float angle_deg = 0.0f;
    float confidence = 0.0f;
    float timestamp = 0.0f;
};

struct RadarUiTarget {
    RadarSoundType type = RadarSoundType::HORN;
    float target_angle_deg = 0.0f;
    float display_angle_deg = 0.0f;
    float angle_velocity_deg = 0.0f;
    float confidence = 0.0f;
    float shown_confidence = 0.0f;
    float last_update_time = 0.0f;
    bool active = false;
};

struct RadarRuntime {
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<RadarUiTarget> targets;
    bool app_running = false;
    bool shutdown = false;
    pid_t service_pid = -1;
};

RadarRuntime g_radar_runtime;
constexpr float kRadarDuplicateAngleDeg = 10.0f;
constexpr float kRadarLifetimeSec = 6.0f;
constexpr float kRadarAngleSpring = 22.0f;
constexpr float kRadarAngleDamping = 8.0f;
constexpr float kRadarAngleMaxSpeedDeg = 260.0f;

// Rear camera app configuration.
constexpr const char* kRearCameraDevice = "/dev/video21";
constexpr int kRearCameraCaptureWidth = 640;
constexpr int kRearCameraCaptureHeight = 480;
constexpr int kRearCameraCaptureFps = 60;
constexpr float kRearCameraDisplayWidthPx = 720.0f;
constexpr float kRearCameraDisplayHeightPx =
    kRearCameraDisplayWidthPx *
    (static_cast<float>(kRearCameraCaptureHeight) /
     static_cast<float>(kRearCameraCaptureWidth));

struct RearCameraSettings {
    std::string source = kRearCameraDevice;
    int width = kRearCameraCaptureWidth;
    int height = kRearCameraCaptureHeight;
    int fps = kRearCameraCaptureFps;
    float display_width_px = kRearCameraDisplayWidthPx;
    float display_height_px = kRearCameraDisplayHeightPx;
};

struct RearCameraRuntime {
    std::mutex mtx;
    std::condition_variable cv;
    cv::Mat latest_rgba;
    RearCameraSettings settings;
    bool app_running = false;
    bool shutdown = false;
    bool source_ready = false;
    bool frame_ready = false;
    bool open_failed = false;
    float last_frame_time = 0.0f;
    std::string status_text = "CAMERA OFFLINE";
    std::string source_label = "REAR CAMERA";
};

RearCameraRuntime g_rear_camera_runtime;

constexpr float kSignTranslateTextLifetimeSec = 1.5f;
constexpr float kSpeechToTextTextLifetimeSec = 3.0f;
constexpr float kHearingAssistSignTextLifetimeSec = 3.2f;
constexpr float kHearingAssistSpeechTextLifetimeSec = 4.6f;
constexpr double kSignTranslateTtsDedupSec = 1.2;
constexpr const char* kDefaultSignTranslateTtsDevice = "plughw:4,0";
constexpr const char* kDefaultSignTranslateTtsOutput = "/tmp/sign_translate_tts.wav";
constexpr const char* kSignTranslateTtsCacheDir = "/home/elf/.cache/sign_translate_tts_cache";
constexpr const char* kSignTranslateTtsServiceScript = "sign_translate_tts_service.py";
constexpr const char* kSignTranslateTtsServiceStderrLog = "/tmp/sign_translate_tts_service.stderr.log";

struct SignTranslateRuntime {
    std::mutex mtx;
    std::condition_variable cv;
    bool app_running = false;
    bool background_capture_enabled = false;
    bool shutdown = false;
    bool backend_ready = false;
    bool backend_thinking = false;
    pid_t service_pid = -1;
    int output_fd = -1;
    int input_fd = -1;
    std::string display_text;
    float display_confidence = 0.0f;
    float display_start_time = -1000.0f;
    float thinking_transition_time = -1000.0f;
    std::string status_text = "等待手语输入";
};

SignTranslateRuntime g_sign_translate_runtime;

struct SignTranslateTtsRuntime {
    std::mutex mtx;
    std::condition_variable cv;
    bool app_running = false;
    bool state_dirty = false;
    bool shutdown = false;
    bool warned_unavailable = false;
    std::string pending_text;
    std::string last_spoken_text;
    double last_spoken_time_sec = -1000.0;
};

SignTranslateTtsRuntime g_sign_translate_tts_runtime;

struct SpeechToTextRuntime {
    std::mutex mtx;
    std::condition_variable cv;
    bool app_running = false;
    bool shutdown = false;
    bool backend_ready = false;
    pid_t service_pid = -1;
    int output_fd = -1;
    std::string display_text;
    std::string status_text = "WAITING FOR SPEECH";
    float display_start_time = -1000.0f;
    int last_segment_id = -1;
};

SpeechToTextRuntime g_speech_to_text_runtime;

constexpr size_t kHandNodeCount = 26;
constexpr size_t kHandNodeCoordCount = 3;
constexpr size_t kHandCaptureMaxFrames = 360;
constexpr double kHandCaptureEndGapSec = 0.18;

struct HandFrame {
    long long frame_id = 0;
    long long timestamp_us = 0;
    int hand_type = 1;
    float palm_x = 0.0f;
    float palm_y = 0.0f;
    float palm_z = 0.0f;
    std::array<std::array<float, kHandNodeCoordCount>, kHandNodeCount> nodes{};
};

struct HandCaptureRuntime {
    std::mutex mtx;
    std::deque<HandFrame> frames;
    bool active = false;
    bool saved_once = false;
    long long capture_id = 0;
    double last_right_hand_time_sec = -1000.0;
    double last_terminal_print_sec = -1000.0;
    std::filesystem::path log_dir;
};

HandCaptureRuntime g_hand_capture_runtime;

// ==========================================
// 导航子系统：全局数据总线与 UDP 线程
// ==========================================
struct NavContext {
    std::mutex mtx;
    std::string state = "STANDBY";
    float state_timer = 0.0f;
    float last_packet_time = 0.0f;
    
    bool has_origin = false;
    double origin_lng = 0.0;
    double origin_lat = 0.0;
    std::vector<Point2D> route_pts;
    std::string last_path_str = "";
    
    float car_x = 0.0f, car_y = 0.0f;
    float display_car_x = 0.0f, display_car_y = 0.0f;
    float raw_heading = M_PI / 2.0f;
    float smooth_heading = M_PI / 2.0f;
    
    float total_dist = 1.0f;
    float current_dist = 0.0f;
    float dist_to_turn = 9999.0f;
    float display_current_dist = 0.0f;
    float display_dist_to_turn = 9999.0f;
    std::string action_str = "";
    float target_turn_type = 0.0f;
    std::string dest_name = "";
    
    float arrow_smooth = 0.0f;
};
NavContext g_nav;

constexpr uint8_t kBluetoothNavChannel = 1;
constexpr const char* kButtonSerialDevice = "/dev/ttyS9";
constexpr speed_t kButtonSerialBaud = B115200;
constexpr int kButtonSerialRetryDelayMs = 1000;

enum class InputAction { LEFT, RIGHT, ENTER, BACK, TOGGLE_ENTER_BACK };

std::mutex g_input_action_mutex;
void dispatch_input_action(InputAction action, const char* input_source);

float adjust_symmetric_shear(float delta, const char* input_source) {
    float current = g_symmetric_shear.load(std::memory_order_relaxed);
    float desired = current + delta;
    while (!g_symmetric_shear.compare_exchange_weak(
        current,
        desired,
        std::memory_order_relaxed,
        std::memory_order_relaxed)) {
        desired = current + delta;
    }
    g_symmetric_shear_revision.fetch_add(1, std::memory_order_relaxed);
    char shear_buf[64];
    snprintf(shear_buf, sizeof(shear_buf), "%.5f", desired);
    std::cout << ">>> [" << input_source << "] kDefaultSymmetricShear = "
              << shear_buf << "\n";
    return desired;
}

void set_desktop_tuning_mode_locked(bool enabled, const char* input_source) {
    if (g_target_state.load() != SysState::DESKTOP) return;

    const bool previous = g_desktop_tuning_mode.exchange(enabled, std::memory_order_relaxed);
    if (previous == enabled) return;

    char shear_buf[64];
    snprintf(
        shear_buf,
        sizeof(shear_buf),
        "%.5f",
        g_symmetric_shear.load(std::memory_order_relaxed));
    std::cout << ">>> [" << input_source << "] "
              << (enabled ? "desktop tuning mode enabled" : "desktop tuning mode disabled")
              << ", kDefaultSymmetricShear = " << shear_buf << "\n";
}

void toggle_desktop_tuning_mode(const char* input_source) {
    std::lock_guard<std::mutex> lock(g_input_action_mutex);
    if (g_target_state.load() != SysState::DESKTOP) return;
    const bool enable = !g_desktop_tuning_mode.load(std::memory_order_relaxed);
    set_desktop_tuning_mode_locked(enable, input_source);
}

float monotonic_time_sec() {
    return std::chrono::duration<float>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

float normalize_angle_deg(float angle) {
    while (angle > 180.0f) angle -= 360.0f;
    while (angle < -180.0f) angle += 360.0f;
    return angle;
}

float angle_distance_deg(float a, float b) {
    return std::fabs(normalize_angle_deg(a - b));
}

void stop_child_process(pid_t pid) {
    if (pid <= 0) return;

    pid_t pgid = getpgid(pid);
    if (pgid > 0) {
        kill(-pgid, SIGTERM);
    } else {
        kill(pid, SIGTERM);
    }
    for (int i = 0; i < 40; ++i) {
        int status = 0;
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid || result == -1) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (pgid > 0) {
        kill(-pgid, SIGKILL);
    } else {
        kill(pid, SIGKILL);
    }
    waitpid(pid, nullptr, 0);
}

void stop_process_by_pid(pid_t pid) {
    if (pid <= 0 || pid == getpid()) return;
    if (kill(pid, 0) != 0) return;

    kill(pid, SIGTERM);
    for (int i = 0; i < 40; ++i) {
        if (kill(pid, 0) != 0) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    kill(pid, SIGKILL);
}

void run_quiet_system_command(const std::string& command) {
    if (command.empty()) return;
    int rc = std::system(command.c_str());
    (void)rc;
}

void cleanup_stale_runtime_helpers() {
    {
        std::ifstream pid_file(kArHudPidFile);
        pid_t old_pid = -1;
        if (pid_file >> old_pid) {
            stop_process_by_pid(old_pid);
        }
    }

    run_quiet_system_command("pkill -f ultraleap_rknn_live >/dev/null 2>&1");
    run_quiet_system_command("pkill -f sherpa-onnx-alsa >/dev/null 2>&1");
    run_quiet_system_command("pkill -f sound_radar_service.py >/dev/null 2>&1");

    std::ofstream pid_file(kArHudPidFile, std::ios::trunc);
    if (pid_file.is_open()) {
        pid_file << getpid() << "\n";
    }
}


double json_to_double(const json& payload, const char* key, double fallback) {
    auto it = payload.find(key);
    if (it == payload.end() || it->is_null()) return fallback;

    try {
        if (it->is_number()) return it->get<double>();
        if (it->is_string()) return std::stod(it->get<std::string>());
    } catch (...) {
    }
    return fallback;
}

std::string json_to_string(const json& payload, const char* key, const std::string& fallback) {
    auto it = payload.find(key);
    if (it == payload.end() || it->is_null()) return fallback;

    try {
        if (it->is_string()) return it->get<std::string>();
        if (it->is_number()) return std::to_string(it->get<double>());
        if (it->is_boolean()) return it->get<bool>() ? "true" : "false";
    } catch (...) {
    }
    return fallback;
}

bool radar_type_from_string(const std::string& value, RadarSoundType& out_type) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (lower == "horn" || lower == "car_horn" || lower == "vehicle_horn") {
        out_type = RadarSoundType::HORN;
        return true;
    }
    if (lower == "dog" || lower == "dog_bark" || lower == "bark") {
        out_type = RadarSoundType::DOG;
        return true;
    }
    return false;
}

void set_radar_app_running(bool running) {
    {
        std::lock_guard<std::mutex> lock(g_radar_runtime.mtx);
        g_radar_runtime.app_running = running;
        if (!running) {
            g_radar_runtime.targets.clear();
        }
    }
    g_radar_runtime.cv.notify_all();
}

bool wait_radar_service_state(bool should_be_running, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        bool running = false;
        bool app_running = false;
        {
            std::lock_guard<std::mutex> lock(g_radar_runtime.mtx);
            running = g_radar_runtime.service_pid > 0;
            app_running = g_radar_runtime.app_running;
        }

        if (should_be_running) {
            if (app_running && running) return true;
        } else {
            if (!app_running && !running) return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

int env_to_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

RearCameraSettings load_rear_camera_settings() {
    RearCameraSettings settings;
    settings.source = kRearCameraDevice;
    settings.width = kRearCameraCaptureWidth;
    settings.height = kRearCameraCaptureHeight;
    settings.fps = kRearCameraCaptureFps;
    settings.display_width_px = kRearCameraDisplayWidthPx;
    settings.display_height_px = kRearCameraDisplayHeightPx;
    return settings;
}

std::string rear_camera_source_label(const RearCameraSettings& settings) {
    if (!settings.source.empty() && std::all_of(settings.source.begin(), settings.source.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        })) {
        return "USB CAMERA #" + settings.source;
    }
    return settings.source;
}

bool open_rear_camera_capture(const RearCameraSettings& settings, cv::VideoCapture& cap) {
    cap.release();

    bool opened = false;
    if (!settings.source.empty() && std::all_of(settings.source.begin(), settings.source.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        })) {
        const int index = std::stoi(settings.source);
        opened = cap.open(index, cv::CAP_V4L2);
        if (!opened) opened = cap.open(index);
    } else {
        opened = cap.open(settings.source, cv::CAP_V4L2);
        if (!opened) opened = cap.open(settings.source);
    }

    if (!opened || !cap.isOpened()) return false;

    cap.set(cv::CAP_PROP_FRAME_WIDTH, settings.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, settings.height);
    cap.set(cv::CAP_PROP_FPS, settings.fps);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    return true;
}

cv::Mat apply_rear_camera_transform(const cv::Mat& frame_bgr, const RearCameraSettings& settings) {
    if (frame_bgr.empty()) return {};
    cv::Mat transformed;
    if (frame_bgr.cols != settings.width || frame_bgr.rows != settings.height) {
        cv::resize(frame_bgr, transformed, cv::Size(settings.width, settings.height), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        transformed = frame_bgr.clone();
    }
    return transformed;
}

bool file_exists_quiet(const std::string& path) {
    if (path.empty()) return false;
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

std::string resolve_executable_dir() {
    try {
        return std::filesystem::canonical("/proc/self/exe").parent_path().string();
    } catch (...) {
        try {
            return std::filesystem::current_path().string();
        } catch (...) {
            return ".";
        }
    }
}

std::string first_existing_path(const std::vector<std::string>& candidates) {
    for (const auto& candidate : candidates) {
        if (file_exists_quiet(candidate)) return candidate;
    }
    return "";
}

std::string trim_ascii(const std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

bool is_valid_utf8(const std::string& text) {
    int remaining = 0;
    for (unsigned char c : text) {
        if (remaining == 0) {
            if ((c >> 7) == 0b0) continue;
            if ((c >> 5) == 0b110) remaining = 1;
            else if ((c >> 4) == 0b1110) remaining = 2;
            else if ((c >> 3) == 0b11110) remaining = 3;
            else return false;
        } else {
            if ((c >> 6) != 0b10) return false;
            --remaining;
        }
    }
    return remaining == 0;
}

std::string strip_ascii_control_chars(const std::string& text) {
    std::string cleaned;
    cleaned.reserve(text.size());
    for (unsigned char c : text) {
        if (c == '\t' || c == ' ' || c >= 0x20) {
            if (c != 0x7f) {
                cleaned.push_back(static_cast<char>(c));
            }
        }
    }
    return cleaned;
}

std::string strip_ansi_sequences(const std::string& text) {
    std::string cleaned;
    cleaned.reserve(text.size());

    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == 0x1b) {
            if (i + 1 < text.size() && text[i + 1] == '[') {
                i += 2;
                while (i < text.size()) {
                    unsigned char seq = static_cast<unsigned char>(text[i]);
                    if ((seq >= '@' && seq <= '~') || std::isalpha(seq)) {
                        break;
                    }
                    ++i;
                }
                continue;
            }
            continue;
        }
        cleaned.push_back(static_cast<char>(c));
    }

    return cleaned;
}

std::vector<std::string> detect_preferred_alsa_playback_devices(const std::string& preferred_device) {
    std::vector<std::string> devices;
    auto add_device = [&](const std::string& device) {
        if (device.empty()) return;
        if (std::find(devices.begin(), devices.end(), device) == devices.end()) {
            devices.push_back(device);
        }
    };

    add_device(preferred_device);

    FILE* pipe = popen("aplay -l 2>/dev/null", "r");
    if (!pipe) {
        add_device("default");
        return devices;
    }

    char buffer[512];
    std::vector<std::pair<std::string, std::string>> usb_playback_devices;
    std::vector<std::pair<std::string, std::string>> other_playback_devices;

    while (fgets(buffer, sizeof(buffer), pipe)) {
        std::string line(buffer);
        if (line.rfind("card ", 0) != 0) continue;

        const size_t card_start = 5;
        const size_t card_end = line.find(':', card_start);
        if (card_end == std::string::npos) continue;
        const std::string card_index = trim_ascii(line.substr(card_start, card_end - card_start));

        const size_t short_name_start = card_end + 1;
        const size_t short_name_end = line.find('[', short_name_start);
        std::string short_name = trim_ascii(
            line.substr(short_name_start, short_name_end == std::string::npos ? std::string::npos
                                                                              : short_name_end - short_name_start));

        const size_t device_marker = line.find("device ", card_end);
        if (device_marker == std::string::npos) continue;
        const size_t device_start = device_marker + 7;
        const size_t device_end = line.find(':', device_start);
        if (device_end == std::string::npos) continue;
        const std::string device_index = trim_ascii(line.substr(device_start, device_end - device_start));

        std::string lower = line;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        const std::string plughw = "plughw:" + card_index + "," + device_index;
        const std::string named = short_name.empty()
            ? std::string()
            : "plughw:CARD=" + short_name + ",DEV=" + device_index;
        const std::string sysdefault = short_name.empty()
            ? std::string()
            : "sysdefault:CARD=" + short_name;

        if (lower.find("uacdemo") != std::string::npos ||
            lower.find("usb") != std::string::npos ||
            lower.find("audio") != std::string::npos) {
            usb_playback_devices.push_back({plughw, named});
            if (!sysdefault.empty()) usb_playback_devices.push_back({sysdefault, ""});
        } else {
            other_playback_devices.push_back({plughw, named});
            if (!sysdefault.empty()) other_playback_devices.push_back({sysdefault, ""});
        }
    }
    pclose(pipe);

    auto add_pairs = [&](const std::vector<std::pair<std::string, std::string>>& entries) {
        for (const auto& entry : entries) {
            add_device(entry.first);
            add_device(entry.second);
        }
    };

    add_pairs(usb_playback_devices);
    if (usb_playback_devices.empty()) {
        add_pairs(other_playback_devices);
    }

    add_device("default");
    return devices;
}

std::string make_tts_cache_path(const std::string& text) {
    static const char* hex = "0123456789abcdef";
    std::string key;
    key.reserve(text.size() * 2);
    for (unsigned char c : text) {
        key.push_back(hex[(c >> 4) & 0x0f]);
        key.push_back(hex[c & 0x0f]);
    }
    return std::string(kSignTranslateTtsCacheDir) + "/" + key + ".wav";
}

std::string normalize_tts_text(const std::string& text) {
    std::string normalized;
    normalized.reserve(text.size());
    bool previous_was_space = false;
    for (unsigned char c : text) {
        const bool is_space = std::isspace(c) != 0;
        if (is_space) {
            if (!normalized.empty() && !previous_was_space) {
                normalized.push_back(' ');
            }
        } else {
            normalized.push_back(static_cast<char>(c));
        }
        previous_was_space = is_space;
    }

    while (!normalized.empty() && normalized.front() == ' ') normalized.erase(normalized.begin());
    while (!normalized.empty() && normalized.back() == ' ') normalized.pop_back();
    return normalized;
}

bool run_process_wait(const std::vector<std::string>& args, bool quiet = true) {
    if (args.empty() || args[0].empty()) return false;

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid == 0) {
        if (quiet) {
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                dup2(devnull, STDOUT_FILENO);
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }
        }
        execv(args[0].c_str(), argv.data());
        _exit(127);
    }

    if (pid < 0) return false;

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return false;
    }

    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

struct SignTranslateTtsServiceProcess {
    pid_t pid = -1;
    FILE* input = nullptr;
    FILE* output = nullptr;
};

void close_sign_translate_tts_service(SignTranslateTtsServiceProcess& service) {
    if (service.input) {
        json req;
        req["cmd"] = "shutdown";
        const std::string payload = req.dump();
        std::fprintf(service.input, "%s\n", payload.c_str());
        std::fflush(service.input);
        std::fclose(service.input);
        service.input = nullptr;
    }
    if (service.output) {
        std::fclose(service.output);
        service.output = nullptr;
    }
    if (service.pid > 0) {
        stop_child_process(service.pid);
        service.pid = -1;
    }
}

bool start_sign_translate_tts_service(
    const std::string& python_bin,
    const std::string& script_path,
    const std::string& acoustic_model,
    const std::string& vocoder_model,
    const std::string& lexicon_path,
    const std::string& tokens_path,
    const std::string& rule_fsts,
    SignTranslateTtsServiceProcess& service) {
    close_sign_translate_tts_service(service);

    int stdout_pipe[2] = {-1, -1};
    int stdin_pipe[2] = {-1, -1};
    if (pipe(stdout_pipe) != 0 || pipe(stdin_pipe) != 0) {
        if (stdout_pipe[0] >= 0) close(stdout_pipe[0]);
        if (stdout_pipe[1] >= 0) close(stdout_pipe[1]);
        if (stdin_pipe[0] >= 0) close(stdin_pipe[0]);
        if (stdin_pipe[1] >= 0) close(stdin_pipe[1]);
        return false;
    }

    pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        close(stdout_pipe[0]);
        close(stdin_pipe[1]);
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        int err_fd = open(kSignTranslateTtsServiceStderrLog, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (err_fd >= 0) {
            dup2(err_fd, STDERR_FILENO);
            close(err_fd);
        }
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        char* argv_exec[16];
        int argi = 0;
        argv_exec[argi++] = const_cast<char*>(python_bin.c_str());
        argv_exec[argi++] = const_cast<char*>("-u");
        argv_exec[argi++] = const_cast<char*>(script_path.c_str());
        argv_exec[argi++] = const_cast<char*>("--acoustic-model");
        argv_exec[argi++] = const_cast<char*>(acoustic_model.c_str());
        argv_exec[argi++] = const_cast<char*>("--vocoder");
        argv_exec[argi++] = const_cast<char*>(vocoder_model.c_str());
        argv_exec[argi++] = const_cast<char*>("--lexicon");
        argv_exec[argi++] = const_cast<char*>(lexicon_path.c_str());
        argv_exec[argi++] = const_cast<char*>("--tokens");
        argv_exec[argi++] = const_cast<char*>(tokens_path.c_str());
        argv_exec[argi++] = const_cast<char*>("--rule-fsts");
        argv_exec[argi++] = const_cast<char*>(rule_fsts.c_str());
        argv_exec[argi++] = nullptr;
        execv(python_bin.c_str(), argv_exec);
        _exit(127);
    }

    close(stdout_pipe[1]);
    close(stdin_pipe[0]);
    if (pid <= 0) {
        close(stdout_pipe[0]);
        close(stdin_pipe[1]);
        return false;
    }

    FILE* out = fdopen(stdout_pipe[0], "r");
    FILE* in = fdopen(stdin_pipe[1], "w");
    if (!out || !in) {
        if (out) std::fclose(out); else close(stdout_pipe[0]);
        if (in) std::fclose(in); else close(stdin_pipe[1]);
        stop_child_process(pid);
        return false;
    }
    std::setvbuf(in, nullptr, _IOLBF, 0);

    char line[4096];
    if (!std::fgets(line, sizeof(line), out)) {
        std::fclose(in);
        std::fclose(out);
        stop_child_process(pid);
        return false;
    }

    try {
        const json ready = json::parse(line);
        if (!ready.value("ok", false) || ready.value("type", "") != "ready") {
            std::fclose(in);
            std::fclose(out);
            stop_child_process(pid);
            return false;
        }
    } catch (...) {
        std::fclose(in);
        std::fclose(out);
        stop_child_process(pid);
        return false;
    }

    service.pid = pid;
    service.input = in;
    service.output = out;
    return true;
}

bool request_sign_translate_tts_service(
    SignTranslateTtsServiceProcess& service,
    const std::string& text,
    const std::string& output_path) {
    if (service.pid <= 0 || !service.input || !service.output) return false;

    json req;
    req["cmd"] = "synthesize";
    req["text"] = text;
    req["output"] = output_path;
    const std::string payload = req.dump();
    if (std::fprintf(service.input, "%s\n", payload.c_str()) < 0) {
        return false;
    }
    std::fflush(service.input);

    char line[4096];
    if (!std::fgets(line, sizeof(line), service.output)) {
        return false;
    }

    try {
        const json resp = json::parse(line);
        return resp.value("ok", false) && resp.value("output", "") == output_path;
    } catch (...) {
        return false;
    }
}

void queue_sign_translate_tts(const std::string& raw_text) {
    const std::string text = normalize_tts_text(raw_text);
    if (text.empty()) return;

    const double now = monotonic_time_sec();
    std::lock_guard<std::mutex> lock(g_sign_translate_tts_runtime.mtx);
    if (!g_sign_translate_tts_runtime.app_running) return;
    if (g_sign_translate_tts_runtime.pending_text == text) return;
    if (g_sign_translate_tts_runtime.last_spoken_text == text &&
        now - g_sign_translate_tts_runtime.last_spoken_time_sec < kSignTranslateTtsDedupSec) {
        return;
    }

    g_sign_translate_tts_runtime.pending_text = text;
    g_sign_translate_tts_runtime.cv.notify_all();
}

void sign_translate_tts_thread() {
    const std::string executable_dir = resolve_executable_dir();
    const char* tts_python_env = std::getenv("SIGN_TRANSLATE_TTS_PYTHON");
    const char* tts_service_env = std::getenv("SIGN_TRANSLATE_TTS_SERVICE");
    const char* tts_bin_env = std::getenv("SIGN_TRANSLATE_TTS_BIN");
    const char* tts_model_env = std::getenv("SIGN_TRANSLATE_TTS_MODEL");
    const char* tts_vocoder_env = std::getenv("SIGN_TRANSLATE_TTS_VOCODER");
    const char* tts_lexicon_env = std::getenv("SIGN_TRANSLATE_TTS_LEXICON");
    const char* tts_tokens_env = std::getenv("SIGN_TRANSLATE_TTS_TOKENS");
    const char* tts_rule_fsts_env = std::getenv("SIGN_TRANSLATE_TTS_RULE_FSTS");
    const char* tts_device_env = std::getenv("SIGN_TRANSLATE_TTS_DEVICE");

    const std::string tts_bin = tts_bin_env && *tts_bin_env
        ? std::string(tts_bin_env)
        : first_existing_path({
            "/home/elf/.local/bin/sherpa-onnx-offline-tts",
            "/usr/local/bin/sherpa-onnx-offline-tts",
            executable_dir + "/sherpa-onnx-offline-tts"
        });
    const std::string python_bin = tts_python_env && *tts_python_env
        ? std::string(tts_python_env)
        : first_existing_path({
            "/usr/bin/python3",
            "/bin/python3"
        });
    const std::string tts_service_script = tts_service_env && *tts_service_env
        ? std::string(tts_service_env)
        : first_existing_path({
            executable_dir + "/" + kSignTranslateTtsServiceScript,
            "/home/elf/" + std::string(kSignTranslateTtsServiceScript)
        });
    const std::string acoustic_model = tts_model_env && *tts_model_env
        ? std::string(tts_model_env)
        : first_existing_path({
            "/home/elf/sherpa-models/tts/matcha-zh-baker/model-steps-3.onnx",
            executable_dir + "/sherpa-models/tts/matcha-zh-baker/model-steps-3.onnx"
        });
    const std::string vocoder_model = tts_vocoder_env && *tts_vocoder_env
        ? std::string(tts_vocoder_env)
        : first_existing_path({
            "/home/elf/sherpa-models/tts/vocos-22khz-univ.onnx",
            executable_dir + "/sherpa-models/tts/vocos-22khz-univ.onnx"
        });
    const std::string lexicon_path = tts_lexicon_env && *tts_lexicon_env
        ? std::string(tts_lexicon_env)
        : first_existing_path({
            "/home/elf/sherpa-models/tts/matcha-zh-baker/lexicon.txt",
            executable_dir + "/sherpa-models/tts/matcha-zh-baker/lexicon.txt"
        });
    const std::string tokens_path = tts_tokens_env && *tts_tokens_env
        ? std::string(tts_tokens_env)
        : first_existing_path({
            "/home/elf/sherpa-models/tts/matcha-zh-baker/tokens.txt",
            executable_dir + "/sherpa-models/tts/matcha-zh-baker/tokens.txt"
        });
    const std::string rule_fsts = tts_rule_fsts_env && *tts_rule_fsts_env
        ? std::string(tts_rule_fsts_env)
        : std::string("/home/elf/sherpa-models/tts/matcha-zh-baker/phone.fst,") +
              "/home/elf/sherpa-models/tts/matcha-zh-baker/date.fst," +
              "/home/elf/sherpa-models/tts/matcha-zh-baker/number.fst";
    const std::string output_device = tts_device_env && *tts_device_env
        ? std::string(tts_device_env)
        : std::string(kDefaultSignTranslateTtsDevice);
    const std::string aplay_bin = first_existing_path({"/usr/bin/aplay", "/bin/aplay"});

    const bool tts_ready =
        !tts_bin.empty() &&
        !aplay_bin.empty() &&
        file_exists_quiet(acoustic_model) &&
        file_exists_quiet(vocoder_model) &&
        file_exists_quiet(lexicon_path) &&
        file_exists_quiet(tokens_path);
    const bool tts_service_ready =
        !python_bin.empty() &&
        !tts_service_script.empty() &&
        file_exists_quiet(tts_service_script) &&
        file_exists_quiet(acoustic_model) &&
        file_exists_quiet(vocoder_model) &&
        file_exists_quiet(lexicon_path) &&
        file_exists_quiet(tokens_path);

    std::error_code cache_ec;
    std::filesystem::create_directories(kSignTranslateTtsCacheDir, cache_ec);
    SignTranslateTtsServiceProcess tts_service;

    while (true) {
        std::string text;
        bool app_running = false;
        bool state_dirty = false;
        {
            std::unique_lock<std::mutex> lock(g_sign_translate_tts_runtime.mtx);
            g_sign_translate_tts_runtime.cv.wait(lock, [] {
                return g_sign_translate_tts_runtime.shutdown ||
                       g_sign_translate_tts_runtime.state_dirty ||
                       !g_sign_translate_tts_runtime.pending_text.empty();
            });
            if (g_sign_translate_tts_runtime.shutdown) break;
            app_running = g_sign_translate_tts_runtime.app_running;
            state_dirty = g_sign_translate_tts_runtime.state_dirty;
            g_sign_translate_tts_runtime.state_dirty = false;
            if (!app_running) {
                g_sign_translate_tts_runtime.pending_text.clear();
                lock.unlock();
                close_sign_translate_tts_service(tts_service);
                continue;
            }
            text.swap(g_sign_translate_tts_runtime.pending_text);
        }

        if (text.empty() && !state_dirty) continue;
        if (text.empty()) continue;

        if (!tts_ready && tts_service.pid <= 0) {
            std::lock_guard<std::mutex> lock(g_sign_translate_tts_runtime.mtx);
            if (!g_sign_translate_tts_runtime.warned_unavailable) {
                std::cerr << ">>> [SIGN-TTS] TTS runtime or model files are missing, speech output disabled.\n";
                g_sign_translate_tts_runtime.warned_unavailable = true;
            }
            continue;
        }

        const std::string cache_path = make_tts_cache_path(text);
        if (!file_exists_quiet(cache_path)) {
            bool synthesized = false;
            if (tts_service.pid > 0) {
                synthesized = request_sign_translate_tts_service(tts_service, text, cache_path);
                if (!synthesized) {
                    std::cerr << ">>> [SIGN-TTS] resident service request failed, restarting fallback path\n";
                    close_sign_translate_tts_service(tts_service);
                }
            }

            if (!synthesized && tts_service_ready && tts_service.pid <= 0) {
                if (start_sign_translate_tts_service(
                        python_bin,
                        tts_service_script,
                        acoustic_model,
                        vocoder_model,
                        lexicon_path,
                        tokens_path,
                        rule_fsts,
                        tts_service)) {
                    std::cout << ">>> [SIGN-TTS] resident service restarted\n";
                    synthesized = request_sign_translate_tts_service(tts_service, text, cache_path);
                }
            }

            if (!synthesized) {
                const std::vector<std::string> tts_args = {
                    tts_bin,
                    "--num-threads=2",
                    "--matcha-acoustic-model=" + acoustic_model,
                    "--matcha-vocoder=" + vocoder_model,
                    "--matcha-lexicon=" + lexicon_path,
                    "--matcha-tokens=" + tokens_path,
                    "--tts-rule-fsts=" + rule_fsts,
                    "--output-filename=" + cache_path,
                    text
                };
                if (!run_process_wait(tts_args)) {
                    std::cerr << ">>> [SIGN-TTS] failed to synthesize speech for text: " << text << "\n";
                    continue;
                }
                std::cout << ">>> [SIGN-TTS] synthesized by CLI fallback: " << text
                          << " -> " << cache_path << "\n";
            } else {
                std::cout << ">>> [SIGN-TTS] synthesized by resident service: " << text
                          << " -> " << cache_path << "\n";
            }
        } else {
            std::cout << ">>> [SIGN-TTS] cache hit: " << text
                      << " -> " << cache_path << "\n";
        }

        bool played = false;
        std::string played_via;
        const std::vector<std::string> preferred_devices =
            detect_preferred_alsa_playback_devices(output_device);
        std::vector<std::pair<std::vector<std::string>, std::string>> play_attempts;
        for (const auto& device : preferred_devices) {
            play_attempts.push_back({
                {
                    aplay_bin,
                    "-D",
                    device,
                    cache_path
                },
                device
            });
        }
        play_attempts.push_back({
            {
                aplay_bin,
                cache_path
            },
            "implicit-default"
        });

        for (const auto& attempt : play_attempts) {
            std::cout << ">>> [SIGN-TTS] trying playback via " << attempt.second << "\n";
            if (run_process_wait(attempt.first, false)) {
                played = true;
                played_via = attempt.second;
                break;
            }
            std::cerr << ">>> [SIGN-TTS] playback attempt failed via " << attempt.second << "\n";
        }

        if (!played) {
            std::cerr << ">>> [SIGN-TTS] failed to play synthesized speech. wav kept at "
                      << cache_path << "\n";
            continue;
        }
        std::cout << ">>> [SIGN-TTS] playback ok via " << played_via << "\n";

        {
            std::lock_guard<std::mutex> lock(g_sign_translate_tts_runtime.mtx);
            g_sign_translate_tts_runtime.last_spoken_text = text;
            g_sign_translate_tts_runtime.last_spoken_time_sec = monotonic_time_sec();
        }
    }

    close_sign_translate_tts_service(tts_service);
}

void set_rear_camera_app_running(bool running) {
    {
        std::lock_guard<std::mutex> lock(g_rear_camera_runtime.mtx);
        g_rear_camera_runtime.app_running = running;
        g_rear_camera_runtime.settings = load_rear_camera_settings();
        g_rear_camera_runtime.source_label = rear_camera_source_label(g_rear_camera_runtime.settings);
        g_rear_camera_runtime.status_text = running ? "CONNECTING CAMERA" : "CAMERA OFFLINE";
        g_rear_camera_runtime.source_ready = false;
        g_rear_camera_runtime.open_failed = false;
        if (!running) {
            g_rear_camera_runtime.latest_rgba.release();
            g_rear_camera_runtime.frame_ready = false;
        }
    }
    g_rear_camera_runtime.cv.notify_all();
}

void close_sign_translate_output_locked() {
    if (g_sign_translate_runtime.output_fd >= 0) {
        close(g_sign_translate_runtime.output_fd);
        g_sign_translate_runtime.output_fd = -1;
    }
    if (g_sign_translate_runtime.input_fd >= 0) {
        close(g_sign_translate_runtime.input_fd);
        g_sign_translate_runtime.input_fd = -1;
    }
}

void set_sign_translate_app_running(bool running) {
    {
        std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
        g_sign_translate_runtime.app_running = running;
        g_sign_translate_runtime.backend_ready = false;
        g_sign_translate_runtime.backend_thinking = false;
        g_sign_translate_runtime.thinking_transition_time = monotonic_time_sec();
        if (!running) {
            g_sign_translate_runtime.display_text.clear();
            g_sign_translate_runtime.display_confidence = 0.0f;
            g_sign_translate_runtime.display_start_time = -1000.0f;
            g_sign_translate_runtime.status_text = "等待手语输入";
        } else {
            g_sign_translate_runtime.status_text = "等待手语输入";
        }
    }
    g_sign_translate_runtime.cv.notify_all();
    {
        std::lock_guard<std::mutex> tts_lock(g_sign_translate_tts_runtime.mtx);
        g_sign_translate_tts_runtime.app_running = running;
        g_sign_translate_tts_runtime.state_dirty = true;
        if (!running) {
            g_sign_translate_tts_runtime.pending_text.clear();
        }
    }
    g_sign_translate_tts_runtime.cv.notify_all();
}

void set_sign_translate_background_capture(bool enabled) {
    {
        std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
        g_sign_translate_runtime.background_capture_enabled = enabled;
    }
    g_sign_translate_runtime.cv.notify_all();
}

void set_speech_to_text_app_running(bool running) {
    {
        std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
        g_speech_to_text_runtime.app_running = running;
        g_speech_to_text_runtime.backend_ready = false;
        g_speech_to_text_runtime.display_text.clear();
        g_speech_to_text_runtime.display_start_time = -1000.0f;
        g_speech_to_text_runtime.last_segment_id = -1;
        if (!running) {
            g_speech_to_text_runtime.status_text = "WAITING FOR SPEECH";
        } else {
            g_speech_to_text_runtime.status_text = "CONNECTING MICROPHONE";
        }
    }
    g_speech_to_text_runtime.cv.notify_all();
}

void send_sign_translate_command(const char* command) {
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
        fd = g_sign_translate_runtime.input_fd;
    }
    if (fd < 0 || !command) return;

    const std::string line = std::string(command) + "\n";
    ssize_t ignored = write(fd, line.c_str(), line.size());
    (void)ignored;
    std::cout << ">>> [SIGN] command: " << command << "\n";
}

bool wait_sign_translate_state(bool want_running, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lock(g_sign_translate_runtime.mtx);
    while (std::chrono::steady_clock::now() < deadline) {
        const bool started = g_sign_translate_runtime.service_pid > 0 && g_sign_translate_runtime.backend_ready;
        const bool stopped =
            g_sign_translate_runtime.service_pid <= 0 ||
            (!g_sign_translate_runtime.app_running &&
             !g_sign_translate_runtime.backend_thinking);
        if (want_running ? started : stopped) {
            return true;
        }
        g_sign_translate_runtime.cv.wait_for(lock, std::chrono::milliseconds(50));
    }
    return false;
}

bool wait_speech_to_text_state(bool want_running, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lock(g_speech_to_text_runtime.mtx);
    while (std::chrono::steady_clock::now() < deadline) {
        const bool running = g_speech_to_text_runtime.service_pid > 0 && g_speech_to_text_runtime.backend_ready;
        const bool stopped = g_speech_to_text_runtime.service_pid <= 0;
        if (want_running ? running : stopped) return true;
        g_speech_to_text_runtime.cv.wait_for(lock, std::chrono::milliseconds(50));
    }
    return false;
}

std::vector<std::string> split_tab_fields(const std::string& line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (start <= line.size()) {
        const size_t pos = line.find('\t', start);
        if (pos == std::string::npos) {
            fields.push_back(line.substr(start));
            break;
        }
        fields.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return fields;
}

bool parse_hand_frame_line(const std::string& line, HandFrame& frame) {
    const std::vector<std::string> fields = split_tab_fields(line);
    if (fields.size() < 8 || fields[0] != "HAND") return false;

    try {
        frame.frame_id = std::stoll(fields[1]);
        frame.timestamp_us = std::stoll(fields[2]);
        frame.hand_type = std::stoi(fields[3]);
        const int node_count = std::stoi(fields[4]);
        frame.palm_x = std::stof(fields[5]);
        frame.palm_y = std::stof(fields[6]);
        frame.palm_z = std::stof(fields[7]);

        const size_t expected_values =
            static_cast<size_t>(std::max(0, node_count)) * kHandNodeCoordCount;
        if (fields.size() < 8 + expected_values) return false;

        const size_t usable_nodes =
            std::min(kHandNodeCount, static_cast<size_t>(std::max(0, node_count)));
        for (size_t node = 0; node < usable_nodes; ++node) {
            const size_t base = 8 + node * kHandNodeCoordCount;
            frame.nodes[node][0] = std::stof(fields[base]);
            frame.nodes[node][1] = std::stof(fields[base + 1]);
            frame.nodes[node][2] = std::stof(fields[base + 2]);
        }
    } catch (...) {
        return false;
    }
    return true;
}

std::filesystem::path ensure_hand_capture_log_dir() {
    if (!g_hand_capture_runtime.log_dir.empty()) {
        return g_hand_capture_runtime.log_dir;
    }

    std::filesystem::path dir = std::filesystem::path(resolve_executable_dir()) / "hand_frame_logs";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    g_hand_capture_runtime.log_dir = dir;
    return dir;
}

void save_hand_capture_window(
    const std::vector<HandFrame>& frames,
    long long capture_id) {
    if (frames.empty()) return;

    const std::filesystem::path root = ensure_hand_capture_log_dir();
    std::ostringstream folder_name;
    folder_name << "capture_" << std::setw(4) << std::setfill('0') << capture_id;
    const std::filesystem::path capture_dir = root / folder_name.str();

    std::error_code ec;
    std::filesystem::create_directories(capture_dir, ec);
    if (ec) {
        std::cerr << ">>> [HAND] failed to create log dir: " << capture_dir << "\n";
        return;
    }

    const std::filesystem::path csv_path = capture_dir / "tracking_window.csv";
    std::ofstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << ">>> [HAND] failed to open log file: " << csv_path << "\n";
        return;
    }

    file << "index,frame_id,timestamp_us,hand_type,palm_x,palm_y,palm_z";
    for (size_t node = 0; node < kHandNodeCount; ++node) {
        file << ",node" << node << "_x,node" << node << "_y,node" << node << "_z";
    }
    file << "\n";

    file << std::fixed << std::setprecision(3);
    size_t index = 0;
    for (const HandFrame& frame : frames) {
        file << index++
             << "," << frame.frame_id
             << "," << frame.timestamp_us
             << "," << frame.hand_type
             << "," << frame.palm_x
             << "," << frame.palm_y
             << "," << frame.palm_z;
        for (size_t node = 0; node < kHandNodeCount; ++node) {
            file << "," << frame.nodes[node][0]
                 << "," << frame.nodes[node][1]
                 << "," << frame.nodes[node][2];
        }
        file << "\n";
    }

    std::cout << ">>> [HAND] saved one capture window: "
              << csv_path << " (" << frames.size() << " frames, max "
              << kHandCaptureMaxFrames << ")\n";
}

void check_hand_capture_timeout() {
    std::vector<HandFrame> frames_to_save;
    long long capture_id = 0;
    {
        std::lock_guard<std::mutex> lock(g_hand_capture_runtime.mtx);
        if (!g_hand_capture_runtime.active) return;

        const double now = monotonic_time_sec();
        if (now - g_hand_capture_runtime.last_right_hand_time_sec < kHandCaptureEndGapSec) return;

        if (!g_hand_capture_runtime.saved_once) {
            frames_to_save.assign(
                g_hand_capture_runtime.frames.begin(),
                g_hand_capture_runtime.frames.end());
            capture_id = g_hand_capture_runtime.capture_id;
            g_hand_capture_runtime.saved_once = true;
        }

        g_hand_capture_runtime.active = false;
        g_hand_capture_runtime.frames.clear();
    }

    if (!frames_to_save.empty()) {
        save_hand_capture_window(frames_to_save, capture_id);
    }
}

float hand_frame_node_distance(
    const HandFrame& frame,
    size_t a,
    size_t b) {
    if (a >= kHandNodeCount || b >= kHandNodeCount) return 9999.0f;
    const float dx = frame.nodes[a][0] - frame.nodes[b][0];
    const float dy = frame.nodes[a][1] - frame.nodes[b][1];
    const float dz = frame.nodes[a][2] - frame.nodes[b][2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void handle_hand_tracking_line(const std::string& line) {
    // 手部追踪数据捕获和打印功能已移除
    // 手语翻译app通过 handle_sign_translate_output_line 接收原始 HAND\t 数据行
    // 无需在此处理，数据已在 sign_translate_supervisor_thread 中传输给手语识别后端
}

void handle_sign_translate_output_line(const std::string& line) {
    if (line.rfind("HAND\t", 0) == 0) {
        handle_hand_tracking_line(line);
    } else if (line.rfind("STATE\t", 0) == 0) {
        const std::string state_name = line.substr(6);
        std::cout << ">>> [SIGN] state: " << state_name << "\n";
        std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
        if (state_name == "READY") {
            g_sign_translate_runtime.backend_ready = true;
            g_sign_translate_runtime.status_text = "等待手语输入";
        } else if (state_name == "THINKING") {
            g_sign_translate_runtime.backend_ready = true;
            g_sign_translate_runtime.backend_thinking = true;
            g_sign_translate_runtime.thinking_transition_time = monotonic_time_sec();
            g_sign_translate_runtime.status_text = "...";
        } else if (state_name == "IDLE") {
            g_sign_translate_runtime.backend_ready = true;
            g_sign_translate_runtime.backend_thinking = false;
            g_sign_translate_runtime.thinking_transition_time = monotonic_time_sec();
            g_sign_translate_runtime.status_text = "等待手语输入";
        }
        g_sign_translate_runtime.cv.notify_all();
    } else if (line.rfind("UNKNOWN\t", 0) == 0) {
        float confidence = 0.0f;
        try {
            confidence = std::stof(line.substr(8));
        } catch (...) {
            confidence = 0.0f;
        }
        std::cout << ">>> [SIGN] unknown, confidence=" << confidence << "\n";

        std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
        g_sign_translate_runtime.display_text.clear();
        g_sign_translate_runtime.display_confidence = confidence;
        g_sign_translate_runtime.display_start_time = -1000.0f;
        g_sign_translate_runtime.backend_thinking = false;
        g_sign_translate_runtime.thinking_transition_time = monotonic_time_sec();
        g_sign_translate_runtime.status_text = "等待手语输入";
        g_sign_translate_runtime.cv.notify_all();
    } else if (line.rfind("RESULT\t", 0) == 0) {
        const size_t first_tab = line.find('\t');
        const size_t second_tab = line.find('\t', first_tab + 1);
        if (second_tab == std::string::npos) return;

        const std::string label = line.substr(first_tab + 1, second_tab - first_tab - 1);
        float confidence = 0.0f;
        try {
            confidence = std::stof(line.substr(second_tab + 1));
        } catch (...) {
            confidence = 0.0f;
        }
        std::cout << ">>> [SIGN] result: " << label << " (confidence=" << confidence << ")\n";

        std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
        g_sign_translate_runtime.display_text = label;
        g_sign_translate_runtime.display_confidence = confidence;
        g_sign_translate_runtime.display_start_time = monotonic_time_sec();
        g_sign_translate_runtime.backend_thinking = false;
        g_sign_translate_runtime.thinking_transition_time = g_sign_translate_runtime.display_start_time;
        g_sign_translate_runtime.status_text = "等待手语输入";
        g_sign_translate_runtime.cv.notify_all();
        queue_sign_translate_tts(label);
    }
}

void handle_speech_to_text_output_line(const std::string& line) {
    const std::string without_ansi = strip_ansi_sequences(line);
    const std::string cleaned = strip_ascii_control_chars(without_ansi);
    const std::string trimmed = normalize_tts_text(cleaned);
    if (trimmed.empty()) return;
    if (!is_valid_utf8(trimmed)) return;

    std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
    if (trimmed.find("Started! Please speak") != std::string::npos ||
        trimmed.find("Recording started!") != std::string::npos ||
        trimmed.find("Use recording device:") != std::string::npos ||
        trimmed.find("Current sample rate:") != std::string::npos) {
        g_speech_to_text_runtime.backend_ready = true;
        g_speech_to_text_runtime.status_text = "LISTENING";
        g_speech_to_text_runtime.cv.notify_all();
        return;
    }

    if (trimmed[0] == '{') return;
    if (trimmed.rfind("/home/", 0) == 0) return;
    if (trimmed.find("OnlineRecognizerConfig(") != std::string::npos) return;
    if (trimmed.find("Number of threads:") != std::string::npos) return;
    if (trimmed.find("Real time factor") != std::string::npos) return;
    if (trimmed.find("Start to create recognizer") != std::string::npos) return;
    if (trimmed.find("Recognizer created in") != std::string::npos) return;
    if (trimmed.find("Use recording device:") != std::string::npos) return;
    if (trimmed.find("Current sample rate:") != std::string::npos) return;
    if (trimmed.find("Recording started!") != std::string::npos) return;
    if (trimmed.find("Started! Please speak") != std::string::npos) return;
    if (trimmed.find("/k2-fsa/") != std::string::npos) return;
    if (trimmed.find("provider=") != std::string::npos) return;
    if (trimmed.find("tokens") != std::string::npos && trimmed.find("Online") != std::string::npos) return;

    size_t prefix_start = std::string::npos;
    size_t colon_pos = std::string::npos;
    for (size_t i = 0; i + 1 < trimmed.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(trimmed[i]))) continue;
        size_t j = i;
        while (j < trimmed.size() && std::isdigit(static_cast<unsigned char>(trimmed[j]))) {
            ++j;
        }
        if (j < trimmed.size() && trimmed[j] == ':') {
            prefix_start = i;
            colon_pos = j;
            break;
        }
        i = j;
    }
    if (prefix_start == std::string::npos || colon_pos == std::string::npos) return;

    int segment_id = -1;
    try {
        segment_id = std::stoi(trimmed.substr(prefix_start, colon_pos - prefix_start));
    } catch (...) {
        return;
    }

    const std::string recognized = normalize_tts_text(trimmed.substr(colon_pos + 1));
    if (recognized.empty()) return;
    if (segment_id < g_speech_to_text_runtime.last_segment_id) return;
    if (segment_id == g_speech_to_text_runtime.last_segment_id &&
        recognized == g_speech_to_text_runtime.display_text) {
        return;
    }

    g_speech_to_text_runtime.backend_ready = true;
    g_speech_to_text_runtime.last_segment_id = segment_id;
    g_speech_to_text_runtime.display_text = recognized;
    g_speech_to_text_runtime.display_start_time = monotonic_time_sec();
    g_speech_to_text_runtime.status_text = "LISTENING";
    g_speech_to_text_runtime.cv.notify_all();
    std::cout << ">>> [STT] result[" << segment_id << "]: " << recognized << "\n";
}

void close_speech_to_text_output_locked() {
    if (g_speech_to_text_runtime.output_fd >= 0) {
        close(g_speech_to_text_runtime.output_fd);
        g_speech_to_text_runtime.output_fd = -1;
    }
}

void sign_translate_supervisor_thread() {
    const char* bin_env = std::getenv("SIGN_TRANSLATE_BIN");
    const char* model_env = std::getenv("SIGN_TRANSLATE_MODEL");
    const std::string executable_dir = resolve_executable_dir();

    const std::string binary_path = bin_env && *bin_env
        ? std::string(bin_env)
        : first_existing_path({
            executable_dir + "/ultraleap_rknn_live",
            executable_dir + "/build/ultraleap_rknn_live",
            executable_dir + "/ultraleap_rknn_live/ultraleap_rknn_live",
            executable_dir + "/ultraleap_rknn_live/build/ultraleap_rknn_live",
            "/home/elf/Desktop/ultraleap_rknn_live/ultraleap_rknn_live",
            "/home/elf/Desktop/ultraleap_rknn_live/build/ultraleap_rknn_live"
        });

    const std::string model_path = model_env && *model_env
        ? std::string(model_env)
        : first_existing_path({
            executable_dir + "/model/sign_rk3588.rknn",
            "/home/elf/Desktop/sign_language/rk3588_deploy/model/sign_rk3588.rknn"
        });

    std::string pending;
    pid_t attached_pid = -1;
    bool sign_enabled_sent = false;

    while (true) {
        bool should_run = false;
        bool should_capture = false;
        pid_t child_pid = -1;
        int output_fd = -1;
        {
            std::unique_lock<std::mutex> lock(g_sign_translate_runtime.mtx);
            g_sign_translate_runtime.cv.wait_for(lock, std::chrono::milliseconds(50), [] {
                return g_sign_translate_runtime.shutdown ||
                       g_sign_translate_runtime.app_running ||
                       g_sign_translate_runtime.background_capture_enabled ||
                       g_sign_translate_runtime.service_pid > 0;
            });
            if (g_sign_translate_runtime.shutdown) break;
            should_run = g_sign_translate_runtime.app_running;
            should_capture = g_sign_translate_runtime.background_capture_enabled;
            child_pid = g_sign_translate_runtime.service_pid;
            output_fd = g_sign_translate_runtime.output_fd;
        }

        if ((should_run || should_capture) && child_pid <= 0) {
            if (binary_path.empty()) {
                std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
                g_sign_translate_runtime.status_text = "未找到手语识别程序或模型";
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            int pipefd[2] = {-1, -1};
            int input_pipefd[2] = {-1, -1};
            if (pipe(pipefd) != 0 || pipe(input_pipefd) != 0) {
                if (pipefd[0] >= 0) close(pipefd[0]);
                if (pipefd[1] >= 0) close(pipefd[1]);
                if (input_pipefd[0] >= 0) close(input_pipefd[0]);
                if (input_pipefd[1] >= 0) close(input_pipefd[1]);
                std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
                g_sign_translate_runtime.status_text = "无法创建识别管道";
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            pid_t pid = fork();
            if (pid == 0) {
                setpgid(0, 0);
                char* argv_exec[16];
                int argi = 0;
                argv_exec[argi++] = const_cast<char*>(binary_path.c_str());
                argv_exec[argi++] = const_cast<char*>("--ar-output");
                argv_exec[argi++] = const_cast<char*>("--result-only");
                argv_exec[argi++] = const_cast<char*>("--hand-output");
                argv_exec[argi++] = const_cast<char*>("--defer-infer");
                if (!model_path.empty()) {
                    argv_exec[argi++] = const_cast<char*>("--model");
                    argv_exec[argi++] = const_cast<char*>(model_path.c_str());
                }
                argv_exec[argi++] = const_cast<char*>("--min-frames");
                argv_exec[argi++] = const_cast<char*>("15");
                argv_exec[argi++] = const_cast<char*>("--max-frames");
                argv_exec[argi++] = const_cast<char*>("900");
                argv_exec[argi++] = nullptr;

                close(pipefd[0]);
                close(input_pipefd[1]);
                dup2(pipefd[1], STDOUT_FILENO);
                dup2(input_pipefd[0], STDIN_FILENO);
                int devnull = open("/dev/null", O_WRONLY);
                if (devnull >= 0) {
                    dup2(devnull, STDERR_FILENO);
                    close(devnull);
                }
                close(pipefd[1]);
                close(input_pipefd[0]);
                execv(binary_path.c_str(), argv_exec);
                _exit(127);
            }

            close(pipefd[1]);
            close(input_pipefd[0]);
            if (pid > 0) {
                int flags = fcntl(pipefd[0], F_GETFL, 0);
                if (flags >= 0) {
                    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);
                }
                flags = fcntl(input_pipefd[1], F_GETFL, 0);
                if (flags >= 0) {
                    fcntl(input_pipefd[1], F_SETFL, flags | O_NONBLOCK);
                }

                std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
                g_sign_translate_runtime.service_pid = pid;
                g_sign_translate_runtime.output_fd = pipefd[0];
                g_sign_translate_runtime.input_fd = input_pipefd[1];
                g_sign_translate_runtime.backend_ready = false;
                g_sign_translate_runtime.backend_thinking = false;
                g_sign_translate_runtime.status_text = "等待手语输入";
                pending.clear();
                attached_pid = pid;
                sign_enabled_sent = false;
                g_sign_translate_runtime.cv.notify_all();
            } else {
                close(pipefd[0]);
                close(input_pipefd[1]);
                std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
                g_sign_translate_runtime.status_text = "无法启动手语识别";
                g_sign_translate_runtime.cv.notify_all();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (!should_run && !should_capture && child_pid > 0) {
            stop_child_process(child_pid);
            {
                std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
                g_sign_translate_runtime.service_pid = -1;
                close_sign_translate_output_locked();
                g_sign_translate_runtime.backend_ready = false;
                g_sign_translate_runtime.backend_thinking = false;
                g_sign_translate_runtime.status_text = "等待手语输入";
                g_sign_translate_runtime.cv.notify_all();
            }
            pending.clear();
            attached_pid = -1;
            sign_enabled_sent = false;
            continue;
        }

        if (child_pid > 0 && output_fd >= 0) {
            if (should_run != sign_enabled_sent) {
                send_sign_translate_command(should_run ? "SIGN_ON" : "SIGN_OFF");
                sign_enabled_sent = should_run;
            }

            if (attached_pid != child_pid) {
                pending.clear();
                attached_pid = child_pid;
            }

            char buffer[1024];
            while (true) {
                ssize_t bytes_read = read(output_fd, buffer, sizeof(buffer));
                if (bytes_read > 0) {
                    pending.append(buffer, static_cast<size_t>(bytes_read));
                    size_t newline_pos = 0;
                    while ((newline_pos = pending.find('\n')) != std::string::npos) {
                        std::string frame = pending.substr(0, newline_pos);
                        pending.erase(0, newline_pos + 1);
                        if (!frame.empty() && frame.back() == '\r') frame.pop_back();
                        if (frame.empty()) continue;
                        handle_sign_translate_output_line(frame);
                    }
                    continue;
                }

                if (bytes_read == 0) {
                    break;
                }

                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break;
                }
                break;
            }

            // 移除手部数据捕获超时检查（已不再需要）

            int status = 0;
            pid_t wait_result = waitpid(child_pid, &status, WNOHANG);
            if (wait_result == child_pid) {
                std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
                g_sign_translate_runtime.service_pid = -1;
                close_sign_translate_output_locked();
                g_sign_translate_runtime.backend_ready = false;
                g_sign_translate_runtime.backend_thinking = false;
                if (g_sign_translate_runtime.app_running) {
                    g_sign_translate_runtime.status_text = "识别程序已退出，正在重启";
                } else {
                    g_sign_translate_runtime.status_text = "等待手语输入";
                }
                pending.clear();
                attached_pid = -1;
                sign_enabled_sent = false;
                g_sign_translate_runtime.cv.notify_all();
            }
        }

        // 移除手部数据捕获超时检查（已不再需要）
    }

    pid_t child_pid = -1;
    {
        std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
        child_pid = g_sign_translate_runtime.service_pid;
        g_sign_translate_runtime.service_pid = -1;
        close_sign_translate_output_locked();
        g_sign_translate_runtime.backend_ready = false;
        g_sign_translate_runtime.backend_thinking = false;
        g_sign_translate_runtime.cv.notify_all();
    }
    if (child_pid > 0) {
        stop_child_process(child_pid);
    }
}

bool is_app_state(SysState state) {
    return state == SysState::APP_RADAR ||
           state == SysState::APP_SIGN_TRANSLATE ||
           state == SysState::APP_REAR_CAMERA ||
           state == SysState::APP_SPEECH_TO_TEXT ||
           state == SysState::APP_HEARING_ASSIST ||
           state == SysState::APP_PLACEHOLDER ||
           state == SysState::APP_NAVIGATION ||
           state == SysState::APP_RIDING_MODE;
}

void enter_selected_app_locked(const char* input_source) {
    const int val = g_selected_app.load();
    if (val == 0) {
        set_radar_app_running(true);
        g_target_state = SysState::APP_RADAR;
    } else if (val == 1) {
        set_sign_translate_app_running(false);
        set_sign_translate_background_capture(false);
        wait_sign_translate_state(false, 1500);
        set_sign_translate_background_capture(true);
        set_sign_translate_app_running(true);
        g_target_state = SysState::APP_SIGN_TRANSLATE;
        if (!wait_sign_translate_state(true, 8000)) {
            std::cerr << ">>> [" << input_source << "] sign translate backend start timeout.\n";
        }
    } else if (val == 2) {
        set_rear_camera_app_running(true);
        g_target_state = SysState::APP_REAR_CAMERA;
    } else if (val == 3) {
        g_target_state = SysState::APP_NAVIGATION;
    } else if (val == 4) {
        // 骑行模式：启动导航、后视镜、声音雷达
        set_radar_app_running(true);
        set_rear_camera_app_running(true);
        g_target_state = SysState::APP_RIDING_MODE;
        std::cout << ">>> [" << input_source << "] entering RIDING MODE (navigation + rear camera + radar)\n";
    } else if (val == 5) {
        set_sign_translate_app_running(false);
        set_sign_translate_background_capture(false);
        wait_sign_translate_state(false, 1500);
        set_speech_to_text_app_running(false);
        wait_speech_to_text_state(false, 1500);
        set_sign_translate_background_capture(true);
        set_sign_translate_app_running(true);
        set_speech_to_text_app_running(true);
        g_target_state = SysState::APP_HEARING_ASSIST;
        if (!wait_sign_translate_state(true, 8000)) {
            std::cerr << ">>> [" << input_source << "] hearing assist sign backend start timeout.\n";
        }
        if (!wait_speech_to_text_state(true, 8000)) {
            std::cerr << ">>> [" << input_source << "] hearing assist speech backend start timeout.\n";
        }
    } else if (val == 6) {
        set_speech_to_text_app_running(false);
        wait_speech_to_text_state(false, 1500);
        set_speech_to_text_app_running(true);
        g_target_state = SysState::APP_SPEECH_TO_TEXT;
        if (!wait_speech_to_text_state(true, 8000)) {
            std::cerr << ">>> [" << input_source << "] speech to text backend start timeout.\n";
        }
    } else {
        g_target_state = SysState::APP_PLACEHOLDER;
    }
    std::cout << ">>> [" << input_source << "] entering app [" << val << "] ...\n";
}

void return_to_desktop_locked(SysState state, const char* input_source) {
    if (state == SysState::APP_RADAR) {
        set_radar_app_running(false);
        if (!wait_radar_service_state(false, 4000)) {
            std::cerr << ">>> [" << input_source << "] radar backend stop timeout.\n";
        }
    }
    if (state == SysState::APP_SIGN_TRANSLATE) {
        set_sign_translate_app_running(false);
        set_sign_translate_background_capture(false);
        if (!wait_sign_translate_state(false, 8000)) {
            std::cerr << ">>> [" << input_source << "] sign translate backend stop timeout.\n";
        }
    }
    if (state == SysState::APP_SPEECH_TO_TEXT) {
        set_speech_to_text_app_running(false);
        if (!wait_speech_to_text_state(false, 8000)) {
            std::cerr << ">>> [" << input_source << "] speech to text backend stop timeout.\n";
        }
    }
    if (state == SysState::APP_HEARING_ASSIST) {
        set_sign_translate_app_running(false);
        set_sign_translate_background_capture(false);
        set_speech_to_text_app_running(false);
        if (!wait_sign_translate_state(false, 8000)) {
            std::cerr << ">>> [" << input_source << "] hearing assist sign backend stop timeout.\n";
        }
        if (!wait_speech_to_text_state(false, 8000)) {
            std::cerr << ">>> [" << input_source << "] hearing assist speech backend stop timeout.\n";
        }
        std::cout << ">>> [" << input_source << "] exiting HEARING ASSIST MODE\n";
    }
    if (state == SysState::APP_REAR_CAMERA) set_rear_camera_app_running(false);
    if (state == SysState::APP_RIDING_MODE) {
        // 骑行模式退出：关闭所有子系统
        set_radar_app_running(false);
        set_rear_camera_app_running(false);
        if (!wait_radar_service_state(false, 4000)) {
            std::cerr << ">>> [" << input_source << "] radar backend stop timeout.\n";
        }
        std::cout << ">>> [" << input_source << "] exiting RIDING MODE\n";
    }
    g_target_state = SysState::DESKTOP;
    std::cout << ">>> [" << input_source << "] return to desktop.\n";
}

void dispatch_input_action(InputAction action, const char* input_source) {
    std::lock_guard<std::mutex> lock(g_input_action_mutex);
    const SysState state = g_target_state.load();

    if (state == SysState::STANDBY) {
        if (action == InputAction::ENTER || action == InputAction::TOGGLE_ENTER_BACK) {
            set_sign_translate_background_capture(true);
            g_target_state = SysState::BOOT_PLAY;
            std::cout << ">>> [" << input_source << "] standby released, entering boot sequence.\n";
        }
        return;
    }

    if (state == SysState::DESKTOP) {
        if (g_desktop_tuning_mode.load(std::memory_order_relaxed)) {
            if (action == InputAction::LEFT) {
                adjust_symmetric_shear(-kSymmetricShearAdjustStep, input_source);
            } else if (action == InputAction::RIGHT) {
                adjust_symmetric_shear(kSymmetricShearAdjustStep, input_source);
            }
            return;
        }

        if (action == InputAction::LEFT) {
            const int val = g_selected_app.load();
            if (val > 0) g_selected_app.store(val - 1);
        } else if (action == InputAction::RIGHT) {
            const int val = g_selected_app.load();
            if (val < TOTAL_APPS - 1) g_selected_app.store(val + 1);
        } else if (action == InputAction::ENTER || action == InputAction::TOGGLE_ENTER_BACK) {
            enter_selected_app_locked(input_source);
        }
        return;
    }

    if (is_app_state(state) &&
        (action == InputAction::BACK || action == InputAction::TOGGLE_ENTER_BACK)) {
        return_to_desktop_locked(state, input_source);
    }
}

int open_button_serial_port() {
    int fd = open(kButtonSerialDevice, O_RDWR | O_NOCTTY);
    if (fd < 0) return -1;

    struct termios tty {};
    if (tcgetattr(fd, &tty) != 0) {
        close(fd);
        return -1;
    }

    cfsetospeed(&tty, kButtonSerialBaud);
    cfsetispeed(&tty, kButtonSerialBaud);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~OPOST;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 10;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        close(fd);
        return -1;
    }

    tcflush(fd, TCIFLUSH);
    return fd;
}

void handle_uart_button_byte(uint8_t byte) {
    switch (byte) {
        case 0x04:
            dispatch_input_action(InputAction::LEFT, "UART");
            break;
        case 0x02:
            dispatch_input_action(InputAction::RIGHT, "UART");
            break;
        case 0x05:
            dispatch_input_action(InputAction::TOGGLE_ENTER_BACK, "UART");
            break;
        default:
            break;
    }
}

void uart_button_input_thread() {
    bool announced_ready = false;
    bool reported_open_error = false;

    while (true) {
        int fd = open_button_serial_port();
        if (fd < 0) {
            if (!reported_open_error) {
                std::cerr << ">>> UART button listener failed to open " << kButtonSerialDevice
                          << ", retrying...\n";
                reported_open_error = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kButtonSerialRetryDelayMs));
            continue;
        }

        if (!announced_ready) {
            std::cout << ">>> UART button listener ready on " << kButtonSerialDevice << "\n";
            announced_ready = true;
        } else if (reported_open_error) {
            std::cout << ">>> UART button listener reconnected on " << kButtonSerialDevice << "\n";
        }
        reported_open_error = false;

        unsigned char buf[256];
        while (true) {
            errno = 0;
            const ssize_t bytes_read = read(fd, buf, sizeof(buf));
            if (bytes_read > 0) {
                for (ssize_t i = 0; i < bytes_read; ++i) {
                    handle_uart_button_byte(buf[i]);
                }
                continue;
            }

            if (bytes_read == 0) continue;
            if (errno == EINTR) continue;

            std::cerr << ">>> UART button listener read failed on " << kButtonSerialDevice
                      << ", reconnecting...\n";
            break;
        }

        close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(kButtonSerialRetryDelayMs));
    }
}

void upsert_radar_detection(const RadarDetection& detection) {
    std::lock_guard<std::mutex> lock(g_radar_runtime.mtx);

    for (auto& item : g_radar_runtime.targets) {
        if (!item.active) continue;
        if (item.type == detection.type && angle_distance_deg(item.target_angle_deg, detection.angle_deg) <= kRadarDuplicateAngleDeg) {
            item.type = detection.type;
            item.target_angle_deg = detection.angle_deg;
            item.confidence = detection.confidence;
            item.shown_confidence = std::max(item.shown_confidence, detection.confidence);
            item.last_update_time = detection.timestamp;
            item.active = true;
            return;
        }
    }

    RadarUiTarget target;
    target.type = detection.type;
    target.target_angle_deg = detection.angle_deg;
    target.display_angle_deg = detection.angle_deg;
    target.confidence = detection.confidence;
    target.shown_confidence = detection.confidence;
    target.last_update_time = detection.timestamp;
    target.active = true;
    g_radar_runtime.targets.push_back(target);

    g_radar_runtime.targets.erase(
        std::remove_if(
            g_radar_runtime.targets.begin(),
            g_radar_runtime.targets.end(),
            [detection](const RadarUiTarget& item) {
                return !item.active || (detection.timestamp - item.last_update_time) > kRadarLifetimeSec;
            }
        ),
        g_radar_runtime.targets.end()
    );
    if (g_radar_runtime.targets.size() > 18) {
        g_radar_runtime.targets.erase(g_radar_runtime.targets.begin(), g_radar_runtime.targets.begin() + (g_radar_runtime.targets.size() - 18));
    }
}

void gps_to_local(double lng, double lat, float& x, float& y) {
    double rad_lat = g_nav.origin_lat * M_PI / 180.0;
    x = static_cast<float>((lng - g_nav.origin_lng) * 111320.0 * std::cos(rad_lat));
    y = static_cast<float>((lat - g_nav.origin_lat) * 111320.0);
}

void apply_navigation_payload(const json& payload) {
    std::lock_guard<std::mutex> lock(g_nav.mtx);
    float nav_now = monotonic_time_sec();

    g_nav.last_packet_time = nav_now;

    std::string status = json_to_string(payload, "status", "standby");
    if (status == "standby") {
        if (g_nav.state != "STANDBY") {
            g_nav.state = "STANDBY";
            g_nav.state_timer = nav_now;
        }
        g_nav.has_origin = false;
        g_nav.route_pts.clear();
        g_nav.last_path_str.clear();
        g_nav.current_dist = 0.0f;
        g_nav.display_current_dist = 0.0f;
        g_nav.total_dist = 1.0f;
        g_nav.dist_to_turn = 9999.0f;
        g_nav.display_dist_to_turn = 9999.0f;
        g_nav.action_str.clear();
        g_nav.dest_name.clear();
        return;
    }

    std::string path_str = json_to_string(payload, "full_path", "");
    if (!path_str.empty() && path_str != g_nav.last_path_str) {
        g_nav.last_path_str = path_str;
        g_nav.route_pts.clear();
        g_nav.has_origin = false;

        std::stringstream ss(path_str);
        std::string pt_str;
        while (std::getline(ss, pt_str, ';')) {
            size_t comma = pt_str.find(',');
            if (comma == std::string::npos) continue;

            try {
                double lng = std::stod(pt_str.substr(0, comma));
                double lat = std::stod(pt_str.substr(comma + 1));
                if (!g_nav.has_origin) {
                    g_nav.origin_lng = lng;
                    g_nav.origin_lat = lat;
                    g_nav.has_origin = true;
                }

                float lx = 0.0f, ly = 0.0f;
                gps_to_local(lng, lat, lx, ly);
                g_nav.route_pts.push_back({lx, ly});
            } catch (...) {
            }
        }
    }

    if (g_nav.route_pts.size() < 2) return;

    double cur_lng = json_to_double(payload, "cur_lng", 0.0);
    double cur_lat = json_to_double(payload, "cur_lat", 0.0);
    gps_to_local(cur_lng, cur_lat, g_nav.car_x, g_nav.car_y);

    g_nav.total_dist = static_cast<float>(std::max(1.0, json_to_double(payload, "total_dist", 1.0)));
    float remain = static_cast<float>(std::max(0.0, json_to_double(payload, "remain_dist", 0.0)));
    g_nav.current_dist = std::max(0.0f, g_nav.total_dist - remain);
    g_nav.dist_to_turn = static_cast<float>(std::max(0.0, json_to_double(payload, "next_dist", 9999.0)));
    g_nav.action_str = json_to_string(payload, "next_action", "PROCEED");
    g_nav.dest_name = json_to_string(payload, "dest_name", "DESTINATION");

    if (g_nav.action_str.find("靠左") != std::string::npos) g_nav.target_turn_type = -0.4f;
    else if (g_nav.action_str.find("左转") != std::string::npos) g_nav.target_turn_type = -1.0f;
    else if (g_nav.action_str.find("靠右") != std::string::npos) g_nav.target_turn_type = 0.4f;
    else if (g_nav.action_str.find("右转") != std::string::npos) g_nav.target_turn_type = 1.0f;
    else if (g_nav.action_str.find("调头") != std::string::npos || g_nav.action_str.find("掉头") != std::string::npos) g_nav.target_turn_type = -1.0f;
    else g_nav.target_turn_type = 0.0f;

    float min_d = 1e9f;
    size_t best_idx = 0;
    for (size_t i = 0; i + 1 < g_nav.route_pts.size(); ++i) {
        float mid_x = (g_nav.route_pts[i].x + g_nav.route_pts[i + 1].x) / 2.0f;
        float mid_y = (g_nav.route_pts[i].y + g_nav.route_pts[i + 1].y) / 2.0f;
        float d = std::hypot(mid_x - g_nav.car_x, mid_y - g_nav.car_y);
        if (d < min_d) {
            min_d = d;
            best_idx = i;
        }
    }

    if (best_idx + 1 < g_nav.route_pts.size()) {
        g_nav.raw_heading = std::atan2(
            g_nav.route_pts[best_idx + 1].y - g_nav.route_pts[best_idx].y,
            g_nav.route_pts[best_idx + 1].x - g_nav.route_pts[best_idx].x);
    }

    if (g_nav.state == "STANDBY") {
        g_nav.state = "TRANSITION_IN";
        g_nav.state_timer = nav_now;
        g_nav.smooth_heading = g_nav.raw_heading;
        g_nav.display_car_x = g_nav.car_x;
        g_nav.display_car_y = g_nav.car_y;
        g_nav.display_current_dist = g_nav.current_dist;
        g_nav.display_dist_to_turn = g_nav.dist_to_turn;
    }

    if (g_nav.state == "NAVIGATING" && remain < 15.0f) {
        g_nav.state = "ARRIVED";
        g_nav.state_timer = nav_now;
    }
}

sdp_session_t* register_navigation_sdp_service(uint8_t channel) {
    auto* record = sdp_record_alloc();
    if (!record) return nullptr;

    uuid_t svc_uuid;
    uuid_t root_uuid;
    uuid_t l2cap_uuid;
    uuid_t rfcomm_uuid;
    sdp_list_t* svc_class_list = nullptr;
    sdp_list_t* root_list = nullptr;
    sdp_list_t* l2cap_list = nullptr;
    sdp_list_t* rfcomm_list = nullptr;
    sdp_list_t* proto_list = nullptr;
    sdp_list_t* access_proto_list = nullptr;
    sdp_list_t* profile_list = nullptr;
    sdp_data_t* channel_data = nullptr;

    sdp_uuid16_create(&svc_uuid, SERIAL_PORT_SVCLASS_ID);
    sdp_set_service_id(record, svc_uuid);
    svc_class_list = sdp_list_append(nullptr, &svc_uuid);
    sdp_set_service_classes(record, svc_class_list);

    sdp_uuid16_create(&root_uuid, PUBLIC_BROWSE_GROUP);
    root_list = sdp_list_append(nullptr, &root_uuid);
    sdp_set_browse_groups(record, root_list);

    sdp_uuid16_create(&l2cap_uuid, L2CAP_UUID);
    l2cap_list = sdp_list_append(nullptr, &l2cap_uuid);
    proto_list = sdp_list_append(nullptr, l2cap_list);

    sdp_uuid16_create(&rfcomm_uuid, RFCOMM_UUID);
    channel_data = sdp_data_alloc(SDP_UINT8, &channel);
    rfcomm_list = sdp_list_append(nullptr, &rfcomm_uuid);
    sdp_list_append(rfcomm_list, channel_data);
    proto_list = sdp_list_append(proto_list, rfcomm_list);

    access_proto_list = sdp_list_append(nullptr, proto_list);
    sdp_set_access_protos(record, access_proto_list);

    sdp_profile_desc_t profile{};
    sdp_uuid16_create(&profile.uuid, SERIAL_PORT_PROFILE_ID);
    profile.version = 0x0100;
    profile_list = sdp_list_append(nullptr, &profile);
    sdp_set_profile_descs(record, profile_list);

    sdp_set_info_attr(record, "ELF2 Navigation", "Aura", "Bluetooth navigation receiver");

    bdaddr_t src_bdaddr = {{0, 0, 0, 0, 0, 0}};
    bdaddr_t dst_bdaddr = {{0, 0, 0, 0xff, 0xff, 0xff}};
    sdp_session_t* session = sdp_connect(&src_bdaddr, &dst_bdaddr, SDP_RETRY_IF_BUSY);
    if (!session) {
        if (channel_data) sdp_data_free(channel_data);
        if (profile_list) sdp_list_free(profile_list, nullptr);
        if (access_proto_list) sdp_list_free(access_proto_list, nullptr);
        if (proto_list) sdp_list_free(proto_list, nullptr);
        if (rfcomm_list) sdp_list_free(rfcomm_list, nullptr);
        if (l2cap_list) sdp_list_free(l2cap_list, nullptr);
        if (root_list) sdp_list_free(root_list, nullptr);
        if (svc_class_list) sdp_list_free(svc_class_list, nullptr);
        sdp_record_free(record);
        return nullptr;
    }

    if (sdp_record_register(session, record, 0) < 0) {
        sdp_close(session);
        if (channel_data) sdp_data_free(channel_data);
        if (profile_list) sdp_list_free(profile_list, nullptr);
        if (access_proto_list) sdp_list_free(access_proto_list, nullptr);
        if (proto_list) sdp_list_free(proto_list, nullptr);
        if (rfcomm_list) sdp_list_free(rfcomm_list, nullptr);
        if (l2cap_list) sdp_list_free(l2cap_list, nullptr);
        if (root_list) sdp_list_free(root_list, nullptr);
        if (svc_class_list) sdp_list_free(svc_class_list, nullptr);
        sdp_record_free(record);
        return nullptr;
    }

    if (channel_data) sdp_data_free(channel_data);
    if (profile_list) sdp_list_free(profile_list, nullptr);
    if (access_proto_list) sdp_list_free(access_proto_list, nullptr);
    if (proto_list) sdp_list_free(proto_list, nullptr);
    if (rfcomm_list) sdp_list_free(rfcomm_list, nullptr);
    if (l2cap_list) sdp_list_free(l2cap_list, nullptr);
    if (root_list) sdp_list_free(root_list, nullptr);
    if (svc_class_list) sdp_list_free(svc_class_list, nullptr);
    return session;
}

void bluetooth_navigation_thread() {
    int server_sock = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM);
    if (server_sock < 0) {
        std::cerr << ">>> Bluetooth navigation socket init failed.\n";
        return;
    }

    sockaddr_rc local_addr{};
    local_addr.rc_family = AF_BLUETOOTH;
    std::memset(&local_addr.rc_bdaddr, 0, sizeof(local_addr.rc_bdaddr));
    local_addr.rc_channel = kBluetoothNavChannel;

    if (bind(server_sock, reinterpret_cast<struct sockaddr*>(&local_addr), sizeof(local_addr)) < 0) {
        std::cerr << ">>> Bluetooth navigation bind failed on RFCOMM channel " << static_cast<int>(kBluetoothNavChannel) << ".\n";
        close(server_sock);
        return;
    }

    if (listen(server_sock, 1) < 0) {
        std::cerr << ">>> Bluetooth navigation listen failed.\n";
        close(server_sock);
        return;
    }

    sdp_session_t* sdp_session = register_navigation_sdp_service(kBluetoothNavChannel);
    if (!sdp_session) {
        std::cerr << ">>> SDP registration failed. Android pairing may connect only if channel 1 is forced manually.\n";
    }

    std::cout << ">>> ELF2 navigation Bluetooth RFCOMM ready (channel "
              << static_cast<int>(kBluetoothNavChannel) << ").\n";

    while (true) {
        sockaddr_rc remote_addr{};
        socklen_t addr_len = sizeof(remote_addr);
        int client_sock = accept(server_sock, reinterpret_cast<struct sockaddr*>(&remote_addr), &addr_len);
        if (client_sock < 0) continue;

        char remote_mac[18] = {0};
        ba2str(&remote_addr.rc_bdaddr, remote_mac);
        std::cout << ">>> Bluetooth nav client connected: " << remote_mac << "\n";

        std::string pending;
        char buffer[4096];

        while (true) {
            ssize_t bytes_read = read(client_sock, buffer, sizeof(buffer));
            if (bytes_read <= 0) break;

            pending.append(buffer, static_cast<size_t>(bytes_read));
            size_t newline_pos = 0;
            while ((newline_pos = pending.find('\n')) != std::string::npos) {
                std::string frame = pending.substr(0, newline_pos);
                pending.erase(0, newline_pos + 1);
                if (!frame.empty() && frame.back() == '\r') frame.pop_back();
                if (frame.empty()) continue;

                try {
                    json payload = json::parse(frame);
                    apply_navigation_payload(payload);
                } catch (...) {
                }
            }
        }

        close(client_sock);
        std::cout << ">>> Bluetooth nav client disconnected.\n";
    }

    if (sdp_session) sdp_close(sdp_session);
    close(server_sock);
}

void radar_backend_thread() {
    constexpr int kRadarPort = 19091;
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) {
        std::cerr << ">>> Radar socket init failed.\n";
        return;
    }

    int opt = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(kRadarPort);

    if (bind(server_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << ">>> Radar socket bind failed on port " << kRadarPort << ".\n";
        close(server_sock);
        return;
    }

    if (listen(server_sock, 1) < 0) {
        std::cerr << ">>> Radar socket listen failed.\n";
        close(server_sock);
        return;
    }

    std::cout << ">>> Sound radar backend ready on 127.0.0.1:" << kRadarPort << "\n";

    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(server_sock, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_sock < 0) continue;

        std::string pending;
        char buffer[2048];

        while (true) {
            ssize_t bytes_read = read(client_sock, buffer, sizeof(buffer));
            if (bytes_read <= 0) break;

            pending.append(buffer, static_cast<size_t>(bytes_read));
            size_t newline_pos = 0;
            while ((newline_pos = pending.find('\n')) != std::string::npos) {
                std::string frame = pending.substr(0, newline_pos);
                pending.erase(0, newline_pos + 1);
                if (!frame.empty() && frame.back() == '\r') frame.pop_back();
                if (frame.empty()) continue;

                try {
                    json payload = json::parse(frame);
                    RadarSoundType type;
                    if (!radar_type_from_string(json_to_string(payload, "type", ""), type)) continue;

                    RadarDetection detection;
                    detection.type = type;
                    detection.angle_deg = static_cast<float>(json_to_double(payload, "angle", 0.0));
                    detection.confidence = static_cast<float>(json_to_double(payload, "confidence", 0.0));
                    detection.timestamp = monotonic_time_sec();
                    upsert_radar_detection(detection);
                } catch (...) {
                }
            }
        }

        close(client_sock);
    }
}

void radar_service_supervisor_thread() {
    const char* radar_cmd_env = std::getenv("SOUND_RADAR_CMD");
    const std::string executable_dir = resolve_executable_dir();
    const std::string radar_script = first_existing_path({
        executable_dir + "/sound_radar_service.py",
        std::filesystem::current_path().string() + "/sound_radar_service.py",
        "/home/elf/Desktop/sound_radar_service.py",
        "/home/elf/Desktop/ELF2/sound_radar_service.py"
    });

    std::string radar_cmd;
    if (radar_cmd_env && *radar_cmd_env) {
        radar_cmd = std::string(radar_cmd_env);
    } else if (!radar_script.empty()) {
        radar_cmd = "cd \"" + std::filesystem::path(radar_script).parent_path().string() +
                    "\" && SOUND_RADAR_QUIET=1 python3 \"" +
                    std::filesystem::path(radar_script).filename().string() + "\"";
    } else {
        std::cerr << ">>> Sound radar service script not found. "
                  << "Set SOUND_RADAR_CMD or place sound_radar_service.py next to the executable.\n";
    }

    while (true) {
        bool should_run = false;
        pid_t child_pid = -1;
        {
            std::unique_lock<std::mutex> lock(g_radar_runtime.mtx);
            g_radar_runtime.cv.wait(lock, [] {
                return g_radar_runtime.shutdown || g_radar_runtime.app_running || g_radar_runtime.service_pid > 0;
            });
            if (g_radar_runtime.shutdown) break;
            should_run = g_radar_runtime.app_running;
            child_pid = g_radar_runtime.service_pid;
        }

        if (should_run && child_pid <= 0) {
            if (radar_cmd.empty()) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            pid_t pid = fork();
            if (pid == 0) {
                setpgid(0, 0);
                execl("/bin/sh", "sh", "-lc", radar_cmd.c_str(), static_cast<char*>(nullptr));
                _exit(127);
            }
            if (pid > 0) {
                std::lock_guard<std::mutex> lock(g_radar_runtime.mtx);
                g_radar_runtime.service_pid = pid;
                std::cout << ">>> Sound radar service started. pid=" << pid << "\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }

        if (!should_run && child_pid > 0) {
            std::lock_guard<std::mutex> lock(g_radar_runtime.mtx);
            g_radar_runtime.service_pid = -1;
            stop_child_process(child_pid);
            continue;
        }

        if (child_pid > 0) {
            int status = 0;
            pid_t result = waitpid(child_pid, &status, WNOHANG);
            if (result == child_pid) {
                std::lock_guard<std::mutex> lock(g_radar_runtime.mtx);
                g_radar_runtime.service_pid = -1;
                if (g_radar_runtime.app_running) {
                    if (WIFEXITED(status)) {
                        std::cout << ">>> Sound radar service exited with code "
                                  << WEXITSTATUS(status) << ", restarting...\n";
                    } else if (WIFSIGNALED(status)) {
                        std::cout << ">>> Sound radar service killed by signal "
                                  << WTERMSIG(status) << ", restarting...\n";
                    } else {
                        std::cout << ">>> Sound radar service exited, restarting...\n";
                    }
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    pid_t child_pid = -1;
    {
        std::lock_guard<std::mutex> lock(g_radar_runtime.mtx);
        child_pid = g_radar_runtime.service_pid;
        g_radar_runtime.service_pid = -1;
    }
    if (child_pid > 0) {
        stop_child_process(child_pid);
    }
}

void rear_camera_capture_thread() {
    cv::VideoCapture cap;

    while (true) {
        RearCameraSettings settings;
        {
            std::unique_lock<std::mutex> lock(g_rear_camera_runtime.mtx);
            g_rear_camera_runtime.cv.wait(lock, [] {
                return g_rear_camera_runtime.shutdown || g_rear_camera_runtime.app_running;
            });
            if (g_rear_camera_runtime.shutdown) break;
            settings = g_rear_camera_runtime.settings;
        }

        if (!open_rear_camera_capture(settings, cap)) {
            {
                std::lock_guard<std::mutex> lock(g_rear_camera_runtime.mtx);
                g_rear_camera_runtime.source_ready = false;
                g_rear_camera_runtime.open_failed = true;
                g_rear_camera_runtime.frame_ready = false;
                g_rear_camera_runtime.status_text = "CAMERA OPEN FAILED";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(g_rear_camera_runtime.mtx);
            g_rear_camera_runtime.source_ready = true;
            g_rear_camera_runtime.open_failed = false;
            g_rear_camera_runtime.status_text = "LIVE";
        }

        while (true) {
            RearCameraSettings current_settings;
            bool should_run = false;
            {
                std::lock_guard<std::mutex> lock(g_rear_camera_runtime.mtx);
                should_run = g_rear_camera_runtime.app_running;
                current_settings = g_rear_camera_runtime.settings;
            }
            if (!should_run) break;

            cv::Mat frame_bgr;
            if (!cap.read(frame_bgr) || frame_bgr.empty()) {
                {
                    std::lock_guard<std::mutex> lock(g_rear_camera_runtime.mtx);
                    g_rear_camera_runtime.status_text = "SIGNAL LOST";
                    g_rear_camera_runtime.frame_ready = false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            cv::Mat transformed = apply_rear_camera_transform(frame_bgr, current_settings);
            if (transformed.empty()) continue;

            cv::Mat rgba;
            cv::cvtColor(transformed, rgba, cv::COLOR_BGR2RGBA);

            {
                std::lock_guard<std::mutex> lock(g_rear_camera_runtime.mtx);
                g_rear_camera_runtime.latest_rgba = rgba;
                g_rear_camera_runtime.frame_ready = true;
                g_rear_camera_runtime.source_ready = true;
                g_rear_camera_runtime.open_failed = false;
                g_rear_camera_runtime.status_text = "LIVE";
                g_rear_camera_runtime.last_frame_time = monotonic_time_sec();
            }
        }

        cap.release();
        {
            std::lock_guard<std::mutex> lock(g_rear_camera_runtime.mtx);
            g_rear_camera_runtime.source_ready = false;
            if (!g_rear_camera_runtime.app_running) {
                g_rear_camera_runtime.status_text = "CAMERA OFFLINE";
                g_rear_camera_runtime.frame_ready = false;
                g_rear_camera_runtime.latest_rgba.release();
            }
        }
    }

    cap.release();
}

void terminal_input_thread() {
    std::cout << "\n========================================================\n";
    std::cout << "[AR TACTICAL OS - 核心已加载]\n";
    std::cout << "控制说明：\n";
    std::cout << " [A] / [D] : 桌面左右切换 App\n";
    std::cout << " [Y]       : 确认并进入选中 App\n";
    std::cout << " [B]       : 退出 App 返回主桌面\n";
    std::cout << " [T]       : Toggle tuning mode\n";
    std::cout << "========================================================\n\n";
    
    std::string input;
    while (std::cin >> input) {
        if (input == "a" || input == "A") {
            dispatch_input_action(InputAction::LEFT, "STDIN");
            continue;
        }
        if (input == "d" || input == "D") {
            dispatch_input_action(InputAction::RIGHT, "STDIN");
            continue;
        }
        if (input == "y" || input == "Y") {
            dispatch_input_action(InputAction::ENTER, "STDIN");
            continue;
        }
        if (input == "b" || input == "B") {
            dispatch_input_action(InputAction::BACK, "STDIN");
            continue;
        }
        if (input == "t" || input == "T") {
            toggle_desktop_tuning_mode("STDIN");
            continue;
        }

        if (g_target_state.load() == SysState::DESKTOP) {
            if (input == "a" || input == "A") {
                int val = g_selected_app.load();
                if (val > 0) g_selected_app.store(val - 1);
            }
            else if (input == "d" || input == "D") {
                int val = g_selected_app.load();
                if (val < TOTAL_APPS - 1) g_selected_app.store(val + 1);
            }
            else if (input == "y" || input == "Y") {
                int val = g_selected_app.load();
                if (val == 0) {
                    set_radar_app_running(true);
                    g_target_state = SysState::APP_RADAR;
                } else if (val == 1) {
                    set_sign_translate_app_running(true);
                    g_target_state = SysState::APP_SIGN_TRANSLATE;
                } else if (val == 2) {
                    set_rear_camera_app_running(true);
                    g_target_state = SysState::APP_REAR_CAMERA;
                } else if (val == 3) {
                    g_target_state = SysState::APP_NAVIGATION;
                } else if (val == 4) {
                    set_radar_app_running(true);
                    set_rear_camera_app_running(true);
                    g_target_state = SysState::APP_RIDING_MODE;
                } else if (val == 5) {
                    set_sign_translate_app_running(false);
                    set_sign_translate_background_capture(false);
                    wait_sign_translate_state(false, 1500);
                    set_speech_to_text_app_running(false);
                    wait_speech_to_text_state(false, 1500);
                    set_sign_translate_background_capture(true);
                    set_sign_translate_app_running(true);
                    set_speech_to_text_app_running(true);
                    g_target_state = SysState::APP_HEARING_ASSIST;
                } else if (val == 6) {
                    set_speech_to_text_app_running(false);
                    wait_speech_to_text_state(false, 1500);
                    set_speech_to_text_app_running(true);
                    g_target_state = SysState::APP_SPEECH_TO_TEXT;
                } else {
                    g_target_state = SysState::APP_PLACEHOLDER;
                }
                std::cout << ">>> 正在进入应用 [" << val << "] ...\n";
            }
        }
        else if (
            g_target_state.load() == SysState::APP_RADAR ||
            g_target_state.load() == SysState::APP_REAR_CAMERA ||
            g_target_state.load() == SysState::APP_HEARING_ASSIST ||
            g_target_state.load() == SysState::APP_PLACEHOLDER ||
            g_target_state.load() == SysState::APP_NAVIGATION ||
            g_target_state.load() == SysState::APP_RIDING_MODE) {
            if (input == "b" || input == "B") {
                if (g_target_state.load() == SysState::APP_RADAR) set_radar_app_running(false);
                if (g_target_state.load() == SysState::APP_REAR_CAMERA) set_rear_camera_app_running(false);
                if (g_target_state.load() == SysState::APP_HEARING_ASSIST) {
                    set_sign_translate_app_running(false);
                    set_sign_translate_background_capture(false);
                    set_speech_to_text_app_running(false);
                }
                if (g_target_state.load() == SysState::APP_RIDING_MODE) {
                    set_radar_app_running(false);
                    set_rear_camera_app_running(false);
                }
                g_target_state = SysState::DESKTOP;
                std::cout << ">>> 返回主屏幕。\n";
            }
        }
    }
}

// ==========================================
// 核心光学畸变映射
// ==========================================
float evaluatePolynomial(float u, float v, const std::vector<float>& coefs) {
    float result = 0.0f; int idx = 0;
    for (int i = 0; i <= 3; ++i) {       
        for (int j = 0; j <= 3; ++j) {   
            result += coefs[idx++] * std::pow(u, i) * std::pow(v, j);
        }
    }
    return result;
}

constexpr float kDefaultBaselineMm = 64.0f;
constexpr float kDefaultDepthMm = 1200.0f;
constexpr float kDefaultPlaneCenterXMm = 0.0f;
constexpr float kDefaultPlaneCenterYMm = 0.0f;
constexpr bool kDefaultSwapEyes = true;
constexpr float kDefaultSymmetricPlaneAngleDeg = 0.0f;
constexpr float kDefaultPanelWidthMm = 640.0f;
constexpr float kDefaultPanelHeightMm = 360.0f;
constexpr int kGridSize = 240;
constexpr float kUiWidthPx = 1280.0f;
constexpr float kUiHeightPx = 720.0f;
constexpr float kUiPxPerMmX = kUiWidthPx / kDefaultPanelWidthMm;
constexpr float kUiPxPerMmY = kUiHeightPx / kDefaultPanelHeightMm;
constexpr float kRadarCenterU = 0.5f;
constexpr float kRadarCenterV = 0.5f;
constexpr float kRadarRadiusU = 1.12f;
constexpr float kRadarRadiusV = 1.12f * (kDefaultPanelWidthMm / kDefaultPanelHeightMm);
constexpr float kNavigationOuterFramePadPx = 30.0f;
constexpr float kRidingVirtualCanvasScale = 2.5f;
constexpr float kRidingHudPanelScale = 2.5f;
constexpr float kRidingNavigationScale = 0.74f;
constexpr float kRidingNavigationLeftMarginPx = 388.0f;
constexpr float kRidingNavigationTopMarginPx = -18.0f;
constexpr float kRidingRearCameraScale = 0.83f;
constexpr float kRidingRearCameraTopPx = 70.0f;
constexpr float kRidingRadarGapPx = 162.0f;
constexpr float kBootUvScale = 5.0f;

struct EyeRenderAdjustments {
    float plane_angle_deg;
    float panel_shear;
};

float evaluateEyeRayComponent(float local_u, float local_v, const std::vector<float>& coeffs) {
    return evaluatePolynomial(1.0f - local_u, local_v, coeffs);
}

void transformEyeRay(float raw_ray_x, float raw_ray_y, float& ray_x, float& ray_y) {
    ray_x = -raw_ray_y;
    ray_y = raw_ray_x;
}

EyeRenderAdjustments getEyeAdjustments(float eye_origin_x_mm, float symmetric_shear) {
    EyeRenderAdjustments adjustments{
        0.5f * kDefaultSymmetricPlaneAngleDeg,
        0.5f * symmetric_shear,
    };
    if (eye_origin_x_mm > 0.0f) {
        adjustments.plane_angle_deg = -adjustments.plane_angle_deg;
        adjustments.panel_shear = -adjustments.panel_shear;
    }
    return adjustments;
}

void intersectTiltedPlane(
    float ray_x,
    float ray_y,
    float eye_origin_x_mm,
    float plane_center_x_mm,
    float depth_mm,
    float plane_angle_deg,
    float& hit_x_mm,
    float& hit_y_mm) {
    const float angle_rad = plane_angle_deg * static_cast<float>(M_PI / 180.0);
    const float normal_x = std::sin(angle_rad);
    const float normal_z = std::cos(angle_rad);
    const float center_offset_x = plane_center_x_mm - eye_origin_x_mm;
    const float numerator = (normal_x * center_offset_x) + (normal_z * depth_mm);
    const float denominator = (normal_x * ray_x) + normal_z;

    float distance = depth_mm;
    if (std::fabs(denominator) > 1e-6f) {
        distance = numerator / denominator;
    }

    hit_x_mm = eye_origin_x_mm + distance * ray_x;
    hit_y_mm = distance * ray_y;
}

void computePanelUv(
    float hit_x_mm,
    float hit_y_mm,
    float panel_center_x_mm,
    float panel_center_y_mm,
    float panel_width_mm,
    float panel_height_mm,
    float panel_shear,
    float& src_u,
    float& src_v) {
    const float local_x = hit_x_mm - panel_center_x_mm;
    const float local_y = hit_y_mm - panel_center_y_mm;
    const float sheared_y = local_y + (panel_shear * local_x);

    src_u = (local_x + (panel_width_mm * 0.5f)) / panel_width_mm;
    src_v = (sheared_y + (panel_height_mm * 0.5f)) / panel_height_mm;
}

bool loadCalibrationJson(const std::string& filepath, json& calib) {
    const std::vector<std::string> candidates = {
        filepath,
        "calibration.json",
    };

    for (const auto& candidate : candidates) {
        std::ifstream file(candidate);
        if (!file.is_open()) continue;

        try {
            file >> calib;
            return true;
        } catch (...) {
            std::cerr << "[ERROR] Failed to parse calibration JSON: " << candidate << std::endl;
            return false;
        }
    }

    std::cerr << "[ERROR] Failed to open calibration file. Tried:";
    for (const auto& candidate : candidates) {
        std::cerr << " " << candidate;
    }
    std::cerr << std::endl;
    return false;
}


bool generateDistortionMeshes(const std::string& filepath, EyeMeshData& leftMesh, EyeMeshData& rightMesh) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] 找不到畸变校准文件: " << filepath << std::endl;
        return false;
    }
    json calib;
    try { file >> calib; } catch (...) { return false; }
    if (calib.contains("v2CalibrationValues")) calib = calib["v2CalibrationValues"];

    std::vector<float> left_x, left_y, right_x, right_y;
    if (calib.contains("left_uv_to_rect_x")) left_x = calib["left_uv_to_rect_x"].get<std::vector<float>>(); else return false;
    if (calib.contains("left_uv_to_rect_y")) left_y = calib["left_uv_to_rect_y"].get<std::vector<float>>(); else return false;
    if (calib.contains("right_uv_to_rect_x")) right_x = calib["right_uv_to_rect_x"].get<std::vector<float>>(); else return false;
    if (calib.contains("right_uv_to_rect_y")) right_y = calib["right_uv_to_rect_y"].get<std::vector<float>>(); else return false;

    float left_eye_offset[2] = {0.0f, 0.0f}, right_eye_offset[2] = {0.0f, 0.0f};
    if (calib.contains("left_eye_offset")) { left_eye_offset[0] = calib["left_eye_offset"][0].get<float>(); left_eye_offset[1] = calib["left_eye_offset"][1].get<float>(); }
    if (calib.contains("right_eye_offset")) { right_eye_offset[0] = calib["right_eye_offset"][0].get<float>(); right_eye_offset[1] = calib["right_eye_offset"][1].get<float>(); }
    float baseline_m = calib.value("baseline", 64.0f) / 1000.0f;

    const float eye_fov_deg = 70.0f;
    const float aspect = 1280.0f / 720.0f;
    const float center_x = 0.0f, center_y = 0.0f, center_z = 3.0f;
    const float plane_width_m = 2.1f;
    const float plane_height_m = plane_width_m * (720.0f / 1280.0f);
    const float plane_left = center_x - (plane_width_m * 0.5f), plane_top = center_y + (plane_height_m * 0.5f);
    const float tan_half_fov_y = std::tan((eye_fov_deg * M_PI / 180.0f) * 0.5f);
    const float tan_half_fov_x = tan_half_fov_y * aspect;
    const int GRID_SIZE = 240;

    auto buildEyeMesh = [&](EyeMeshData& mesh, const std::vector<float>& coef_x, const std::vector<float>& coef_y, bool isLeftEye) {
        mesh.vertices.clear();
        mesh.indices.clear();
        mesh.vertices.reserve((GRID_SIZE + 1) * (GRID_SIZE + 1));
        mesh.indices.reserve(GRID_SIZE * GRID_SIZE * 6);

        float eye_origin_x = isLeftEye ? (-baseline_m * 0.5f) : (baseline_m * 0.5f);
        for (int y = 0; y <= GRID_SIZE; ++y) {
            for (int x = 0; x <= GRID_SIZE; ++x) {
                float grid_u = (float)x / GRID_SIZE;
                float grid_v = (float)y / GRID_SIZE;
                float screen_u = isLeftEye ? (grid_u * 0.5f) : (0.5f + grid_u * 0.5f);
                float screen_v = grid_v;
                float half_local_x = isLeftEye ? (screen_u * 2.0f) : ((screen_u - 0.5f) * 2.0f);
                float rect_x = evaluatePolynomial(half_local_x, screen_v, coef_x);
                float rect_y = evaluatePolynomial(half_local_x, screen_v, coef_y);
                float eye_uv_x = 0.5f + (rect_x / (2.0f * tan_half_fov_x)) + (isLeftEye ? left_eye_offset[0] : right_eye_offset[0]);
                float eye_uv_y = 0.5f - (rect_y / (2.0f * tan_half_fov_y)) + (isLeftEye ? left_eye_offset[1] : right_eye_offset[1]);
                float camera_slope_x = (eye_uv_x - 0.5f) * (2.0f * tan_half_fov_x);
                float camera_slope_y = (0.5f - eye_uv_y) * (2.0f * tan_half_fov_y);
                float world_x = eye_origin_x + (-camera_slope_y * center_z);
                float world_y = camera_slope_x * center_z;
                float src_u = 1.0f - ((world_x - plane_left) / plane_width_m);
                float src_v = (plane_top - world_y) / plane_height_m;
                mesh.vertices.push_back({screen_u * 2.0f - 1.0f, (1.0f - screen_v) * 2.0f - 1.0f, 0.0f, src_u, src_v});
            }
        }
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                unsigned short tl = static_cast<unsigned short>(y * (GRID_SIZE + 1) + x);
                unsigned short tr = static_cast<unsigned short>(tl + 1);
                unsigned short bl = static_cast<unsigned short>((y + 1) * (GRID_SIZE + 1) + x);
                unsigned short br = static_cast<unsigned short>(bl + 1);
                mesh.indices.push_back(tl); mesh.indices.push_back(bl); mesh.indices.push_back(tr);
                mesh.indices.push_back(tr); mesh.indices.push_back(bl); mesh.indices.push_back(br);
            }
        }
    };

    buildEyeMesh(leftMesh, left_x, left_y, true);
    buildEyeMesh(rightMesh, right_x, right_y, false);
    return true;
}

bool generateDistortionMeshesHudFromCalibration(
    const json& calibration_json,
    EyeMeshData& leftMesh,
    EyeMeshData& rightMesh,
    float panel_scale) {
    const json& calib =
        calibration_json.contains("v2CalibrationValues")
            ? calibration_json["v2CalibrationValues"]
            : calibration_json;

    std::vector<float> left_x, left_y, right_x, right_y;
    if (calib.contains("left_uv_to_rect_x")) left_x = calib["left_uv_to_rect_x"].get<std::vector<float>>(); else return false;
    if (calib.contains("left_uv_to_rect_y")) left_y = calib["left_uv_to_rect_y"].get<std::vector<float>>(); else return false;
    if (calib.contains("right_uv_to_rect_x")) right_x = calib["right_uv_to_rect_x"].get<std::vector<float>>(); else return false;
    if (calib.contains("right_uv_to_rect_y")) right_y = calib["right_uv_to_rect_y"].get<std::vector<float>>(); else return false;

    const float baseline_mm = calib.value("baseline", kDefaultBaselineMm);
    const std::vector<float>& left_coeff_x = kDefaultSwapEyes ? right_x : left_x;
    const std::vector<float>& left_coeff_y = kDefaultSwapEyes ? right_y : left_y;
    const std::vector<float>& right_coeff_x = kDefaultSwapEyes ? left_x : right_x;
    const std::vector<float>& right_coeff_y = kDefaultSwapEyes ? left_y : right_y;
    const float symmetric_shear = g_symmetric_shear.load(std::memory_order_relaxed);

    auto buildEyeMesh = [&](EyeMeshData& mesh, const std::vector<float>& coef_x, const std::vector<float>& coef_y, bool isLeftEye) {
        mesh.vertices.clear();
        mesh.indices.clear();
        mesh.vertices.reserve((kGridSize + 1) * (kGridSize + 1));
        mesh.indices.reserve(kGridSize * kGridSize * 6);

        const float eye_origin_x_mm = isLeftEye ? (-baseline_mm * 0.5f) : (baseline_mm * 0.5f);
        const EyeRenderAdjustments adjustments = getEyeAdjustments(eye_origin_x_mm, symmetric_shear);

        for (int y = 0; y <= kGridSize; ++y) {
            for (int x = 0; x <= kGridSize; ++x) {
                const float grid_u = static_cast<float>(x) / static_cast<float>(kGridSize);
                const float grid_v = static_cast<float>(y) / static_cast<float>(kGridSize);
                const float screen_u = isLeftEye ? (grid_u * 0.5f) : (0.5f + grid_u * 0.5f);
                const float screen_v = grid_v;
                const float local_u = isLeftEye ? (screen_u * 2.0f) : ((screen_u - 0.5f) * 2.0f);

                const float raw_ray_x = evaluateEyeRayComponent(local_u, screen_v, coef_x);
                const float raw_ray_y = evaluateEyeRayComponent(local_u, screen_v, coef_y);

                float ray_x = 0.0f;
                float ray_y = 0.0f;
                transformEyeRay(raw_ray_x, raw_ray_y, ray_x, ray_y);

                float hit_x_mm = 0.0f;
                float hit_y_mm = 0.0f;
                intersectTiltedPlane(
                    ray_x,
                    ray_y,
                    eye_origin_x_mm,
                    kDefaultPlaneCenterXMm,
                    kDefaultDepthMm,
                    adjustments.plane_angle_deg,
                    hit_x_mm,
                    hit_y_mm
                );

                float src_u = 0.0f;
                float src_v = 0.0f;
                computePanelUv(
                    hit_x_mm,
                    hit_y_mm,
                    kDefaultPlaneCenterXMm,
                    kDefaultPlaneCenterYMm,
                    kDefaultPanelWidthMm * panel_scale,
                    kDefaultPanelHeightMm * panel_scale,
                    adjustments.panel_shear,
                    src_u,
                    src_v
                );

                mesh.vertices.push_back({screen_u * 2.0f - 1.0f, (1.0f - screen_v) * 2.0f - 1.0f, 0.0f, src_u, src_v});
            }
        }

        for (int y = 0; y < kGridSize; ++y) {
            for (int x = 0; x < kGridSize; ++x) {
                unsigned short tl = static_cast<unsigned short>(y * (kGridSize + 1) + x);
                unsigned short tr = static_cast<unsigned short>(tl + 1);
                unsigned short bl = static_cast<unsigned short>((y + 1) * (kGridSize + 1) + x);
                unsigned short br = static_cast<unsigned short>(bl + 1);
                mesh.indices.push_back(tl); mesh.indices.push_back(bl); mesh.indices.push_back(tr);
                mesh.indices.push_back(tr); mesh.indices.push_back(bl); mesh.indices.push_back(br);
            }
        }
    };

    buildEyeMesh(leftMesh, left_coeff_x, left_coeff_y, true);
    buildEyeMesh(rightMesh, right_coeff_x, right_coeff_y, false);
    return true;
}

bool generateDistortionMeshesHud50cmFromCalibration(const json& calibration_json, EyeMeshData& leftMesh, EyeMeshData& rightMesh) {
    return generateDistortionMeshesHudFromCalibration(calibration_json, leftMesh, rightMesh, 1.0f);
}

bool generateDistortionMeshesHud50cm(const std::string& filepath, EyeMeshData& leftMesh, EyeMeshData& rightMesh) {
    json calib;
    if (!loadCalibrationJson(filepath, calib)) return false;
    return generateDistortionMeshesHud50cmFromCalibration(calib, leftMesh, rightMesh);
}

// ==========================================
// 其他 UI 模块
// ==========================================
void draw_icon(NVGcontext* vg, int app_id, float x, float y, float size, NVGcolor color) {
    nvgSave(vg);
    nvgTranslate(vg, x, y);
    nvgScale(vg, size / 100.0f, size / 100.0f); 
    nvgStrokeColor(vg, color);
    nvgStrokeWidth(vg, 4.0f);
    nvgLineCap(vg, NVG_ROUND);
    nvgLineJoin(vg, NVG_ROUND);

    if (app_id == 0) {
        nvgBeginPath(vg); nvgArc(vg, 0, 0, 25, M_PI/4, 3*M_PI/4, NVG_CCW); nvgStroke(vg);
        nvgBeginPath(vg); nvgArc(vg, 0, 0, 40, M_PI/4, 3*M_PI/4, NVG_CCW); nvgStroke(vg);
        nvgBeginPath(vg); nvgCircle(vg, 0, 25, 8); nvgFillColor(vg, color); nvgFill(vg);
    } else if (app_id == 1) {
        nvgBeginPath(vg);
        nvgMoveTo(vg, -15, 30); nvgLineTo(vg, -15, -10); nvgArcTo(vg, -15, -20, -5, -20, 10); nvgLineTo(vg, -5, 10); 
        nvgMoveTo(vg, -5, 10); nvgLineTo(vg, -5, -25); nvgArcTo(vg, -5, -35, 5, -35, 10); nvgLineTo(vg, 5, 10); 
        nvgMoveTo(vg, 5, 10); nvgLineTo(vg, 5, -15); nvgArcTo(vg, 5, -25, 15, -25, 10); nvgLineTo(vg, 15, 20); 
        nvgMoveTo(vg, 15, 15); nvgLineTo(vg, 25, 5); nvgArcTo(vg, 35, -5, 35, 10, 8); 
        nvgLineTo(vg, 25, 35); nvgLineTo(vg, -10, 35); nvgClosePath(vg); nvgStroke(vg);
    } else if (app_id == 2) {
        nvgBeginPath(vg); nvgRoundedRect(vg, -35, -25, 70, 50, 8); nvgStroke(vg);
        nvgBeginPath(vg); nvgCircle(vg, 0, 0, 15); nvgStroke(vg);
        nvgBeginPath(vg); nvgCircle(vg, 20, -15, 3); nvgFillColor(vg, color); nvgFill(vg);
    } else if (app_id == 3) {
        nvgBeginPath(vg); nvgMoveTo(vg, -20, 25); nvgLineTo(vg, 0, -30); nvgLineTo(vg, 20, 25); nvgLineTo(vg, 0, 10); nvgClosePath(vg);
        nvgStroke(vg);
    } else if (app_id == 4) {
        // 骑行模式图标：自行车+雷达波纹
        // 自行车轮子
        nvgBeginPath(vg); nvgCircle(vg, -18, 8, 10); nvgStroke(vg);
        nvgBeginPath(vg); nvgCircle(vg, 18, 8, 10); nvgStroke(vg);
        // 车架
        nvgBeginPath(vg); nvgMoveTo(vg, -18, 8); nvgLineTo(vg, 0, -15); nvgLineTo(vg, 18, 8); nvgStroke(vg);
        nvgBeginPath(vg); nvgMoveTo(vg, 0, -15); nvgLineTo(vg, 0, -5); nvgStroke(vg);
        // 雷达波纹（表示综合模式）
        nvgBeginPath(vg); nvgCircle(vg, 0, -20, 8); nvgStrokeWidth(vg, 1.5f); nvgStroke(vg);
        nvgBeginPath(vg); nvgCircle(vg, 0, -20, 4); nvgStrokeWidth(vg, 2.0f); nvgStroke(vg);
    } else if (app_id == 5) {
        nvgBeginPath(vg);
        nvgRoundedRect(vg, -36, -30, 72, 60, 18);
        nvgStroke(vg);

        nvgBeginPath(vg);
        nvgMoveTo(vg, -18, -4);
        nvgLineTo(vg, -2, -4);
        nvgLineTo(vg, 14, -18);
        nvgLineTo(vg, 14, 18);
        nvgLineTo(vg, -2, 4);
        nvgLineTo(vg, -18, 4);
        nvgClosePath(vg);
        nvgFillColor(vg, color);
        nvgFill(vg);

        nvgBeginPath(vg);
        nvgArc(vg, 20, 0, 10, -0.8f, 0.8f, NVG_CW);
        nvgStroke(vg);
        nvgBeginPath(vg);
        nvgArc(vg, 24, 0, 18, -0.9f, 0.9f, NVG_CW);
        nvgStroke(vg);

        nvgBeginPath(vg);
        nvgMoveTo(vg, -28, -22);
        nvgLineTo(vg, -22, -36);
        nvgLineTo(vg, -10, -18);
        nvgStroke(vg);
        nvgBeginPath(vg);
        nvgMoveTo(vg, -30, 6);
        nvgLineTo(vg, -18, 6);
        nvgMoveTo(vg, -30, 18);
        nvgLineTo(vg, -10, 18);
        nvgStroke(vg);
    } else if (app_id == 6) {
        nvgBeginPath(vg); nvgRoundedRect(vg, -18, -22, 36, 44, 12); nvgStroke(vg);
        nvgBeginPath(vg); nvgMoveTo(vg, -10, 26); nvgLineTo(vg, 10, 26); nvgStroke(vg);
        nvgBeginPath(vg); nvgMoveTo(vg, 0, 22); nvgLineTo(vg, 0, 34); nvgStroke(vg);
        nvgBeginPath(vg); nvgArc(vg, 0, -2, 28, -M_PI * 0.25f, M_PI * 0.25f, NVG_CW); nvgStroke(vg);
        nvgBeginPath(vg); nvgArc(vg, 0, -2, 18, -M_PI * 0.25f, M_PI * 0.25f, NVG_CW); nvgStroke(vg);
        nvgBeginPath(vg); nvgArc(vg, 0, -2, 8, -M_PI * 0.25f, M_PI * 0.25f, NVG_CW); nvgStroke(vg);
    }
    nvgRestore(vg);
}

float g_current_scroll = 0.0f;
void render_desktop_nvg(NVGcontext* vg, float width, float height, int fontNormal, float state_time) {
    // 放大比例：28/18 = 1.555...
    const float scale_factor = 28.0f / 18.0f;

    float fade_alpha = std::min(1.0f, state_time / 0.5f);
    float enter_offset_y = (1.0f - fade_alpha) * 40.0f * scale_factor;
    // 回归中心：使用 height * 0.5f 而不是 height * 0.25f
    float base_y = height * 0.5f + enter_offset_y;

    // 等比例放大所有尺寸
    float icon_spacing = 126.0f * scale_factor;      // 原126 -> 196
    float icon_size = 58.0f * scale_factor;          // 原58 -> 90

    int current_page = g_selected_app.load() / 4;
    float target_scroll = (current_page * width);
    g_current_scroll += (target_scroll - g_current_scroll) * 0.15f;

    float group_width = 3 * icon_spacing;
    float start_x = width * 0.5f - group_width * 0.5f;

    nvgSave(vg); nvgTranslate(vg, -g_current_scroll, 0);
    const char* app_names[] = {"SOUND RADAR", "SIGN TRANSLATE", "REAR CAMERA", "NAVIGATION", "RIDING MODE", "HEARING ASSIST", "SPEECH TO TEXT"};
    for (int i = 0; i < TOTAL_APPS; i++) {
        int page_index = i / 4, slot_index = i % 4;
        float x = start_x + (slot_index * icon_spacing) + (page_index * width);
        float y = base_y;
        bool is_selected = (i == g_selected_app.load());
        float target_scale = is_selected ? 1.28f : 1.08f;
        float target_offset_y = is_selected ? -18.0f * scale_factor : 0.0f;  // 原-18 -> -28

        NVGcolor icon_color = is_selected ? nvgRGBAf(1.0f, 1.0f, 1.0f, 1.0f * fade_alpha) : nvgRGBAf(0.85f, 0.92f, 0.95f, 0.95f * fade_alpha);
        NVGcolor box_color = is_selected ? nvgRGBAf(0.0f, 0.90f, 0.70f, 0.28f * fade_alpha) : nvgRGBAf(0.12f, 0.16f, 0.18f, 0.20f * fade_alpha);

        // 放大图标背景框：原92x92, 圆角14
        float box_size = 92.0f * scale_factor;  // -> 143
        float box_corner = 14.0f * scale_factor; // -> 22
        nvgBeginPath(vg);
        nvgRoundedRect(vg, x - box_size/2, y + target_offset_y - box_size/2, box_size, box_size, box_corner);
        nvgFillColor(vg, box_color); nvgFill(vg);
        if (is_selected) {
            nvgStrokeColor(vg, nvgRGBAf(0.0f, 1.0f, 0.8f, 1.0f * fade_alpha));
            nvgStrokeWidth(vg, 2.0f * scale_factor);  // 原2 -> 3.1
            nvgStroke(vg);
        }
        draw_icon(vg, i, x, y + target_offset_y, icon_size * target_scale, icon_color);

        if (is_selected && fontNormal != -1) {
            nvgFontSize(vg, 28.0f);  // 原18 -> 28，与screen cast一致
            nvgFontFace(vg, "sans");
            nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
            nvgFillColor(vg, nvgRGBAf(0.95f, 1.0f, 1.0f, 1.0f * fade_alpha));
            nvgText(vg, x, y + 68.0f * scale_factor, app_names[i], NULL);  // 原68 -> 106
        }
    }
    nvgRestore(vg);

    int total_pages = (TOTAL_APPS - 1) / 4 + 1;
    if (total_pages > 1) {
        float dots_spacing = 20.0f * scale_factor;  // 原20 -> 31
        float dots_radius = 4.0f * scale_factor;    // 原4 -> 6.2
        float dots_w = (total_pages - 1) * dots_spacing;
        float dots_start_x = width / 2.0f - dots_w / 2.0f;
        float dots_offset = 100.0f * scale_factor;  // 原100 -> 156
        for (int p = 0; p < total_pages; p++) {
            nvgBeginPath(vg);
            nvgCircle(vg, dots_start_x + p * dots_spacing, base_y + dots_offset, dots_radius);
            nvgFillColor(vg, p == current_page ? nvgRGBAf(0.0f, 1.0f, 0.8f, fade_alpha) : nvgRGBAf(1.0f, 1.0f, 1.0f, 0.45f * fade_alpha));
            nvgFill(vg);
        }
    }

    if (g_desktop_tuning_mode.load(std::memory_order_relaxed) && fontNormal != -1) {
        const float panel_w = 300.0f;
        const float panel_h = 126.0f;
        const float panel_x = width * 0.5f - panel_w * 0.5f;
        const float panel_y = height * 0.5f - panel_h * 0.5f;

        nvgBeginPath(vg);
        nvgRoundedRect(vg, panel_x, panel_y, panel_w, panel_h, 20.0f);
        nvgFillColor(vg, nvgRGBAf(0.05f, 0.08f, 0.10f, 0.90f * fade_alpha));
        nvgFill(vg);
        nvgStrokeColor(vg, nvgRGBAf(0.32f, 0.78f, 0.72f, 0.95f * fade_alpha));
        nvgStrokeWidth(vg, 2.5f);
        nvgStroke(vg);

        nvgFontFace(vg, "sans");
        nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
        nvgFontSize(vg, 54.0f);
        nvgFillColor(vg, nvgRGBAf(0.92f, 0.98f, 0.98f, 0.98f * fade_alpha));
        nvgText(vg, width * 0.5f, panel_y + 46.0f, "调参模式", nullptr);

        char shear_buf[64];
        snprintf(shear_buf, sizeof(shear_buf), "SHEAR %.5f", g_symmetric_shear.load(std::memory_order_relaxed));
        nvgFontSize(vg, 22.0f);
        nvgFillColor(vg, nvgRGBAf(0.68f, 0.90f, 0.88f, 0.90f * fade_alpha));
        nvgText(vg, width * 0.5f, panel_y + 92.0f, shear_buf, nullptr);
    }
}

void render_placeholder_nvg(NVGcontext* vg, float width, float height, int fontNormal, float state_time) {
    float fade_alpha = std::min(1.0f, state_time / 0.5f);
    if (fontNormal != -1) {
        nvgFontSize(vg, 28.0f); nvgFontFace(vg, "sans"); nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
        nvgFillColor(vg, nvgRGBAf(1.0f, 0.65f, 0.35f, fade_alpha)); nvgText(vg, width / 2.0f, height / 2.0f - 20, "[ MODULE NOT INSTALLED ]", NULL);
    }
}

void speech_to_text_supervisor_thread() {
    const char* stt_bin_env = std::getenv("SPEECH_TO_TEXT_BIN");
    const char* stt_encoder_env = std::getenv("SPEECH_TO_TEXT_ENCODER");
    const char* stt_decoder_env = std::getenv("SPEECH_TO_TEXT_DECODER");
    const char* stt_joiner_env = std::getenv("SPEECH_TO_TEXT_JOINER");
    const char* stt_tokens_env = std::getenv("SPEECH_TO_TEXT_TOKENS");
    const char* stt_device_env = std::getenv("SPEECH_TO_TEXT_DEVICE");
    const std::string executable_dir = resolve_executable_dir();

    const std::string stt_bin = stt_bin_env && *stt_bin_env
        ? std::string(stt_bin_env)
        : first_existing_path({
            "/home/elf/.local/bin/sherpa-onnx-alsa",
            "/usr/local/bin/sherpa-onnx-alsa",
            executable_dir + "/sherpa-onnx-alsa"
        });
    const std::string encoder_path = stt_encoder_env && *stt_encoder_env
        ? std::string(stt_encoder_env)
        : first_existing_path({
            "/home/elf/sherpa-models/stt/zipformer-small/encoder.rknn",
            executable_dir + "/sherpa-models/stt/zipformer-small/encoder.rknn"
        });
    const std::string decoder_path = stt_decoder_env && *stt_decoder_env
        ? std::string(stt_decoder_env)
        : first_existing_path({
            "/home/elf/sherpa-models/stt/zipformer-small/decoder.rknn",
            executable_dir + "/sherpa-models/stt/zipformer-small/decoder.rknn"
        });
    const std::string joiner_path = stt_joiner_env && *stt_joiner_env
        ? std::string(stt_joiner_env)
        : first_existing_path({
            "/home/elf/sherpa-models/stt/zipformer-small/joiner.rknn",
            executable_dir + "/sherpa-models/stt/zipformer-small/joiner.rknn"
        });
    const std::string tokens_path = stt_tokens_env && *stt_tokens_env
        ? std::string(stt_tokens_env)
        : first_existing_path({
            "/home/elf/sherpa-models/stt/zipformer-small/tokens.txt",
            executable_dir + "/sherpa-models/stt/zipformer-small/tokens.txt"
        });
    const std::string input_device = stt_device_env && *stt_device_env
        ? std::string(stt_device_env)
        : std::string(kDefaultSignTranslateTtsDevice);

    std::string pending;

    while (true) {
        bool should_run = false;
        pid_t child_pid = -1;
        int output_fd = -1;
        {
            std::unique_lock<std::mutex> lock(g_speech_to_text_runtime.mtx);
            g_speech_to_text_runtime.cv.wait_for(lock, std::chrono::milliseconds(50), [] {
                return g_speech_to_text_runtime.shutdown ||
                       g_speech_to_text_runtime.app_running ||
                       g_speech_to_text_runtime.service_pid > 0;
            });
            if (g_speech_to_text_runtime.shutdown) break;
            should_run = g_speech_to_text_runtime.app_running;
            child_pid = g_speech_to_text_runtime.service_pid;
            output_fd = g_speech_to_text_runtime.output_fd;
        }

        if (should_run && child_pid <= 0) {
            if (stt_bin.empty() ||
                !file_exists_quiet(encoder_path) ||
                !file_exists_quiet(decoder_path) ||
                !file_exists_quiet(joiner_path) ||
                !file_exists_quiet(tokens_path)) {
                std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
                g_speech_to_text_runtime.status_text = "STT MODEL MISSING";
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            int pipefd[2] = {-1, -1};
            if (pipe(pipefd) != 0) {
                if (pipefd[0] >= 0) close(pipefd[0]);
                if (pipefd[1] >= 0) close(pipefd[1]);
                std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
                g_speech_to_text_runtime.status_text = "STT PIPE FAILED";
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            pid_t pid = fork();
            if (pid == 0) {
                setpgid(0, 0);
                const std::string encoder_arg = "--encoder=" + encoder_path;
                const std::string decoder_arg = "--decoder=" + decoder_path;
                const std::string joiner_arg = "--joiner=" + joiner_path;
                const std::string tokens_arg = "--tokens=" + tokens_path;
                char* argv_exec[16];
                int argi = 0;
                argv_exec[argi++] = const_cast<char*>(stt_bin.c_str());
                argv_exec[argi++] = const_cast<char*>("--provider=rknn");
                argv_exec[argi++] = const_cast<char*>("--num-threads=1");
                argv_exec[argi++] = const_cast<char*>(encoder_arg.c_str());
                argv_exec[argi++] = const_cast<char*>(decoder_arg.c_str());
                argv_exec[argi++] = const_cast<char*>(joiner_arg.c_str());
                argv_exec[argi++] = const_cast<char*>(tokens_arg.c_str());
                argv_exec[argi++] = const_cast<char*>(input_device.c_str());
                argv_exec[argi++] = nullptr;

                close(pipefd[0]);
                dup2(pipefd[1], STDOUT_FILENO);
                dup2(pipefd[1], STDERR_FILENO);
                close(pipefd[1]);
                execv(stt_bin.c_str(), argv_exec);
                _exit(127);
            }

            close(pipefd[1]);
            if (pid > 0) {
                int flags = fcntl(pipefd[0], F_GETFL, 0);
                if (flags >= 0) {
                    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);
                }

                std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
                g_speech_to_text_runtime.service_pid = pid;
                g_speech_to_text_runtime.output_fd = pipefd[0];
                g_speech_to_text_runtime.backend_ready = false;
                g_speech_to_text_runtime.status_text = "LISTENING";
                pending.clear();
                g_speech_to_text_runtime.cv.notify_all();
            } else {
                close(pipefd[0]);
                std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
                g_speech_to_text_runtime.status_text = "STT START FAILED";
                g_speech_to_text_runtime.cv.notify_all();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (!should_run && child_pid > 0) {
            stop_child_process(child_pid);
            {
                std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
                g_speech_to_text_runtime.service_pid = -1;
                close_speech_to_text_output_locked();
                g_speech_to_text_runtime.backend_ready = false;
                g_speech_to_text_runtime.display_text.clear();
                g_speech_to_text_runtime.display_start_time = -1000.0f;
                g_speech_to_text_runtime.status_text = "WAITING FOR SPEECH";
                g_speech_to_text_runtime.cv.notify_all();
            }
            pending.clear();
            continue;
        }

        if (child_pid > 0 && output_fd >= 0) {
            char buffer[1024];
            while (true) {
                ssize_t bytes_read = read(output_fd, buffer, sizeof(buffer));
                if (bytes_read > 0) {
                    pending.append(buffer, static_cast<size_t>(bytes_read));
                    size_t split_pos = 0;
                    while ((split_pos = pending.find_first_of("\r\n")) != std::string::npos) {
                        std::string frame = pending.substr(0, split_pos);
                        size_t erase_len = 1;
                        while (split_pos + erase_len < pending.size() &&
                               (pending[split_pos + erase_len] == '\r' || pending[split_pos + erase_len] == '\n')) {
                            ++erase_len;
                        }
                        pending.erase(0, split_pos + erase_len);
                        if (frame.empty()) continue;
                        handle_speech_to_text_output_line(frame);
                    }
                    if (!pending.empty()) {
                        handle_speech_to_text_output_line(pending);
                    }
                    continue;
                }

                if (bytes_read == 0) {
                    if (!pending.empty()) {
                        handle_speech_to_text_output_line(pending);
                    }
                    break;
                }
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    if (!pending.empty()) {
                        handle_speech_to_text_output_line(pending);
                    }
                    break;
                }
                break;
            }

            int status = 0;
            pid_t wait_result = waitpid(child_pid, &status, WNOHANG);
            if (wait_result == child_pid) {
                std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
                g_speech_to_text_runtime.service_pid = -1;
                close_speech_to_text_output_locked();
                g_speech_to_text_runtime.backend_ready = false;
                g_speech_to_text_runtime.status_text = g_speech_to_text_runtime.app_running
                    ? "STT RESTARTING"
                    : "WAITING FOR SPEECH";
                pending.clear();
                g_speech_to_text_runtime.cv.notify_all();
            }
        }
    }

    pid_t child_pid = -1;
    {
        std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
        child_pid = g_speech_to_text_runtime.service_pid;
        g_speech_to_text_runtime.service_pid = -1;
        close_speech_to_text_output_locked();
        g_speech_to_text_runtime.backend_ready = false;
        g_speech_to_text_runtime.cv.notify_all();
    }
    if (child_pid > 0) {
        stop_child_process(child_pid);
    }
}

void draw_rear_camera_panel(
    NVGcontext* vg,
    const cv::Mat& rgba_frame,
    bool frame_ready,
    float frame_x,
    float frame_y,
    float frame_w,
    float frame_h,
    float fade_alpha,
    float corner_radius,
    float stroke_width,
    int& image_handle,
    int& image_w,
    int& image_h) {
    nvgBeginPath(vg);
    nvgRoundedRect(vg, frame_x, frame_y, frame_w, frame_h, corner_radius);
    nvgFillColor(vg, nvgRGBAf(0.05f, 0.08f, 0.10f, 0.96f * fade_alpha));
    nvgFill(vg);
    nvgStrokeColor(vg, nvgRGBAf(0.36f, 0.40f, 0.45f, 0.96f * fade_alpha));
    nvgStrokeWidth(vg, stroke_width);
    nvgStroke(vg);

    if (frame_ready) {
        if (image_handle == -1 || image_w != rgba_frame.cols || image_h != rgba_frame.rows) {
            if (image_handle != -1) nvgDeleteImage(vg, image_handle);
            image_handle = nvgCreateImageRGBA(vg, rgba_frame.cols, rgba_frame.rows, 0, rgba_frame.data);
            image_w = rgba_frame.cols;
            image_h = rgba_frame.rows;
        } else {
            nvgUpdateImage(vg, image_handle, rgba_frame.data);
        }

        if (image_handle != -1) {
            const float scale = std::max(
                frame_w / static_cast<float>(rgba_frame.rows),
                frame_h / static_cast<float>(rgba_frame.cols));
            const float draw_w = rgba_frame.cols * scale;
            const float draw_h = rgba_frame.rows * scale;

            nvgSave(vg);
            nvgTranslate(vg, frame_x + frame_w * 0.5f, frame_y + frame_h * 0.5f);
            nvgRotate(vg, static_cast<float>(M_PI) / 2.0f);
            NVGpaint img_paint = nvgImagePattern(
                vg,
                -draw_h * 0.5f,
                -draw_w * 0.5f,
                draw_h,
                draw_w,
                0.0f,
                image_handle,
                fade_alpha);
            nvgBeginPath(vg);
            nvgRoundedRect(vg, -frame_h * 0.5f, -frame_w * 0.5f, frame_h, frame_w, corner_radius);
            nvgFillPaint(vg, img_paint);
            nvgFill(vg);
            nvgRestore(vg);
        }
    } else {
        nvgBeginPath(vg);
        nvgRoundedRect(vg, frame_x, frame_y, frame_w, frame_h, corner_radius);
        nvgFillColor(vg, nvgRGBAf(0.02f, 0.03f, 0.05f, 0.98f * fade_alpha));
        nvgFill(vg);
    }
}

void render_rear_camera_nvg(NVGcontext* vg, float width, float height, int fontNormal, float state_time) {
    static int nvg_rear_image = -1;
    static int image_w = 0;
    static int image_h = 0;

    const float fade_alpha = std::min(1.0f, state_time / 0.35f);

    cv::Mat rgba_frame;
    RearCameraSettings settings;
    bool frame_ready = false;
    {
        std::lock_guard<std::mutex> lock(g_rear_camera_runtime.mtx);
        rgba_frame = g_rear_camera_runtime.latest_rgba;
        settings = g_rear_camera_runtime.settings;
        frame_ready = g_rear_camera_runtime.frame_ready && !rgba_frame.empty();
    }

    const float frame_w = std::min(settings.display_width_px, width - 120.0f);
    const float frame_h = std::min(settings.display_height_px, height - 120.0f);
    const float frame_x = (width - frame_w) * 0.5f;
    const float frame_y = (height - frame_h) * 0.5f;

    nvgBeginPath(vg);
    nvgRect(vg, 0.0f, 0.0f, width, height);
    nvgFillColor(vg, nvgRGBAf(0.01f, 0.02f, 0.03f, fade_alpha));
    nvgFill(vg);

    draw_rear_camera_panel(
        vg,
        rgba_frame,
        frame_ready,
        frame_x,
        frame_y,
        frame_w,
        frame_h,
        fade_alpha,
        28.0f,
        3.5f,
        nvg_rear_image,
        image_w,
        image_h);
}

void render_sign_translate_nvg(NVGcontext* vg, float width, float height, int fontNormal, float state_time) {
    const float fade_alpha = std::min(1.0f, state_time / 0.35f);
    const float now = monotonic_time_sec();

    std::string display_text;
    std::string status_text;
    float display_confidence = 0.0f;
    float display_start_time = -1000.0f;
    bool backend_thinking = false;
    float thinking_transition_time = -1000.0f;
    {
        std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
        display_text = g_sign_translate_runtime.display_text;
        status_text = g_sign_translate_runtime.status_text;
        display_confidence = g_sign_translate_runtime.display_confidence;
        display_start_time = g_sign_translate_runtime.display_start_time;
        backend_thinking = g_sign_translate_runtime.backend_thinking;
        thinking_transition_time = g_sign_translate_runtime.thinking_transition_time;
    }

    const float panel_w = 560.0f;
    const float panel_h = 180.0f;
    const float panel_x = width * 0.5f - panel_w * 0.5f;
    const float panel_y = height * 0.5f - panel_h * 0.5f - 24.0f;

    nvgBeginPath(vg);
    nvgRect(vg, 0.0f, 0.0f, width, height);
    nvgFillColor(vg, nvgRGBAf(0.01f, 0.02f, 0.03f, 0.50f * fade_alpha));
    nvgFill(vg);

    nvgBeginPath(vg);
    nvgRoundedRect(vg, panel_x, panel_y, panel_w, panel_h, 28.0f);
    nvgFillColor(vg, nvgRGBAf(0.05f, 0.08f, 0.10f, 0.90f * fade_alpha));
    nvgFill(vg);
    nvgStrokeColor(vg, nvgRGBAf(0.32f, 0.78f, 0.72f, 0.95f * fade_alpha));
    nvgStrokeWidth(vg, 3.0f);
    nvgStroke(vg);

    nvgBeginPath(vg);
    nvgCircle(vg, width * 0.5f, panel_y + 26.0f, 6.0f);
    nvgFillColor(vg, nvgRGBAf(0.34f, 0.96f, 0.88f, 0.95f * fade_alpha));
    nvgFill(vg);

    if (fontNormal != -1) {
        nvgFontFace(vg, "sans");
        nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);

        nvgFontSize(vg, 20.0f);
        nvgFillColor(vg, nvgRGBAf(0.72f, 0.86f, 0.90f, 0.95f * fade_alpha));
        nvgText(vg, width * 0.5f, panel_y + 56.0f, "手语输入", nullptr);

        const float age = now - display_start_time;
        const bool show_result =
            !display_text.empty() && age >= 0.0f && age < kSignTranslateTextLifetimeSec;
        if (show_result) {
            float t = 1.0f - (age / kSignTranslateTextLifetimeSec);
            t = std::clamp(t, 0.0f, 1.0f);
            const float text_alpha = fade_alpha * t * t * (3.0f - 2.0f * t);

            NVGpaint text_gradient = nvgLinearGradient(
                vg,
                panel_x + 72.0f, panel_y + 76.0f,
                panel_x + panel_w - 72.0f, panel_y + panel_h - 24.0f,
                nvgRGBAf(0.35f, 0.98f, 0.86f, text_alpha),
                nvgRGBAf(0.96f, 1.00f, 1.00f, text_alpha));

            nvgFontSize(vg, 52.0f);
            nvgFillPaint(vg, text_gradient);
            nvgText(vg, width * 0.5f, panel_y + 110.0f, display_text.c_str(), nullptr);

            char conf_buf[64];
            snprintf(conf_buf, sizeof(conf_buf), "置信度 %.0f%%", display_confidence * 100.0f);
            nvgFontSize(vg, 18.0f);
            nvgFillColor(vg, nvgRGBAf(0.82f, 0.94f, 0.96f, 0.85f * text_alpha));
            nvgText(vg, width * 0.5f, panel_y + 144.0f, conf_buf, nullptr);
        } else {
            float thinking_alpha = 0.0f;
            const float thinking_t = std::clamp((now - thinking_transition_time) / 0.18f, 0.0f, 1.0f);
            if (backend_thinking) {
                thinking_alpha = fade_alpha * (thinking_t * thinking_t * (3.0f - 2.0f * thinking_t));
            } else {
                const float inv_t = 1.0f - thinking_t;
                thinking_alpha = fade_alpha * std::max(0.0f, inv_t * inv_t * (3.0f - 2.0f * inv_t));
            }

            if (thinking_alpha > 0.02f) {
                nvgFontSize(vg, 44.0f);
                nvgFillColor(vg, nvgRGBAf(0.78f, 0.90f, 0.96f, 0.80f * thinking_alpha));
                nvgText(vg, width * 0.5f, panel_y + 108.0f, "...", nullptr);
            } else {
                nvgFontSize(vg, 38.0f);
                nvgFillColor(vg, nvgRGBAf(0.78f, 0.84f, 0.88f, 0.60f * fade_alpha));
                nvgText(vg, width * 0.5f, panel_y + 108.0f, status_text.c_str(), nullptr);
            }
        }
    }
}

void draw_sound_radar_symbol(NVGcontext* vg, RadarSoundType type, float size, float alpha) {
    const NVGcolor stroke = type == RadarSoundType::HORN
        ? nvgRGBAf(0.96f, 0.98f, 1.0f, alpha)
        : nvgRGBAf(1.0f, 0.97f, 0.93f, alpha);
    const NVGcolor accent = type == RadarSoundType::HORN
        ? nvgRGBAf(0.20f, 0.92f, 1.0f, alpha)
        : nvgRGBAf(1.0f, 0.72f, 0.35f, alpha);

    nvgSave(vg);
    nvgScale(vg, size / 100.0f, size / 100.0f);
    nvgStrokeWidth(vg, 4.0f);
    nvgLineCap(vg, NVG_ROUND);
    nvgLineJoin(vg, NVG_ROUND);

    nvgBeginPath(vg);
    nvgRoundedRect(vg, -34, -34, 68, 68, 22);
    nvgFillColor(vg, nvgRGBAf(0.06f, 0.08f, 0.10f, 0.92f * alpha));
    nvgFill(vg);
    nvgStrokeColor(vg, nvgRGBAf(accent.r, accent.g, accent.b, 1.0f * alpha));
    nvgStroke(vg);

    if (type == RadarSoundType::HORN) {
        nvgBeginPath(vg);
        nvgMoveTo(vg, -18, 7);
        nvgLineTo(vg, -6, 7);
        nvgLineTo(vg, 8, 19);
        nvgLineTo(vg, 8, -19);
        nvgLineTo(vg, -6, -7);
        nvgLineTo(vg, -18, -7);
        nvgClosePath(vg);
        nvgFillColor(vg, stroke);
        nvgFill(vg);

        nvgBeginPath(vg);
        nvgArc(vg, 13, 0, 10, -0.8f, 0.8f, NVG_CW);
        nvgStrokeColor(vg, accent);
        nvgStroke(vg);
        nvgBeginPath(vg);
        nvgArc(vg, 17, 0, 18, -0.9f, 0.9f, NVG_CW);
        nvgStrokeColor(vg, nvgRGBAf(accent.r, accent.g, accent.b, 0.82f * alpha));
        nvgStroke(vg);
    } else {
        nvgBeginPath(vg);
        nvgMoveTo(vg, -16, -8);
        nvgLineTo(vg, -24, -22);
        nvgLineTo(vg, -10, -18);
        nvgClosePath(vg);
        nvgFillColor(vg, accent);
        nvgFill(vg);

        nvgBeginPath(vg);
        nvgMoveTo(vg, 16, -8);
        nvgLineTo(vg, 24, -22);
        nvgLineTo(vg, 10, -18);
        nvgClosePath(vg);
        nvgFill(vg);

        nvgBeginPath(vg);
        nvgRoundedRect(vg, -22, -16, 44, 38, 18);
        nvgFillColor(vg, stroke);
        nvgFill(vg);

        nvgBeginPath(vg); nvgCircle(vg, -9, -2, 3); nvgFillColor(vg, nvgRGBAf(0.10f, 0.12f, 0.16f, alpha)); nvgFill(vg);
        nvgBeginPath(vg); nvgCircle(vg, 9, -2, 3); nvgFill(vg);
        nvgBeginPath(vg); nvgCircle(vg, 0, 8, 4); nvgFillColor(vg, accent); nvgFill(vg);
        nvgBeginPath(vg);
        nvgMoveTo(vg, -8, 14);
        nvgQuadTo(vg, 0, 19, 8, 14);
        nvgStrokeColor(vg, nvgRGBAf(0.10f, 0.12f, 0.16f, alpha));
        nvgStroke(vg);
    }

    nvgRestore(vg);
}

void draw_sound_radar_item(NVGcontext* vg, const RadarUiTarget& target, float center_x, float center_y, float radius, float now_sec) {
    float age = now_sec - target.last_update_time;
    if (age < 0.0f) age = 0.0f;
    if (!target.active || age > kRadarLifetimeSec) return;

    float fade = std::clamp(1.0f - age / kRadarLifetimeSec, 0.0f, 1.0f);
    fade = fade * fade * (3.0f - 2.0f * fade);
    float angle_rad = target.display_angle_deg * static_cast<float>(M_PI / 180.0f);
    float px = center_x + std::sin(angle_rad) * radius;
    float py = center_y - std::cos(angle_rad) * radius;
    float pulse = 1.0f;
    float glow_r = 30.0f + 8.0f * target.confidence;

    nvgSave(vg);
    nvgTranslate(vg, px, py);
    nvgRotate(vg, angle_rad);

    NVGcolor glow = target.type == RadarSoundType::HORN
        ? nvgRGBAf(0.20f, 0.92f, 1.0f, 0.28f * fade)
        : nvgRGBAf(1.0f, 0.72f, 0.35f, 0.28f * fade);
    nvgBeginPath(vg);
    nvgCircle(vg, 0, 0, glow_r * pulse);
    nvgFillColor(vg, glow);
    nvgFill(vg);

    NVGcolor marker = target.type == RadarSoundType::HORN
        ? nvgRGBAf(0.72f, 0.96f, 1.0f, 1.0f * fade)
        : nvgRGBAf(1.0f, 0.84f, 0.55f, 1.0f * fade);

    auto draw_arc_triangle = [&](float offset_x, float direction_sign) {
        nvgBeginPath(vg);
        nvgMoveTo(vg, offset_x + 18.0f * direction_sign, 0.0f);
        nvgLineTo(vg, offset_x - 8.0f * direction_sign, -14.0f);
        nvgLineTo(vg, offset_x - 8.0f * direction_sign, 14.0f);
        nvgClosePath(vg);
        nvgFillColor(vg, marker);
        nvgFill(vg);
    };

    draw_arc_triangle(-82.0f, -1.0f);
    draw_arc_triangle(82.0f, 1.0f);

    draw_sound_radar_symbol(vg, target.type, 66.0f + 10.0f * target.confidence, fade);
    nvgRestore(vg);
}

Point2D radar_uv_to_fbo(float u, float v) {
    return {
        u * kUiWidthPx,
        v * kUiHeightPx
    };
}

void advance_radar_target(RadarUiTarget& target, float dt_sec) {
    if (dt_sec <= 0.0f) return;

    float delta = normalize_angle_deg(target.target_angle_deg - target.display_angle_deg);
    float accel = delta * kRadarAngleSpring - target.angle_velocity_deg * kRadarAngleDamping;
    target.angle_velocity_deg += accel * dt_sec;
    target.angle_velocity_deg = std::clamp(target.angle_velocity_deg, -kRadarAngleMaxSpeedDeg, kRadarAngleMaxSpeedDeg);
    target.display_angle_deg = normalize_angle_deg(target.display_angle_deg + target.angle_velocity_deg * dt_sec);
}

std::vector<RadarUiTarget> snapshot_radar_targets(float now_sec) {
    std::vector<RadarUiTarget> targets;
    {
        std::lock_guard<std::mutex> lock(g_radar_runtime.mtx);
        static float last_update_sec = now_sec;
        float dt_sec = std::clamp(now_sec - last_update_sec, 0.0f, 0.05f);
        last_update_sec = now_sec;
        for (auto& item : g_radar_runtime.targets) {
            advance_radar_target(item, dt_sec);
        }
        targets = g_radar_runtime.targets;
    }
    return targets;
}

void render_sound_radar_nvg(NVGcontext* vg, float time_sec, float cx, float cy, float state_time) {
    const float now_sec = monotonic_time_sec();
    const float safe_margin_px = 24.0f;
    const float marker_extent_px = 100.0f;
    const float symbol_extent_px = 44.0f;
    const float glow_extent_px = 40.0f;
    const float content_extent_px = std::max(marker_extent_px, std::max(symbol_extent_px, glow_extent_px));

    std::vector<RadarUiTarget> targets = snapshot_radar_targets(now_sec);

    Point2D center_px = radar_uv_to_fbo(kRadarCenterU, kRadarCenterV);
    float max_radius_x = std::min(
        center_px.x - (content_extent_px + safe_margin_px),
        (kUiWidthPx - center_px.x) - (content_extent_px + safe_margin_px)
    );
    float max_radius_y = std::min(
        center_px.y - (content_extent_px + safe_margin_px),
        (kUiHeightPx - center_px.y) - (content_extent_px + safe_margin_px)
    );

    for (const auto& target : targets) {
        float angle_rad = target.display_angle_deg * static_cast<float>(M_PI / 180.0f);
        float orbit_u = kRadarCenterU + std::sin(angle_rad) * kRadarRadiusU;
        float orbit_v = kRadarCenterV - std::cos(angle_rad) * kRadarRadiusV;
        Point2D orbit_px = radar_uv_to_fbo(orbit_u, orbit_v);
        float desired_radius_px = std::hypot(orbit_px.x - center_px.x, orbit_px.y - center_px.y);
        float orbit_radius_px = std::max(24.0f, std::min(desired_radius_px, std::min(max_radius_x, max_radius_y)));
        draw_sound_radar_item(vg, target, center_px.x, center_px.y, orbit_radius_px, now_sec);
    }
}

void render_speech_to_text_nvg(NVGcontext* vg, float width, float height, int fontNormal, float state_time) {
    const float fade_alpha = std::min(1.0f, state_time / 0.35f);
    const float now = monotonic_time_sec();

    std::string display_text;
    std::string status_text;
    float display_start_time = -1000.0f;
    {
        std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
        display_text = g_speech_to_text_runtime.display_text;
        status_text = g_speech_to_text_runtime.status_text;
        display_start_time = g_speech_to_text_runtime.display_start_time;
    }

    const float panel_w = 560.0f;
    const float panel_h = 180.0f;
    const float panel_x = width * 0.5f - panel_w * 0.5f;
    const float panel_y = height * 0.5f - panel_h * 0.5f - 24.0f;

    nvgBeginPath(vg);
    nvgRect(vg, 0.0f, 0.0f, width, height);
    nvgFillColor(vg, nvgRGBAf(0.01f, 0.02f, 0.03f, 0.50f * fade_alpha));
    nvgFill(vg);

    nvgBeginPath(vg);
    nvgRoundedRect(vg, panel_x, panel_y, panel_w, panel_h, 28.0f);
    nvgFillColor(vg, nvgRGBAf(0.05f, 0.08f, 0.10f, 0.90f * fade_alpha));
    nvgFill(vg);
    nvgStrokeColor(vg, nvgRGBAf(0.30f, 0.82f, 1.00f, 0.95f * fade_alpha));
    nvgStrokeWidth(vg, 3.0f);
    nvgStroke(vg);

    if (fontNormal != -1) {
        nvgFontFace(vg, "sans");
        nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);

        nvgFontSize(vg, 20.0f);
        nvgFillColor(vg, nvgRGBAf(0.72f, 0.86f, 0.90f, 0.95f * fade_alpha));
        nvgText(vg, width * 0.5f, panel_y + 56.0f, "SPEECH INPUT", nullptr);

        const float age = now - display_start_time;
        const bool show_result =
            !display_text.empty() && age >= 0.0f && age < kSpeechToTextTextLifetimeSec;
        if (show_result) {
            float t = 1.0f - (age / kSpeechToTextTextLifetimeSec);
            t = std::clamp(t, 0.0f, 1.0f);
            const float text_alpha = fade_alpha * t * t * (3.0f - 2.0f * t);
            NVGpaint text_gradient = nvgLinearGradient(
                vg,
                panel_x + 72.0f, panel_y + 76.0f,
                panel_x + panel_w - 72.0f, panel_y + panel_h - 24.0f,
                nvgRGBAf(0.35f, 0.98f, 0.86f, text_alpha),
                nvgRGBAf(0.96f, 1.00f, 1.00f, text_alpha));

            nvgFontSize(vg, 52.0f);
            nvgFillPaint(vg, text_gradient);
            nvgTextBox(vg, panel_x + 56.0f, panel_y + 88.0f, panel_w - 112.0f, display_text.c_str(), nullptr);
        } else {
            nvgFontSize(vg, 38.0f);
            nvgFillColor(vg, nvgRGBAf(0.78f, 0.84f, 0.88f, 0.60f * fade_alpha));
            nvgText(vg, width * 0.5f, panel_y + 108.0f, status_text.c_str(), nullptr);
        }
    }
}

void draw_hearing_assist_text_panel(
    NVGcontext* vg,
    float panel_x,
    float panel_y,
    float panel_w,
    float panel_h,
    const char* title,
    const std::string& display_text,
    const std::string& status_text,
    float accent_r,
    float accent_g,
    float accent_b,
    float fade_alpha,
    float visible_alpha,
    bool use_text_box,
    float confidence = -1.0f) {
    nvgBeginPath(vg);
    nvgRoundedRect(vg, panel_x, panel_y, panel_w, panel_h, 34.0f);
    nvgFillColor(vg, nvgRGBAf(0.04f, 0.07f, 0.09f, 0.90f * fade_alpha));
    nvgFill(vg);
    nvgStrokeColor(vg, nvgRGBAf(accent_r, accent_g, accent_b, 0.92f * fade_alpha));
    nvgStrokeWidth(vg, 3.5f);
    nvgStroke(vg);

    nvgBeginPath(vg);
    nvgCircle(vg, panel_x + 28.0f, panel_y + 28.0f, 6.0f);
    nvgFillColor(vg, nvgRGBAf(accent_r, accent_g, accent_b, 0.98f * fade_alpha));
    nvgFill(vg);

    nvgFontFace(vg, "sans");
    nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
    nvgFontSize(vg, 22.0f);
    nvgFillColor(vg, nvgRGBAf(0.76f, 0.88f, 0.92f, 0.95f * fade_alpha));
    nvgText(vg, panel_x + 46.0f, panel_y + 28.0f, title, nullptr);

    if (visible_alpha > 0.02f && !display_text.empty()) {
        NVGpaint text_gradient = nvgLinearGradient(
            vg,
            panel_x + 34.0f, panel_y + 72.0f,
            panel_x + panel_w - 34.0f, panel_y + panel_h - 26.0f,
            nvgRGBAf(accent_r, accent_g, accent_b, visible_alpha),
            nvgRGBAf(0.96f, 1.0f, 1.0f, visible_alpha));
        nvgFontSize(vg, 48.0f);
        nvgFillPaint(vg, text_gradient);
        if (use_text_box) {
            nvgTextBox(vg, panel_x + 34.0f, panel_y + 80.0f, panel_w - 68.0f, display_text.c_str(), nullptr);
        } else {
            nvgTextBox(vg, panel_x + 34.0f, panel_y + 80.0f, panel_w - 68.0f, display_text.c_str(), nullptr);
        }

        if (confidence >= 0.0f) {
            char conf_buf[64];
            snprintf(conf_buf, sizeof(conf_buf), "CONFIDENCE %.0f%%", confidence * 100.0f);
            nvgFontSize(vg, 18.0f);
            nvgFillColor(vg, nvgRGBAf(0.82f, 0.94f, 0.96f, 0.86f * visible_alpha));
            nvgTextAlign(vg, NVG_ALIGN_RIGHT | NVG_ALIGN_BOTTOM);
            nvgText(vg, panel_x + panel_w - 26.0f, panel_y + panel_h - 20.0f, conf_buf, nullptr);
        }
    } else {
        nvgFontSize(vg, 34.0f);
        nvgFillColor(vg, nvgRGBAf(0.78f, 0.84f, 0.88f, 0.62f * fade_alpha));
        nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
        nvgTextBox(vg, panel_x + 30.0f, panel_y + panel_h * 0.42f, panel_w - 60.0f, status_text.c_str(), nullptr);
    }
}

void render_hearing_assist_mode_nvg(NVGcontext* vg, float width, float height, int fontNormal, float state_time) {
    const float fade_alpha = std::min(1.0f, state_time / 0.35f);
    const float now = monotonic_time_sec();

    std::string sign_text;
    std::string sign_status;
    float sign_confidence = 0.0f;
    float sign_start = -1000.0f;
    bool sign_thinking = false;
    float sign_thinking_transition = -1000.0f;
    {
        std::lock_guard<std::mutex> lock(g_sign_translate_runtime.mtx);
        sign_text = g_sign_translate_runtime.display_text;
        sign_status = g_sign_translate_runtime.status_text;
        sign_confidence = g_sign_translate_runtime.display_confidence;
        sign_start = g_sign_translate_runtime.display_start_time;
        sign_thinking = g_sign_translate_runtime.backend_thinking;
        sign_thinking_transition = g_sign_translate_runtime.thinking_transition_time;
    }

    std::string speech_text;
    std::string speech_status;
    float speech_start = -1000.0f;
    {
        std::lock_guard<std::mutex> lock(g_speech_to_text_runtime.mtx);
        speech_text = g_speech_to_text_runtime.display_text;
        speech_status = g_speech_to_text_runtime.status_text;
        speech_start = g_speech_to_text_runtime.display_start_time;
    }

    nvgBeginPath(vg);
    nvgRect(vg, 0.0f, 0.0f, width, height);
    nvgFillColor(vg, nvgRGBAf(0.01f, 0.02f, 0.03f, 0.42f * fade_alpha));
    nvgFill(vg);

    const bool expanded_layout = width > (kUiWidthPx * 1.4f);
    const float panel_w = expanded_layout ? 780.0f : 600.0f;
    const float panel_h = expanded_layout ? 300.0f : 250.0f;
    const float top_margin = expanded_layout ? 62.0f : 72.0f;
    const float center_gap = expanded_layout ? 110.0f : 44.0f;
    const float sign_x = (width * 0.5f) - (center_gap * 0.5f) - panel_w;
    const float sign_y = top_margin;
    const float speech_x = (width * 0.5f) + (center_gap * 0.5f);
    const float speech_y = top_margin;

    float sign_visible_alpha = 0.0f;
    const float sign_age = now - sign_start;
    const bool show_sign = !sign_text.empty() && sign_age >= 0.0f && sign_age < kHearingAssistSignTextLifetimeSec;
    if (show_sign) {
        float t = 1.0f - (sign_age / kHearingAssistSignTextLifetimeSec);
        t = std::clamp(t, 0.0f, 1.0f);
        sign_visible_alpha = fade_alpha * t * t * (3.0f - 2.0f * t);
    } else if (sign_thinking) {
        const float thinking_t = std::clamp((now - sign_thinking_transition) / 0.18f, 0.0f, 1.0f);
        sign_visible_alpha = fade_alpha * (thinking_t * thinking_t * (3.0f - 2.0f * thinking_t));
        sign_status = "...";
    }

    float speech_visible_alpha = 0.0f;
    const float speech_age = now - speech_start;
    const bool show_speech = !speech_text.empty() && speech_age >= 0.0f && speech_age < kHearingAssistSpeechTextLifetimeSec;
    if (show_speech) {
        float t = 1.0f - (speech_age / kHearingAssistSpeechTextLifetimeSec);
        t = std::clamp(t, 0.0f, 1.0f);
        speech_visible_alpha = fade_alpha * t * t * (3.0f - 2.0f * t);
    }

    if (fontNormal != -1) {
        draw_hearing_assist_text_panel(
            vg,
            sign_x,
            sign_y,
            panel_w,
            panel_h,
            "SIGN TRANSLATE",
            sign_text,
            sign_status,
            0.34f,
            0.96f,
            0.88f,
            fade_alpha,
            sign_visible_alpha,
            true,
            sign_confidence);

        draw_hearing_assist_text_panel(
            vg,
            speech_x,
            speech_y,
            panel_w,
            panel_h,
            "SPEECH TO TEXT",
            speech_text,
            speech_status,
            0.30f,
            0.82f,
            1.00f,
            fade_alpha,
            speech_visible_alpha,
            true);
    }
}

void render_riding_radar_panel(NVGcontext* vg, float panel_x, float panel_y, float diameter, float time_sec) {
    const float now_sec = monotonic_time_sec();
    std::vector<RadarUiTarget> targets = snapshot_radar_targets(now_sec);

    const float cx = panel_x + diameter * 0.5f;
    const float cy = panel_y + diameter * 0.5f;
    const float radius = diameter * 0.5f;
    const float ring_radius = radius * 0.66f;
    const float sweep_angle = std::fmod(time_sec * 1.35f, static_cast<float>(M_PI) * 2.0f);

    nvgSave(vg);

    NVGpaint panel_glow = nvgRadialGradient(
        vg,
        cx,
        cy,
        radius * 0.18f,
        radius * 1.04f,
        nvgRGBAf(0.10f, 0.20f, 0.36f, 0.34f),
        nvgRGBAf(0.01f, 0.03f, 0.06f, 0.95f));
    nvgBeginPath(vg);
    nvgCircle(vg, cx, cy, radius);
    nvgFillPaint(vg, panel_glow);
    nvgFill(vg);
    nvgStrokeColor(vg, nvgRGBAf(0.36f, 0.74f, 1.0f, 0.94f));
    nvgStrokeWidth(vg, 3.2f);
    nvgStroke(vg);

    nvgBeginPath(vg);
    nvgCircle(vg, cx, cy, radius - 10.0f);
    nvgStrokeColor(vg, nvgRGBAf(0.16f, 0.28f, 0.44f, 0.64f));
    nvgStrokeWidth(vg, 1.4f);
    nvgStroke(vg);

    nvgSave(vg);
    nvgBeginPath(vg);
    nvgCircle(vg, cx, cy, radius - 2.0f);
    nvgScissor(vg, panel_x, panel_y, diameter, diameter);

    for (int i = 0; i < 5; ++i) {
        float t0 = sweep_angle - 0.14f * static_cast<float>(i);
        float t1 = sweep_angle - 0.14f * static_cast<float>(i + 1);
        float alpha = 0.18f - 0.028f * static_cast<float>(i);
        nvgBeginPath(vg);
        nvgMoveTo(vg, cx, cy);
        nvgArc(vg, cx, cy, radius - 6.0f, t0, t1, NVG_CW);
        nvgClosePath(vg);
        nvgFillColor(vg, nvgRGBAf(0.40f, 0.80f, 1.0f, std::max(0.0f, alpha)));
        nvgFill(vg);
    }
    nvgRestore(vg);

    for (int i = 1; i <= 4; ++i) {
        float deco_r = radius * (0.22f + 0.15f * static_cast<float>(i));
        nvgBeginPath(vg);
        nvgCircle(vg, cx, cy, deco_r);
        nvgStrokeColor(vg, nvgRGBAf(0.30f, 0.72f, 0.98f, i == 3 ? 0.64f : 0.22f));
        nvgStrokeWidth(vg, i == 3 ? 2.4f : 1.2f);
        nvgStroke(vg);
    }

    nvgBeginPath(vg);
    nvgMoveTo(vg, cx - radius + 18.0f, cy);
    nvgLineTo(vg, cx + radius - 18.0f, cy);
    nvgMoveTo(vg, cx, cy - radius + 18.0f);
    nvgLineTo(vg, cx, cy + radius - 18.0f);
    nvgStrokeColor(vg, nvgRGBAf(0.24f, 0.60f, 0.94f, 0.24f));
    nvgStrokeWidth(vg, 1.2f);
    nvgStroke(vg);

    nvgBeginPath(vg);
    nvgCircle(vg, cx, cy, radius * 0.08f);
    nvgFillColor(vg, nvgRGBAf(0.62f, 0.88f, 1.0f, 0.94f));
    nvgFill(vg);

    for (const auto& target : targets) {
        float age = now_sec - target.last_update_time;
        if (!target.active || age > kRadarLifetimeSec) continue;

        float fade = std::clamp(1.0f - age / kRadarLifetimeSec, 0.0f, 1.0f);
        fade = fade * fade * (3.0f - 2.0f * fade);
        float angle_rad = target.display_angle_deg * static_cast<float>(M_PI / 180.0f);
        float px = cx + std::sin(angle_rad) * ring_radius;
        float py = cy - std::cos(angle_rad) * ring_radius;

        NVGcolor marker = target.type == RadarSoundType::HORN
            ? nvgRGBAf(0.48f, 1.0f, 0.92f, fade)
            : nvgRGBAf(1.0f, 0.86f, 0.42f, fade);

        nvgBeginPath(vg);
        nvgCircle(vg, px, py, 22.0f + 7.0f * target.confidence);
        nvgFillColor(vg, nvgRGBAf(marker.r, marker.g, marker.b, 0.20f * fade));
        nvgFill(vg);

        nvgBeginPath(vg);
        nvgMoveTo(vg, cx, cy);
        nvgLineTo(vg, px, py);
        nvgStrokeColor(vg, nvgRGBAf(marker.r, marker.g, marker.b, 0.34f * fade));
        nvgStrokeWidth(vg, 2.0f);
        nvgStroke(vg);

        nvgBeginPath(vg);
        nvgCircle(vg, px, py, 8.0f + 5.0f * target.confidence);
        nvgFillColor(vg, nvgRGBAf(marker.r, marker.g, marker.b, 0.96f * fade));
        nvgFill(vg);

        nvgBeginPath(vg);
        nvgCircle(vg, px, py, 4.0f + 2.0f * target.confidence);
        nvgFillColor(vg, nvgRGBAf(0.96f, 0.99f, 1.0f, 0.98f * fade));
        nvgFill(vg);
    }

    nvgRestore(vg);
}

void render_riding_directional_haze(NVGcontext* vg, float view_x, float view_y, float width, float height) {
    const float now_sec = monotonic_time_sec();
    std::vector<RadarUiTarget> targets = snapshot_radar_targets(now_sec);

    float left_intensity = 0.0f;
    float right_intensity = 0.0f;

    for (const auto& target : targets) {
        float age = now_sec - target.last_update_time;
        if (!target.active || age > kRadarLifetimeSec) continue;

        float fade = std::clamp(1.0f - age / kRadarLifetimeSec, 0.0f, 1.0f);
        fade = fade * fade * (3.0f - 2.0f * fade);
        float strength = std::clamp(0.24f + target.confidence * 0.76f, 0.0f, 1.0f) * fade;

        float angle = normalize_angle_deg(target.display_angle_deg);
        if (std::sin(angle * static_cast<float>(M_PI / 180.0f)) < 0.0f) {
            left_intensity = std::max(left_intensity, strength);
        } else {
            right_intensity = std::max(right_intensity, strength);
        }
    }

    const float cx = view_x + width * 0.5f;
    const float cy = view_y + height * 0.5f;
    const float safe_margin_px = 24.0f;
    const float content_extent_px = 100.0f;
    const float base_ring_radius = std::min(
        width * 0.5f - (content_extent_px + safe_margin_px),
        height * 0.5f - (content_extent_px + safe_margin_px));

    auto ring_point = [&](float angle_deg, float radius) {
        const float angle_rad = angle_deg * static_cast<float>(M_PI / 180.0f);
        return Point2D{
            cx + std::sin(angle_rad) * radius,
            cy - std::cos(angle_rad) * radius
        };
    };

    auto draw_side_haze = [&](bool left_side, float intensity) {
        if (intensity <= 0.02f) return;

        const float outer_radius = base_ring_radius * 1.16f;
        const float fringe_radius = base_ring_radius * 0.96f;
        const float glow_radius = base_ring_radius * 1.10f;
        const float start_deg = left_side ? 180.0f : 0.0f;
        const float end_deg = left_side ? 360.0f : 180.0f;
        const float center_deg = left_side ? 270.0f : 90.0f;
        const int segments = 72;
        const float edge_x = left_side ? (cx - outer_radius) : (cx + outer_radius);
        const float spine_x = left_side ? (cx - base_ring_radius * 0.14f) : (cx + base_ring_radius * 0.14f);
        const float upper_y = cy - base_ring_radius * 0.58f;
        const float lower_y = cy + base_ring_radius * 0.58f;
        const float pulse = 0.78f + 0.22f * std::sin(now_sec * 6.4f);

        NVGpaint body_haze = nvgLinearGradient(
            vg,
            edge_x,
            cy,
            spine_x,
            cy,
            nvgRGBAf(0.54f, 0.84f, 1.0f, 0.20f * intensity),
            nvgRGBAf(0.54f, 0.84f, 1.0f, 0.0f));

        nvgBeginPath(vg);
        Point2D start_outer = ring_point(start_deg, outer_radius);
        nvgMoveTo(vg, start_outer.x, start_outer.y);
        for (int i = 1; i <= segments; ++i) {
            const float angle = start_deg + (end_deg - start_deg) * (static_cast<float>(i) / segments);
            Point2D p = ring_point(angle, outer_radius);
            nvgLineTo(vg, p.x, p.y);
        }
        if (left_side) {
            nvgBezierTo(
                vg,
                cx - base_ring_radius * 0.30f, upper_y,
                spine_x, cy - base_ring_radius * 0.18f,
                spine_x, cy);
            nvgBezierTo(
                vg,
                spine_x, cy + base_ring_radius * 0.18f,
                cx - base_ring_radius * 0.30f, lower_y,
                start_outer.x, start_outer.y);
        } else {
            nvgBezierTo(
                vg,
                cx + base_ring_radius * 0.30f, lower_y,
                spine_x, cy + base_ring_radius * 0.18f,
                spine_x, cy);
            nvgBezierTo(
                vg,
                spine_x, cy - base_ring_radius * 0.18f,
                cx + base_ring_radius * 0.30f, upper_y,
                start_outer.x, start_outer.y);
        }
        nvgClosePath(vg);
        nvgFillPaint(vg, body_haze);
        nvgFill(vg);

        for (int band = 0; band < 3; ++band) {
            const float band_outer = outer_radius - (outer_radius - fringe_radius) * (0.22f * band);
            nvgBeginPath(vg);
            Point2D band_start = ring_point(start_deg, band_outer);
            nvgMoveTo(vg, band_start.x, band_start.y);
            for (int i = 1; i <= segments; ++i) {
                const float angle = start_deg + (end_deg - start_deg) * (static_cast<float>(i) / segments);
                Point2D p = ring_point(angle, band_outer);
                nvgLineTo(vg, p.x, p.y);
            }
            if (left_side) {
                nvgBezierTo(
                    vg,
                    cx - base_ring_radius * (0.26f + 0.04f * band), upper_y,
                    spine_x + 8.0f * band, cy - base_ring_radius * (0.16f - 0.02f * band),
                    spine_x + 8.0f * band, cy);
                nvgBezierTo(
                    vg,
                    spine_x + 8.0f * band, cy + base_ring_radius * (0.16f - 0.02f * band),
                    cx - base_ring_radius * (0.26f + 0.04f * band), lower_y,
                    band_start.x, band_start.y);
            } else {
                nvgBezierTo(
                    vg,
                    cx + base_ring_radius * (0.26f + 0.04f * band), lower_y,
                    spine_x - 8.0f * band, cy + base_ring_radius * (0.16f - 0.02f * band),
                    spine_x - 8.0f * band, cy);
                nvgBezierTo(
                    vg,
                    spine_x - 8.0f * band, cy - base_ring_radius * (0.16f - 0.02f * band),
                    cx + base_ring_radius * (0.26f + 0.04f * band), upper_y,
                    band_start.x, band_start.y);
            }
            nvgClosePath(vg);
            float band_alpha = (0.060f - 0.013f * band) * intensity;
            nvgFillColor(vg, nvgRGBAf(0.70f, 0.90f, 1.0f, band_alpha));
            nvgFill(vg);
        }

        for (int i = 0; i < 14; ++i) {
            const float t0 = static_cast<float>(i) / 14.0f;
            const float t1 = static_cast<float>(i + 1) / 14.0f;
            const float a0 = start_deg + (end_deg - start_deg) * t0;
            const float a1 = start_deg + (end_deg - start_deg) * t1;
            const float center_weight = 1.0f - std::min(1.0f, std::fabs(((a0 + a1) * 0.5f) - center_deg) / 90.0f);
            const float segment_alpha = (0.05f + 0.12f * center_weight) * intensity * pulse;
            nvgBeginPath(vg);
            Point2D p0 = ring_point(a0, outer_radius);
            Point2D p1 = ring_point(a1, outer_radius);
            Point2D q1 = ring_point(a1, fringe_radius);
            Point2D q0 = ring_point(a0, fringe_radius);
            nvgMoveTo(vg, p0.x, p0.y);
            nvgLineTo(vg, p1.x, p1.y);
            nvgLineTo(vg, q1.x, q1.y);
            nvgLineTo(vg, q0.x, q0.y);
            nvgClosePath(vg);
            nvgFillColor(vg, nvgRGBAf(0.78f, 0.94f, 1.0f, segment_alpha));
            nvgFill(vg);
        }

        NVGpaint edge_glow = nvgRadialGradient(
            vg,
            edge_x,
            cy,
            base_ring_radius * 0.10f,
            glow_radius,
            nvgRGBAf(0.76f, 0.92f, 1.0f, 0.22f * intensity * pulse),
            nvgRGBAf(0.76f, 0.92f, 1.0f, 0.0f));
        nvgBeginPath(vg);
        nvgCircle(vg, edge_x, cy, glow_radius);
        nvgFillPaint(vg, edge_glow);
        nvgFill(vg);

        for (int i = 0; i < 4; ++i) {
            float ring_r = outer_radius - 6.0f - 9.0f * i;
            nvgBeginPath(vg);
            Point2D p0 = ring_point(start_deg, ring_r);
            nvgMoveTo(vg, p0.x, p0.y);
            for (int s = 1; s <= segments; ++s) {
                const float angle = start_deg + (end_deg - start_deg) * (static_cast<float>(s) / segments);
                Point2D p = ring_point(angle, ring_r);
                nvgLineTo(vg, p.x, p.y);
            }
            nvgStrokeColor(vg, nvgRGBAf(0.84f, 0.96f, 1.0f, (0.10f - 0.018f * i) * intensity * pulse));
            nvgStrokeWidth(vg, 1.3f + (i == 0 ? 0.8f : 0.0f));
            nvgStroke(vg);
        }
    };

    draw_side_haze(true, left_intensity);
    draw_side_haze(false, right_intensity);
}

// ==========================================
// 核心：NanoVG 导航仪表盘渲染器
// ==========================================
NVGcolor lerpColor(NVGcolor a, NVGcolor b, float t) {
    return nvgRGBAf(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t);
}

void world_to_screen(float px, float py, float origin_x, float origin_y, float heading, float cx, float cy, float scale, float& out_x, float& out_y) {
    float dx = px - origin_x;
    float dy = py - origin_y;
    float theta = -(heading - M_PI / 2.0f);
    float rx = dx * std::cos(theta) - dy * std::sin(theta);
    float ry = dx * std::sin(theta) + dy * std::cos(theta);
    out_x = cx + rx * scale;
    out_y = cy - ry * scale;
}

std::vector<Point2D> get_forward_path(float cx, float cy, float lookahead, const std::vector<Point2D>& route) {
    if (route.empty()) return {};
    float min_d = 1e9; size_t best_idx = 0;
    for (size_t i = 0; i < route.size() - 1; i++) {
        float mid_x = (route[i].x + route[i+1].x) / 2.0f;
        float mid_y = (route[i].y + route[i+1].y) / 2.0f;
        float d = std::hypot(mid_x - cx, mid_y - cy);
        if (d < min_d) { min_d = d; best_idx = i; }
    }
    std::vector<Point2D> pts = {{cx, cy}};
    float accum = 0.0f;
    for (size_t i = best_idx + 1; i < route.size(); i++) {
        Point2D p = route[i];
        Point2D last_p = pts.back();
        float dist = std::hypot(p.x - last_p.x, p.y - last_p.y);
        if (accum + dist > lookahead) {
            float ratio = (lookahead - accum) / dist;
            pts.push_back({last_p.x + (p.x - last_p.x) * ratio, last_p.y + (p.y - last_p.y) * ratio});
            break;
        } else {
            pts.push_back(p);
            accum += dist;
        }
    }
    return pts;
}

#if 0
void render_navigation_nvg(NVGcontext* vg, float w, float h, int fontNormal, float t_now) {
    std::lock_guard<std::mutex> lock(g_nav.mtx);
    float nav_now = monotonic_time_sec();
    const float ui_scale = 0.84f;
    const float ui_origin_x = w * 0.5f;
    const float ui_origin_y = h * 0.5f;
    
    // 断线保护
    if (g_nav.state != "STANDBY" && nav_now - g_nav.last_packet_time > 5.0f) {
        g_nav.state = "STANDBY";
        g_nav.state_timer = nav_now;
        g_nav.has_origin = false;
    }

    NVGcolor c_bg = nvgRGB(0, 0, 0);
    NVGcolor c_panel_bg = nvgRGB(28, 28, 30);
    NVGcolor c_border = nvgRGB(58, 58, 60);
    NVGcolor c_primary = nvgRGB(255, 255, 255);
    NVGcolor c_accent = nvgRGB(10, 132, 255);
    NVGcolor c_success = nvgRGB(52, 199, 89);
    NVGcolor c_route_base = nvgRGB(51, 51, 51);
    NVGcolor c_text_dim = nvgRGB(142, 142, 147);

    float pad = 30, gap = 20;
    nvgSave(vg);
    nvgTranslate(vg, ui_origin_x, ui_origin_y);
    nvgScale(vg, ui_scale, ui_scale);
    nvgTranslate(vg, -ui_origin_x, -ui_origin_y);

    auto draw_center_overlay = [&](float cx, float cy, const char* title, const char* subtitle, NVGcolor theme_c, NVGcolor c_text, NVGcolor c_pnl) {
        float box_w = 480, box_h = 160;
        nvgBeginPath(vg); nvgRoundedRect(vg, cx - box_w/2, cy - box_h/2, box_w, box_h, 24);
        nvgFillColor(vg, c_pnl); nvgFill(vg);
        nvgStrokeColor(vg, theme_c); nvgStrokeWidth(vg, 2.0f); nvgStroke(vg);
        
        nvgBeginPath(vg); nvgCircle(vg, cx, cy - box_h/2 + 30, 8); nvgFillColor(vg, theme_c); nvgFill(vg);
        
        nvgFontSize(vg, 36.0f); nvgFontFace(vg, "sans"); nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
        nvgFillColor(vg, c_primary); nvgText(vg, cx, cy + 10, title, NULL);
        nvgFontSize(vg, 20.0f); nvgFillColor(vg, c_text); nvgText(vg, cx, cy + 50, subtitle, NULL);
    };

    if (g_nav.state == "STANDBY") {
        nvgBeginPath(vg); nvgRoundedRect(vg, pad, pad, w - 2*pad, h - 2*pad, 40);
        nvgStrokeColor(vg, c_border); nvgStrokeWidth(vg, 4.0f); nvgStroke(vg);
        
        float view_cx = w / 2, view_cy = h / 2 + 100;
        float prog = (std::sin(t_now * 3.0f) + 1.0f) / 2.0f;
        float pulse_r = 24.0f + prog * 10.0f;
        
        nvgBeginPath(vg); nvgCircle(vg, view_cx, view_cy, pulse_r);
        nvgStrokeColor(vg, c_border); nvgStrokeWidth(vg, 2.0f); nvgStroke(vg);
        
        nvgBeginPath(vg); nvgMoveTo(vg, view_cx, view_cy - 18); nvgLineTo(vg, view_cx + 14, view_cy + 14);
        nvgLineTo(vg, view_cx, view_cy + 6); nvgLineTo(vg, view_cx - 14, view_cy + 14);
        nvgFillColor(vg, c_text_dim); nvgFill(vg); nvgStrokeColor(vg, c_border); nvgStroke(vg);
        
        draw_center_overlay(w/2, h/2 - 50, "AURA SYSTEM STANDBY", "Waiting for tactical navigation data...", c_text_dim, c_text_dim, c_panel_bg);
        nvgRestore(vg);
        return;
    }

    float trans_t = nav_now - g_nav.state_timer;
    float prog = 1.0f;
    if (g_nav.state == "TRANSITION_IN") {
        prog = std::min(trans_t / 1.5f, 1.0f);
        prog = 1.0f - std::pow(1.0f - prog, 3.0f); // ease_out_cubic
        if (prog == 1.0f) g_nav.state = "NAVIGATING";
    }

    NVGcolor cur_pnl = lerpColor(c_bg, c_panel_bg, prog);
    NVGcolor cur_brd = lerpColor(c_bg, c_border, prog);
    NVGcolor cur_pri = lerpColor(c_bg, c_primary, prog);
    NVGcolor cur_acc = lerpColor(c_bg, c_accent, prog);
    NVGcolor cur_suc = lerpColor(c_bg, c_success, prog);
    NVGcolor cur_txt = lerpColor(c_bg, c_text_dim, prog);
    NVGcolor cur_rte = lerpColor(c_bg, c_route_base, prog);

    if (g_nav.state == "NAVIGATING") {
        float angle_diff = std::fmod(g_nav.raw_heading - g_nav.smooth_heading + M_PI, 2 * M_PI) - M_PI;
        g_nav.smooth_heading += angle_diff * 0.08f;
    }

    // 1. CarPlay 主渲染
    float view_cx = w / 2, view_cy = h / 2 + 100 * prog;
    float scale = 0.5f + 2.5f * prog;
    
    // 背景路网
    if (g_nav.route_pts.size() >= 2) {
        nvgBeginPath(vg);
        float sx, sy;
        world_to_screen(g_nav.route_pts[0].x, g_nav.route_pts[0].y, g_nav.car_x, g_nav.car_y, g_nav.smooth_heading, view_cx, view_cy, scale, sx, sy);
        nvgMoveTo(vg, sx, sy);
        for (size_t i = 1; i < g_nav.route_pts.size(); ++i) {
            world_to_screen(g_nav.route_pts[i].x, g_nav.route_pts[i].y, g_nav.car_x, g_nav.car_y, g_nav.smooth_heading, view_cx, view_cy, scale, sx, sy);
            nvgLineTo(vg, sx, sy);
        }
        nvgStrokeColor(vg, cur_rte); nvgStrokeWidth(vg, 10.0f); nvgLineCap(vg, NVG_ROUND); nvgLineJoin(vg, NVG_ROUND); nvgStroke(vg);
    }

    // 蓝色引导流光路线
    std::vector<Point2D> active_pts = get_forward_path(g_nav.car_x, g_nav.car_y, 200, g_nav.route_pts);
    if (active_pts.size() >= 2) {
        std::vector<Point2D> scr_pts;
        for (auto p : active_pts) {
            float sx, sy; world_to_screen(p.x, p.y, g_nav.car_x, g_nav.car_y, g_nav.smooth_heading, view_cx, view_cy, scale, sx, sy);
            scr_pts.push_back({sx, sy});
        }
        
        float phase = std::fmod(t_now * 1.125f, 1.0f);
        for (size_t i = 0; i < scr_pts.size() - 1; i++) {
            float ratio = (float)i / std::max(1.0f, (float)scr_pts.size() - 2.0f);
            float dist_to_phase = std::fmod(phase - ratio + 1.0f, 1.0f);
            float wave = (dist_to_phase < 0.4f) ? (1.0f - dist_to_phase / 0.4f) : 0.0f;
            
            NVGcolor hl_color = nvgRGBAf(96.0f/255.0f * prog, 208.0f/255.0f * prog, 255.0f/255.0f * prog, 1.0f);
            NVGcolor seg_color = lerpColor(cur_acc, hl_color, wave);
            
            nvgBeginPath(vg); nvgMoveTo(vg, scr_pts[i].x, scr_pts[i].y); nvgLineTo(vg, scr_pts[i+1].x, scr_pts[i+1].y);
            nvgStrokeColor(vg, seg_color); nvgStrokeWidth(vg, 12.0f); nvgLineCap(vg, NVG_ROUND); nvgStroke(vg);
        }
        
        // 箭头头部
        if (scr_pts.size() >= 2) {
            Point2D last = scr_pts.back(), prev = scr_pts[scr_pts.size() - 2];
            float angle = std::atan2(last.y - prev.y, last.x - prev.x);
            float arr_s = 32.0f, arr_w = 20.0f;
            nvgBeginPath(vg);
            nvgMoveTo(vg, last.x + arr_s * 0.4f * std::cos(angle), last.y + arr_s * 0.4f * std::sin(angle));
            nvgLineTo(vg, last.x - arr_w * std::cos(angle - M_PI/6), last.y - arr_w * std::sin(angle - M_PI/6));
            nvgLineTo(vg, last.x - arr_s * 0.5f * std::cos(angle), last.y - arr_s * 0.5f * std::sin(angle));
            nvgLineTo(vg, last.x - arr_w * std::cos(angle + M_PI/6), last.y - arr_w * std::sin(angle + M_PI/6));
            nvgFillColor(vg, nvgRGBAf(96.0f/255.0f * prog, 208.0f/255.0f * prog, 255.0f/255.0f * prog, 1.0f)); nvgFill(vg);
        }
    }

    // 车身 Puck
    nvgBeginPath(vg); nvgCircle(vg, view_cx, view_cy, 24); nvgStrokeColor(vg, cur_acc); nvgStrokeWidth(vg, 2.0f); nvgStroke(vg);
    nvgBeginPath(vg); nvgMoveTo(vg, view_cx, view_cy - 18); nvgLineTo(vg, view_cx + 14, view_cy + 14);
    nvgLineTo(vg, view_cx, view_cy + 6); nvgLineTo(vg, view_cx - 14, view_cy + 14);
    nvgFillColor(vg, cur_pri); nvgFill(vg); nvgStrokeColor(vg, cur_brd); nvgStroke(vg);

    // =======================================
    // 浮动面板 UI
    // =======================================
    float box_w = 300, box_h = 200;
    float ui_offset_top = -150 * (1 - prog);
    float ui_offset_bot = 150 * (1 - prog);

    // 2. 微缩地图 (MiniMap)
    float mm_x1 = pad + gap, mm_y1 = pad + gap + ui_offset_top;
    nvgBeginPath(vg); nvgRoundedRect(vg, mm_x1, mm_y1, box_w, box_h, 20);
    nvgFillColor(vg, cur_pnl); nvgFill(vg); nvgStrokeColor(vg, cur_brd); nvgStrokeWidth(vg, 2.0f); nvgStroke(vg);
    
    float mm_cx = mm_x1 + box_w/2, mm_cy = mm_y1 + box_h/2;
    float mm_scale = 0.1f, route_cx = 0.0f, route_cy = 0.0f;
    if (!g_nav.route_pts.empty()) {
        float min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
        for (auto p : g_nav.route_pts) { min_x=std::min(min_x, p.x); max_x=std::max(max_x, p.x); min_y=std::min(min_y, p.y); max_y=std::max(max_y, p.y); }
        route_cx = (min_x + max_x)/2; route_cy = (min_y + max_y)/2;
        mm_scale = std::min((box_w - 40) / std::max(10.0f, max_x - min_x), (box_h - 40) / std::max(10.0f, max_y - min_y)) * 0.8f;
        
        nvgBeginPath(vg);
        float sx, sy; world_to_screen(g_nav.route_pts[0].x, g_nav.route_pts[0].y, route_cx, route_cy, M_PI/2, mm_cx, mm_cy, mm_scale, sx, sy);
        nvgMoveTo(vg, sx, sy);
        for (size_t i=1; i<g_nav.route_pts.size(); ++i) {
            world_to_screen(g_nav.route_pts[i].x, g_nav.route_pts[i].y, route_cx, route_cy, M_PI/2, mm_cx, mm_cy, mm_scale, sx, sy);
            nvgLineTo(vg, sx, sy);
        }
        nvgStrokeColor(vg, cur_rte); nvgStrokeWidth(vg, 4.0f); nvgStroke(vg);
    }
    float car_sx, car_sy; world_to_screen(g_nav.car_x, g_nav.car_y, route_cx, route_cy, M_PI/2, mm_cx, mm_cy, mm_scale, car_sx, car_sy);
    nvgBeginPath(vg); nvgCircle(vg, car_sx, car_sy, 6); nvgFillColor(vg, g_nav.state == "ARRIVED" ? cur_suc : cur_acc); nvgFill(vg);
    
    nvgFontSize(vg, 14.0f); nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
    nvgFillColor(vg, cur_txt); nvgText(vg, mm_x1 + 20, mm_y1 + 20, "OVERALL ROUTE", NULL);

    // 3. 转向指示器 (Turn Indicator)
    float trn_x1 = w - pad - gap - box_w, trn_y1 = pad + gap + ui_offset_top;
    nvgBeginPath(vg); nvgRoundedRect(vg, trn_x1, trn_y1, box_w, box_h, 20);
    nvgFillColor(vg, cur_pnl); nvgFill(vg); nvgStrokeColor(vg, cur_brd); nvgStroke(vg);
    
    float trn_cx = trn_x1 + box_w/2, trn_cy = trn_y1 + box_h/2 - 20;
    float target_turn = (g_nav.state != "ARRIVED" && g_nav.dist_to_turn <= 200.0f) ? g_nav.target_turn_type : 0.0f;
    g_nav.arrow_smooth += (target_turn - g_nav.arrow_smooth) * 0.15f;
    float val = g_nav.arrow_smooth;
    
    nvgBeginPath(vg); nvgMoveTo(vg, trn_cx, trn_cy + 35); nvgLineTo(vg, trn_cx, trn_cy);
    float p2x = trn_cx + 55 * val, p2y = trn_cy - 35 * (1 - std::abs(val));
    nvgLineTo(vg, p2x, p2y);
    nvgStrokeColor(vg, cur_pri); nvgStrokeWidth(vg, 16.0f); nvgLineCap(vg, NVG_BUTT); nvgLineJoin(vg, NVG_MITER); nvgStroke(vg);
    
    float arr_angle = -M_PI/2 + (M_PI/2) * val;
    nvgBeginPath(vg);
    nvgMoveTo(vg, p2x + 26 * std::cos(arr_angle), p2y + 26 * std::sin(arr_angle));
    nvgLineTo(vg, p2x + 20 * std::cos(arr_angle - M_PI/2), p2y + 20 * std::sin(arr_angle - M_PI/2));
    nvgLineTo(vg, p2x + 20 * std::cos(arr_angle + M_PI/2), p2y + 20 * std::sin(arr_angle + M_PI/2));
    nvgFillColor(vg, cur_pri); nvgFill(vg);

    nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    if (g_nav.state == "ARRIVED") {
        nvgFontSize(vg, 24.0f); nvgFillColor(vg, cur_suc); nvgText(vg, trn_cx, trn_y1 + box_h - 40, "ARRIVED", NULL);
    } else if (g_nav.dist_to_turn > 0 && g_nav.dist_to_turn <= 300) {
        char dist_buf[32]; snprintf(dist_buf, sizeof(dist_buf), "%dm", (int)g_nav.dist_to_turn);
        nvgFontSize(vg, 38.0f); nvgFillColor(vg, cur_pri); nvgText(vg, trn_cx, trn_y1 + box_h - 55, dist_buf, NULL);
        nvgFontSize(vg, 18.0f); nvgFillColor(vg, cur_acc); nvgText(vg, trn_cx, trn_y1 + box_h - 25, g_nav.action_str.c_str(), NULL);
    } else {
        nvgFontSize(vg, 20.0f); nvgFillColor(vg, cur_txt); nvgText(vg, trn_cx, trn_y1 + box_h - 40, "PROCEED", NULL);
    }

    // 4. 进度条 (Progress Bar)
    float pb_x1 = pad + gap, pb_x2 = w - pad - gap, pb_y = h - pad - gap - 30 + ui_offset_bot;
    nvgBeginPath(vg); nvgRoundedRect(vg, pb_x1 - 15, pb_y - 35, (pb_x2 - pb_x1) + 30, 60, 20); nvgFillColor(vg, cur_pnl); nvgFill(vg);
    nvgBeginPath(vg); nvgRoundedRect(vg, pb_x1, pb_y - 4, pb_x2 - pb_x1, 8, 4); nvgFillColor(vg, cur_brd); nvgFill(vg);
    
    float ratio = std::max(0.0f, std::min(1.0f, g_nav.current_dist / std::max(1.0f, g_nav.total_dist)));
    float fill_w = (pb_x2 - pb_x1) * ratio;
    NVGcolor bar_c = g_nav.state == "ARRIVED" ? cur_suc : cur_acc;
    if (fill_w > 8) { nvgBeginPath(vg); nvgRoundedRect(vg, pb_x1, pb_y - 4, fill_w, 8, 4); nvgFillColor(vg, bar_c); nvgFill(vg); }
    
    char txt_buf[64]; snprintf(txt_buf, sizeof(txt_buf), "DISTANCE: %dm / %dm", (int)g_nav.current_dist, (int)g_nav.total_dist);
    nvgFontSize(vg, 14.0f); nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_BOTTOM); nvgFillColor(vg, cur_txt); nvgText(vg, pb_x1, pb_y - 12, txt_buf, NULL);
    snprintf(txt_buf, sizeof(txt_buf), "%.1f%%", ratio * 100.0f);
    nvgTextAlign(vg, NVG_ALIGN_RIGHT | NVG_ALIGN_BOTTOM); nvgFillColor(vg, bar_c); nvgText(vg, pb_x2, pb_y - 12, txt_buf, NULL);

    // 外边框
    nvgBeginPath(vg); nvgRoundedRect(vg, pad, pad, w - 2*pad, h - 2*pad, 40); nvgStrokeColor(vg, cur_brd); nvgStrokeWidth(vg, 4.0f); nvgStroke(vg);

    if (g_nav.state == "ARRIVED") {
        std::string sub = "Destination: " + g_nav.dest_name;
        draw_center_overlay(w/2, h/2 - 50, "ARRIVED", sub.c_str(), cur_suc, cur_txt, cur_pnl);
    }
    nvgRestore(vg);
}
#endif

void render_navigation_nvg(NVGcontext* vg, float w, float h, int fontNormal, float t_now) {
    std::lock_guard<std::mutex> lock(g_nav.mtx);
    const float nav_now = monotonic_time_sec();
    const float ui_scale = 0.84f;
    const float ui_origin_x = w * 0.5f;
    const float ui_origin_y = h * 0.5f;

    if (g_nav.state != "STANDBY" && nav_now - g_nav.last_packet_time > 5.0f) {
        g_nav.state = "STANDBY";
        g_nav.state_timer = nav_now;
        g_nav.has_origin = false;
    }

    const NVGcolor c_bg = nvgRGB(0, 0, 0);
    const NVGcolor c_panel_bg = nvgRGB(28, 28, 30);
    const NVGcolor c_border = nvgRGB(58, 58, 60);
    const NVGcolor c_primary = nvgRGB(255, 255, 255);
    const NVGcolor c_accent = nvgRGB(10, 132, 255);
    const NVGcolor c_success = nvgRGB(52, 199, 89);
    const NVGcolor c_route_base = nvgRGB(51, 51, 51);
    const NVGcolor c_text_dim = nvgRGB(142, 142, 147);
    const float nav_brightness = 1.18f;
    auto boost_color = [&](NVGcolor color) {
        return nvgRGBAf(
            std::min(1.0f, color.r * nav_brightness),
            std::min(1.0f, color.g * nav_brightness),
            std::min(1.0f, color.b * nav_brightness),
            color.a
        );
    };
    const NVGcolor c_panel_bg_ui = boost_color(c_panel_bg);
    const NVGcolor c_border_ui = boost_color(c_border);
    const NVGcolor c_text_dim_ui = boost_color(c_text_dim);

    const float pad = 30.0f;
    const float gap = 20.0f;
    const float box_w = 300.0f;
    const float box_h = 200.0f;

    nvgSave(vg);
    nvgTranslate(vg, ui_origin_x, ui_origin_y);
    nvgScale(vg, ui_scale, ui_scale);
    nvgTranslate(vg, -ui_origin_x, -ui_origin_y);

    auto draw_center_overlay = [&](float cx, float cy, const char* title, const char* subtitle, NVGcolor theme_c, NVGcolor c_text, NVGcolor c_pnl) {
        const float overlay_w = 480.0f;
        const float overlay_h = 160.0f;
        nvgBeginPath(vg);
        nvgRoundedRect(vg, cx - overlay_w * 0.5f, cy - overlay_h * 0.5f, overlay_w, overlay_h, 24.0f);
        nvgFillColor(vg, c_pnl);
        nvgFill(vg);
        nvgStrokeColor(vg, theme_c);
        nvgStrokeWidth(vg, 2.0f);
        nvgStroke(vg);

        nvgBeginPath(vg);
        nvgCircle(vg, cx, cy - overlay_h * 0.5f + 30.0f, 8.0f);
        nvgFillColor(vg, theme_c);
        nvgFill(vg);

        nvgFontSize(vg, 36.0f);
        nvgFontFace(vg, "sans");
        nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
        nvgFillColor(vg, c_primary);
        nvgText(vg, cx, cy + 10.0f, title, NULL);
        nvgFontSize(vg, 20.0f);
        nvgFillColor(vg, c_text);
        nvgText(vg, cx, cy + 50.0f, subtitle, NULL);
    };

    if (g_nav.state == "STANDBY") {
        nvgBeginPath(vg);
        nvgRoundedRect(vg, pad, pad, w - 2.0f * pad, h - 2.0f * pad, 40.0f);
        nvgStrokeColor(vg, c_border_ui);
        nvgStrokeWidth(vg, 4.0f);
        nvgStroke(vg);

        const float view_cx = w * 0.5f;
        const float view_cy = h * 0.5f + 100.0f;
        const float pulse_t = (std::sin(t_now * 3.0f) + 1.0f) * 0.5f;
        const float pulse_r = 24.0f + pulse_t * 10.0f;

        nvgBeginPath(vg);
        nvgCircle(vg, view_cx, view_cy, pulse_r);
        nvgStrokeColor(vg, c_border_ui);
        nvgStrokeWidth(vg, 2.0f);
        nvgStroke(vg);

        nvgBeginPath(vg);
        nvgMoveTo(vg, view_cx, view_cy - 18.0f);
        nvgLineTo(vg, view_cx + 14.0f, view_cy + 14.0f);
        nvgLineTo(vg, view_cx, view_cy + 6.0f);
        nvgLineTo(vg, view_cx - 14.0f, view_cy + 14.0f);
        nvgFillColor(vg, c_text_dim_ui);
        nvgFill(vg);
        nvgStrokeColor(vg, c_border_ui);
        nvgStroke(vg);

        draw_center_overlay(w * 0.5f, h * 0.5f - 50.0f, "AURA SYSTEM STANDBY", "Waiting for tactical navigation data...", c_text_dim_ui, c_text_dim_ui, c_panel_bg_ui);
        nvgRestore(vg);
        return;
    }

    float trans_t = nav_now - g_nav.state_timer;
    float prog = 1.0f;
    if (g_nav.state == "TRANSITION_IN") {
        prog = std::min(trans_t / 1.5f, 1.0f);
        prog = 1.0f - std::pow(1.0f - prog, 3.0f);
        if (prog == 1.0f) g_nav.state = "NAVIGATING";
    }

    const NVGcolor cur_pnl = boost_color(lerpColor(c_bg, c_panel_bg, prog));
    const NVGcolor cur_brd = boost_color(lerpColor(c_bg, c_border, prog));
    const NVGcolor cur_pri = boost_color(lerpColor(c_bg, c_primary, prog));
    const NVGcolor cur_acc = boost_color(lerpColor(c_bg, c_accent, prog));
    const NVGcolor cur_suc = boost_color(lerpColor(c_bg, c_success, prog));
    const NVGcolor cur_txt = boost_color(lerpColor(c_bg, c_text_dim, prog));
    const NVGcolor cur_rte = boost_color(lerpColor(c_bg, c_route_base, prog));

    if (g_nav.state == "NAVIGATING") {
        float angle_diff = std::fmod(g_nav.raw_heading - g_nav.smooth_heading + static_cast<float>(M_PI), 2.0f * static_cast<float>(M_PI)) - static_cast<float>(M_PI);
        g_nav.smooth_heading += angle_diff * 0.08f;
    }

    const float position_smooth = 0.18f;
    const float metric_smooth = 0.20f;
    g_nav.display_car_x += (g_nav.car_x - g_nav.display_car_x) * position_smooth;
    g_nav.display_car_y += (g_nav.car_y - g_nav.display_car_y) * position_smooth;
    g_nav.display_current_dist +=
        (g_nav.current_dist - g_nav.display_current_dist) * metric_smooth;
    g_nav.display_dist_to_turn +=
        (g_nav.dist_to_turn - g_nav.display_dist_to_turn) * metric_smooth;

    const float ui_offset_top = -150.0f * (1.0f - prog);
    const float ui_offset_bot = 150.0f * (1.0f - prog);
    const float mm_x1 = pad + gap;
    const float mm_y1 = pad + gap + ui_offset_top;
    const float trn_x1 = w - pad - gap - box_w;
    const float trn_y1 = pad + gap + ui_offset_top;
    const float pb_x1 = pad + gap;
    const float pb_x2 = w - pad - gap;
    const float pb_y = h - pad - gap - 30.0f + ui_offset_bot;
    const float view_cx = w * 0.5f;
    const float view_cy = h * 0.5f + 100.0f * prog;
    const float scale = 0.5f + 2.5f * prog;
    const float route_clip_x = pad + 16.0f;
    const float route_clip_y = pad + 16.0f;
    const float route_clip_w = w - 2.0f * route_clip_x;
    const float route_clip_h = h - 2.0f * route_clip_y;

    nvgSave(vg);
    nvgScissor(vg, route_clip_x, route_clip_y, route_clip_w, route_clip_h);

    if (g_nav.route_pts.size() >= 2) {
        nvgBeginPath(vg);
        float sx = 0.0f, sy = 0.0f;
        world_to_screen(g_nav.route_pts[0].x, g_nav.route_pts[0].y, g_nav.display_car_x, g_nav.display_car_y, g_nav.smooth_heading, view_cx, view_cy, scale, sx, sy);
        nvgMoveTo(vg, sx, sy);
        for (size_t i = 1; i < g_nav.route_pts.size(); ++i) {
            world_to_screen(g_nav.route_pts[i].x, g_nav.route_pts[i].y, g_nav.display_car_x, g_nav.display_car_y, g_nav.smooth_heading, view_cx, view_cy, scale, sx, sy);
            nvgLineTo(vg, sx, sy);
        }
        nvgStrokeColor(vg, cur_rte);
        nvgStrokeWidth(vg, 10.0f);
        nvgLineCap(vg, NVG_ROUND);
        nvgLineJoin(vg, NVG_ROUND);
        nvgStroke(vg);
    }

    std::vector<Point2D> active_pts = get_forward_path(
        g_nav.display_car_x,
        g_nav.display_car_y,
        200.0f,
        g_nav.route_pts);
    if (active_pts.size() >= 2) {
        std::vector<Point2D> scr_pts;
        scr_pts.reserve(active_pts.size());
        for (const auto& p : active_pts) {
            float sx = 0.0f, sy = 0.0f;
            world_to_screen(p.x, p.y, g_nav.display_car_x, g_nav.display_car_y, g_nav.smooth_heading, view_cx, view_cy, scale, sx, sy);
            scr_pts.push_back({sx, sy});
        }

        const float phase = std::fmod(t_now * 1.125f, 1.0f);
        for (size_t i = 0; i + 1 < scr_pts.size(); ++i) {
            float ratio = static_cast<float>(i) / std::max(1.0f, static_cast<float>(scr_pts.size()) - 2.0f);
            float dist_to_phase = std::fmod(phase - ratio + 1.0f, 1.0f);
            float wave = (dist_to_phase < 0.4f) ? (1.0f - dist_to_phase / 0.4f) : 0.0f;
            NVGcolor hl_color = nvgRGBAf(96.0f / 255.0f * prog, 208.0f / 255.0f * prog, 255.0f / 255.0f * prog, 1.0f);
            NVGcolor seg_color = lerpColor(cur_acc, hl_color, wave);

            nvgBeginPath(vg);
            nvgMoveTo(vg, scr_pts[i].x, scr_pts[i].y);
            nvgLineTo(vg, scr_pts[i + 1].x, scr_pts[i + 1].y);
            nvgStrokeColor(vg, seg_color);
            nvgStrokeWidth(vg, 12.0f);
            nvgLineCap(vg, NVG_ROUND);
            nvgStroke(vg);
        }

        if (scr_pts.size() >= 2) {
            const Point2D last = scr_pts.back();
            const Point2D prev = scr_pts[scr_pts.size() - 2];
            const float angle = std::atan2(last.y - prev.y, last.x - prev.x);
            const float arr_s = 32.0f;
            const float arr_w = 20.0f;
            nvgBeginPath(vg);
            nvgMoveTo(vg, last.x + arr_s * 0.4f * std::cos(angle), last.y + arr_s * 0.4f * std::sin(angle));
            nvgLineTo(vg, last.x - arr_w * std::cos(angle - static_cast<float>(M_PI) / 6.0f), last.y - arr_w * std::sin(angle - static_cast<float>(M_PI) / 6.0f));
            nvgLineTo(vg, last.x - arr_s * 0.5f * std::cos(angle), last.y - arr_s * 0.5f * std::sin(angle));
            nvgLineTo(vg, last.x - arr_w * std::cos(angle + static_cast<float>(M_PI) / 6.0f), last.y - arr_w * std::sin(angle + static_cast<float>(M_PI) / 6.0f));
            nvgFillColor(vg, nvgRGBAf(96.0f / 255.0f * prog, 208.0f / 255.0f * prog, 255.0f / 255.0f * prog, 1.0f));
            nvgFill(vg);
        }
    }

    nvgBeginPath(vg);
    nvgCircle(vg, view_cx, view_cy, 24.0f);
    nvgStrokeColor(vg, cur_acc);
    nvgStrokeWidth(vg, 2.0f);
    nvgStroke(vg);
    nvgBeginPath(vg);
    nvgMoveTo(vg, view_cx, view_cy - 18.0f);
    nvgLineTo(vg, view_cx + 14.0f, view_cy + 14.0f);
    nvgLineTo(vg, view_cx, view_cy + 6.0f);
    nvgLineTo(vg, view_cx - 14.0f, view_cy + 14.0f);
    nvgFillColor(vg, cur_pri);
    nvgFill(vg);
    nvgStrokeColor(vg, cur_brd);
    nvgStroke(vg);
    nvgRestore(vg);

    nvgBeginPath(vg);
    nvgRoundedRect(vg, mm_x1, mm_y1, box_w, box_h, 20.0f);
    nvgFillColor(vg, cur_pnl);
    nvgFill(vg);
    nvgStrokeColor(vg, cur_brd);
    nvgStrokeWidth(vg, 2.0f);
    nvgStroke(vg);

    float mm_cx = mm_x1 + box_w * 0.5f;
    float mm_cy = mm_y1 + box_h * 0.5f;
    float mm_scale = 0.1f;
    float route_cx = 0.0f;
    float route_cy = 0.0f;
    nvgSave(vg);
    nvgScissor(vg, mm_x1 + 16.0f, mm_y1 + 16.0f, box_w - 32.0f, box_h - 32.0f);
    if (!g_nav.route_pts.empty()) {
        float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;
        for (const auto& p : g_nav.route_pts) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
        route_cx = (min_x + max_x) * 0.5f;
        route_cy = (min_y + max_y) * 0.5f;
        mm_scale = std::min((box_w - 40.0f) / std::max(10.0f, max_x - min_x), (box_h - 40.0f) / std::max(10.0f, max_y - min_y)) * 0.8f;

        nvgBeginPath(vg);
        float sx = 0.0f, sy = 0.0f;
        world_to_screen(g_nav.route_pts[0].x, g_nav.route_pts[0].y, route_cx, route_cy, static_cast<float>(M_PI) / 2.0f, mm_cx, mm_cy, mm_scale, sx, sy);
        nvgMoveTo(vg, sx, sy);
        for (size_t i = 1; i < g_nav.route_pts.size(); ++i) {
            world_to_screen(g_nav.route_pts[i].x, g_nav.route_pts[i].y, route_cx, route_cy, static_cast<float>(M_PI) / 2.0f, mm_cx, mm_cy, mm_scale, sx, sy);
            nvgLineTo(vg, sx, sy);
        }
        nvgStrokeColor(vg, cur_rte);
        nvgStrokeWidth(vg, 4.0f);
        nvgStroke(vg);
    }
    float car_sx = 0.0f, car_sy = 0.0f;
    world_to_screen(g_nav.display_car_x, g_nav.display_car_y, route_cx, route_cy, static_cast<float>(M_PI) / 2.0f, mm_cx, mm_cy, mm_scale, car_sx, car_sy);
    nvgBeginPath(vg);
    nvgCircle(vg, car_sx, car_sy, 6.0f);
    nvgFillColor(vg, g_nav.state == "ARRIVED" ? cur_suc : cur_acc);
    nvgFill(vg);
    nvgRestore(vg);

    nvgFontSize(vg, 14.0f);
    nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
    nvgFillColor(vg, cur_txt);
    nvgText(vg, mm_x1 + 20.0f, mm_y1 + 20.0f, "OVERALL ROUTE", NULL);

    nvgBeginPath(vg);
    nvgRoundedRect(vg, trn_x1, trn_y1, box_w, box_h, 20.0f);
    nvgFillColor(vg, cur_pnl);
    nvgFill(vg);
    nvgStrokeColor(vg, cur_brd);
    nvgStroke(vg);

    const float trn_cx = trn_x1 + box_w * 0.5f;
    const float trn_cy = trn_y1 + box_h * 0.5f - 20.0f;
    const float target_turn =
        (g_nav.state != "ARRIVED" && g_nav.display_dist_to_turn <= 200.0f)
            ? g_nav.target_turn_type
            : 0.0f;
    g_nav.arrow_smooth += (target_turn - g_nav.arrow_smooth) * 0.15f;
    const float val = g_nav.arrow_smooth;

    nvgBeginPath(vg);
    nvgMoveTo(vg, trn_cx, trn_cy + 35.0f);
    nvgLineTo(vg, trn_cx, trn_cy);
    const float p2x = trn_cx + 55.0f * val;
    const float p2y = trn_cy - 35.0f * (1.0f - std::abs(val));
    nvgLineTo(vg, p2x, p2y);
    nvgStrokeColor(vg, cur_pri);
    nvgStrokeWidth(vg, 16.0f);
    nvgLineCap(vg, NVG_BUTT);
    nvgLineJoin(vg, NVG_MITER);
    nvgStroke(vg);

    const float arr_angle = -static_cast<float>(M_PI) / 2.0f + (static_cast<float>(M_PI) / 2.0f) * val;
    nvgBeginPath(vg);
    nvgMoveTo(vg, p2x + 26.0f * std::cos(arr_angle), p2y + 26.0f * std::sin(arr_angle));
    nvgLineTo(vg, p2x + 20.0f * std::cos(arr_angle - static_cast<float>(M_PI) / 2.0f), p2y + 20.0f * std::sin(arr_angle - static_cast<float>(M_PI) / 2.0f));
    nvgLineTo(vg, p2x + 20.0f * std::cos(arr_angle + static_cast<float>(M_PI) / 2.0f), p2y + 20.0f * std::sin(arr_angle + static_cast<float>(M_PI) / 2.0f));
    nvgFillColor(vg, cur_pri);
    nvgFill(vg);

    nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    if (g_nav.state == "ARRIVED") {
        nvgFontSize(vg, 24.0f);
        nvgFillColor(vg, cur_suc);
        nvgText(vg, trn_cx, trn_y1 + box_h - 40.0f, "ARRIVED", NULL);
    } else if (g_nav.display_dist_to_turn > 0.0f && g_nav.display_dist_to_turn <= 300.0f) {
        char dist_buf[32];
        snprintf(dist_buf, sizeof(dist_buf), "%dm", static_cast<int>(g_nav.display_dist_to_turn));
        nvgFontSize(vg, 38.0f);
        nvgFillColor(vg, cur_pri);
        nvgText(vg, trn_cx, trn_y1 + box_h - 55.0f, dist_buf, NULL);
        nvgFontSize(vg, 18.0f);
        nvgFillColor(vg, cur_acc);
        nvgText(vg, trn_cx, trn_y1 + box_h - 25.0f, g_nav.action_str.c_str(), NULL);
    } else {
        nvgFontSize(vg, 20.0f);
        nvgFillColor(vg, cur_txt);
        nvgText(vg, trn_cx, trn_y1 + box_h - 40.0f, "PROCEED", NULL);
    }

    nvgBeginPath(vg);
    nvgRoundedRect(vg, pb_x1 - 15.0f, pb_y - 35.0f, (pb_x2 - pb_x1) + 30.0f, 60.0f, 20.0f);
    nvgFillColor(vg, cur_pnl);
    nvgFill(vg);
    nvgBeginPath(vg);
    nvgRoundedRect(vg, pb_x1, pb_y - 4.0f, pb_x2 - pb_x1, 8.0f, 4.0f);
    nvgFillColor(vg, cur_brd);
    nvgFill(vg);

    const float ratio = std::max(
        0.0f,
        std::min(1.0f, g_nav.display_current_dist / std::max(1.0f, g_nav.total_dist)));
    const float fill_w = (pb_x2 - pb_x1) * ratio;
    const NVGcolor bar_c = g_nav.state == "ARRIVED" ? cur_suc : cur_acc;
    if (fill_w > 8.0f) {
        nvgBeginPath(vg);
        nvgRoundedRect(vg, pb_x1, pb_y - 4.0f, fill_w, 8.0f, 4.0f);
        nvgFillColor(vg, bar_c);
        nvgFill(vg);
    }

    char txt_buf[64];
    snprintf(
        txt_buf,
        sizeof(txt_buf),
        "DISTANCE: %dm / %dm",
        static_cast<int>(g_nav.display_current_dist),
        static_cast<int>(g_nav.total_dist));
    nvgFontSize(vg, 14.0f);
    nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_BOTTOM);
    nvgFillColor(vg, cur_txt);
    nvgText(vg, pb_x1, pb_y - 12.0f, txt_buf, NULL);
    snprintf(txt_buf, sizeof(txt_buf), "%.1f%%", ratio * 100.0f);
    nvgTextAlign(vg, NVG_ALIGN_RIGHT | NVG_ALIGN_BOTTOM);
    nvgFillColor(vg, bar_c);
    nvgText(vg, pb_x2, pb_y - 12.0f, txt_buf, NULL);

    nvgBeginPath(vg);
    nvgRoundedRect(vg, pad, pad, w - 2.0f * pad, h - 2.0f * pad, 40.0f);
    nvgStrokeColor(vg, cur_brd);
    nvgStrokeWidth(vg, 4.0f);
    nvgStroke(vg);

    if (g_nav.state == "ARRIVED") {
        std::string sub = "Destination: " + g_nav.dest_name;
        draw_center_overlay(w * 0.5f, h * 0.5f - 50.0f, "ARRIVED", sub.c_str(), cur_suc, cur_txt, cur_pnl);
    }
    nvgRestore(vg);
}

void render_riding_mode_nvg(NVGcontext* vg, float width, float height, int fontNormal, float anim_time, float state_time) {
    const float fade_alpha = std::min(1.0f, state_time / 0.35f);

    cv::Mat rgba_frame;
    RearCameraSettings settings;
    bool frame_ready = false;
    {
        std::lock_guard<std::mutex> lock(g_rear_camera_runtime.mtx);
        rgba_frame = g_rear_camera_runtime.latest_rgba;
        settings = g_rear_camera_runtime.settings;
        frame_ready = g_rear_camera_runtime.frame_ready && !rgba_frame.empty();
    }

    const float view_width = kUiWidthPx;
    const float view_height = kUiHeightPx;
    const float canvas_width = view_width * kRidingVirtualCanvasScale;
    const float canvas_height = view_height * kRidingVirtualCanvasScale;
    const float window_x = (canvas_width - view_width) * 0.5f;
    const float window_y = (canvas_height - view_height) * 0.5f;

    const float nav_outer_left = kNavigationOuterFramePadPx;
    const float nav_outer_top = kNavigationOuterFramePadPx;
    const float nav_draw_x = kRidingNavigationLeftMarginPx;
    const float nav_draw_y = kRidingNavigationTopMarginPx;
    const float nav_top_y = nav_draw_y + (nav_outer_top * kRidingNavigationScale);

    const float camera_w = std::min(settings.display_width_px * kRidingRearCameraScale, view_width - 40.0f);
    const float camera_h = std::min(settings.display_height_px * kRidingRearCameraScale, view_height - 40.0f);
    const float camera_x = window_x + (view_width - camera_w) * 0.5f;
    const float camera_y = nav_top_y;
    const float radar_diameter = std::min(camera_h * 1.42f, 432.0f);
    const float radar_x = std::min(
        camera_x + camera_w + kRidingRadarGapPx,
        width - radar_diameter - 24.0f);
    const float radar_y = nav_top_y;

    render_riding_directional_haze(vg, window_x, window_y, view_width, view_height);

    nvgSave(vg);
    nvgTranslate(vg, nav_draw_x, nav_draw_y);
    nvgScale(vg, kRidingNavigationScale, kRidingNavigationScale);
    nvgTranslate(vg, -nav_outer_left, -nav_outer_top);
    render_navigation_nvg(vg, view_width, view_height, fontNormal, anim_time);
    nvgRestore(vg);

    static int nvg_rear_image_riding = -1;
    static int image_w_riding = 0;
    static int image_h_riding = 0;

    draw_rear_camera_panel(
        vg,
        rgba_frame,
        frame_ready,
        camera_x,
        camera_y,
        camera_w,
        camera_h,
        fade_alpha * 0.95f,
        28.0f,
        3.5f,
        nvg_rear_image_riding,
        image_w_riding,
        image_h_riding);

    render_riding_radar_panel(vg, radar_x, radar_y, radar_diameter, anim_time);
}

GLuint compileShader(GLenum type, const char* source) { GLuint s = glCreateShader(type); glShaderSource(s, 1, &source, nullptr); glCompileShader(s); return s; }

int main() {
    cleanup_stale_runtime_helpers();

    std::thread input_thread(terminal_input_thread);
    input_thread.detach(); 
    std::thread uart_input_thread(uart_button_input_thread);
    uart_input_thread.detach();
    std::thread bluetooth_thread(bluetooth_navigation_thread);
    bluetooth_thread.detach();
    std::thread rear_camera_thread(rear_camera_capture_thread);
    rear_camera_thread.detach();
    std::thread radar_backend(radar_backend_thread);
    radar_backend.detach();
    std::thread radar_service(radar_service_supervisor_thread);
    radar_service.detach();
    std::thread sign_translate_service(sign_translate_supervisor_thread);
    sign_translate_service.detach();
    std::thread speech_to_text_service(speech_to_text_supervisor_thread);
    speech_to_text_service.detach();
    std::thread sign_translate_tts(sign_translate_tts_thread);
    sign_translate_tts.detach();

    srand(static_cast<unsigned int>(time(nullptr)));

    int fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::cerr << "[FATAL] 无法打开 /dev/dri/card0，请检查是否有 root 权限 (sudo)" << std::endl;
        return -1;
    }
    
    drmModeRes *resources = drmModeGetResources(fd);
    if (!resources || resources->count_connectors == 0) {
        std::cerr << "[FATAL] 无法获取 DRM 资源，未检测到显示输出架构！" << std::endl;
        return -1;
    }
    
    uint32_t best_connector_id = 0, best_crtc_id = 0; 
    drmModeModeInfo best_mode_info;
    int max_resolution_area = 0;

    for (int i = 0; i < resources->count_connectors; i++) {
        drmModeConnector *conn = drmModeGetConnector(fd, resources->connectors[i]);
        if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
            int current_area = conn->modes[0].hdisplay * conn->modes[0].vdisplay;
            if (current_area > max_resolution_area) {
                max_resolution_area = current_area;
                best_connector_id = conn->connector_id; 
                best_mode_info = conn->modes[0];
                uint32_t temp_crtc_id = 0;
                if (conn->encoder_id) {
                    drmModeEncoder *enc = drmModeGetEncoder(fd, conn->encoder_id);
                    if (enc && enc->crtc_id) temp_crtc_id = enc->crtc_id;
                    if (enc) drmModeFreeEncoder(enc);
                }
                if (temp_crtc_id == 0) {
                    for (int j = 0; j < conn->count_encoders; j++) {
                        drmModeEncoder *enc = drmModeGetEncoder(fd, conn->encoders[j]);
                        if (enc) {
                            for (int k = 0; k < resources->count_crtcs; k++) { 
                                if (enc->possible_crtcs & (1 << k)) { temp_crtc_id = resources->crtcs[k]; break; } 
                            }
                            drmModeFreeEncoder(enc); if (temp_crtc_id != 0) break;
                        }
                    }
                }
                best_crtc_id = temp_crtc_id;
            }
        }
        drmModeFreeConnector(conn);
    }
    
    if (best_connector_id == 0 || best_crtc_id == 0) {
        std::cerr << "[FATAL] 硬件管道匹配失败！未找到任何激活的显示接口。" << std::endl;
        return -1;
    }

    uint32_t connector_id = best_connector_id;
    uint32_t crtc_id = best_crtc_id;
    drmModeModeInfo mode_info = best_mode_info;

    struct gbm_device *gbm = gbm_create_device(fd);
    struct gbm_surface *gs = gbm_surface_create(gbm, mode_info.hdisplay, mode_info.vdisplay, GBM_FORMAT_XRGB8888, GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
    EGLDisplay display = eglGetDisplay((EGLNativeDisplayType)gbm);
    eglInitialize(display, nullptr, nullptr); eglBindAPI(EGL_OPENGL_ES_API);
    EGLConfig config = nullptr; EGLint num_configs;
    const EGLint attribs[] = { EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_STENCIL_SIZE, 8, EGL_NONE };
    EGLConfig configs[64]; eglChooseConfig(display, attribs, configs, 64, &num_configs);
    for (int i = 0; i < num_configs; i++) { EGLint vid; eglGetConfigAttrib(display, configs[i], EGL_NATIVE_VISUAL_ID, &vid); if (vid == GBM_FORMAT_XRGB8888) { config = configs[i]; break; } }
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, (const EGLint[]){ EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE });
    EGLSurface surface = eglCreateWindowSurface(display, config, (EGLNativeWindowType)gs, nullptr);
    eglMakeCurrent(display, surface, surface, context);

    NVGcontext* vg = nvgCreateGLES2(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
    int fontNormal = nvgCreateFont(vg, "sans", "./font.ttf");
    if (fontNormal == -1) {
        std::cerr << "[WARNING] 找不到字体文件 ./font.ttf，文字(特别是中文)可能无法显示！" << std::endl;
    }

    const int ui_width = 1280, ui_height = 720;
    const int riding_ui_width = static_cast<int>(std::round(ui_width * kRidingVirtualCanvasScale));
    const int riding_ui_height = static_cast<int>(std::round(ui_height * kRidingVirtualCanvasScale));

    GLuint fbo, fbo_texture;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &fbo_texture);
    glBindTexture(GL_TEXTURE_2D, fbo_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ui_width, ui_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture, 0);
    GLuint stencil_buf;
    glGenRenderbuffers(1, &stencil_buf);
    glBindRenderbuffer(GL_RENDERBUFFER, stencil_buf);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX8, ui_width, ui_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, stencil_buf);

    GLuint riding_fbo, riding_fbo_texture;
    glGenFramebuffers(1, &riding_fbo);
    glGenTextures(1, &riding_fbo_texture);
    glBindTexture(GL_TEXTURE_2D, riding_fbo_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, riding_ui_width, riding_ui_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, riding_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, riding_fbo_texture, 0);
    GLuint riding_stencil_buf;
    glGenRenderbuffers(1, &riding_stencil_buf);
    glBindRenderbuffer(GL_RENDERBUFFER, riding_stencil_buf);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX8, riding_ui_width, riding_ui_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, riding_stencil_buf);

    json hud_calibration;
    if (!loadCalibrationJson("calibration.json", hud_calibration)) {
        return -1;
    }

    EyeMeshData leftMesh, rightMesh, leftMeshRiding, rightMeshRiding;
    if (!generateDistortionMeshesHud50cmFromCalibration(hud_calibration, leftMesh, rightMesh)) {
        return -1;
    }
    if (!generateDistortionMeshesHudFromCalibration(
            hud_calibration,
            leftMeshRiding,
            rightMeshRiding,
            kRidingHudPanelScale)) {
        return -1;
    }
    unsigned long long applied_shear_revision = g_symmetric_shear_revision.load(std::memory_order_relaxed);

    GLuint left_vbo, left_ebo, right_vbo, right_ebo;
    GLuint left_vbo_riding, left_ebo_riding, right_vbo_riding, right_ebo_riding;
    glGenBuffers(1, &left_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, left_vbo);
    glBufferData(GL_ARRAY_BUFFER, leftMesh.vertices.size() * sizeof(VertexData), leftMesh.vertices.data(), GL_DYNAMIC_DRAW);
    glGenBuffers(1, &left_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, left_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, leftMesh.indices.size() * sizeof(unsigned short), leftMesh.indices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &right_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, right_vbo);
    glBufferData(GL_ARRAY_BUFFER, rightMesh.vertices.size() * sizeof(VertexData), rightMesh.vertices.data(), GL_DYNAMIC_DRAW);
    glGenBuffers(1, &right_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, right_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, rightMesh.indices.size() * sizeof(unsigned short), rightMesh.indices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &left_vbo_riding);
    glBindBuffer(GL_ARRAY_BUFFER, left_vbo_riding);
    glBufferData(GL_ARRAY_BUFFER, leftMeshRiding.vertices.size() * sizeof(VertexData), leftMeshRiding.vertices.data(), GL_DYNAMIC_DRAW);
    glGenBuffers(1, &left_ebo_riding);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, left_ebo_riding);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, leftMeshRiding.indices.size() * sizeof(unsigned short), leftMeshRiding.indices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &right_vbo_riding);
    glBindBuffer(GL_ARRAY_BUFFER, right_vbo_riding);
    glBufferData(GL_ARRAY_BUFFER, rightMeshRiding.vertices.size() * sizeof(VertexData), rightMeshRiding.vertices.data(), GL_DYNAMIC_DRAW);
    glGenBuffers(1, &right_ebo_riding);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, right_ebo_riding);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, rightMeshRiding.indices.size() * sizeof(unsigned short), rightMeshRiding.indices.data(), GL_STATIC_DRAW);
    
    const char* vShaderStr = "attribute vec4 a_pos; attribute vec2 a_uv; varying vec2 v_uv; void main() { gl_Position = vec4(-a_pos.x, -a_pos.y, a_pos.z, a_pos.w); v_uv = a_uv; }";
    const char* fShaderStr =
        "precision mediump float; "
        "varying vec2 v_uv; "
        "uniform sampler2D s_tex; "
        "uniform vec2 u_uv_center; "
        "uniform vec2 u_uv_scale; "
        "void main() { "
        "  vec2 sample_uv = (v_uv - u_uv_center) / u_uv_scale + u_uv_center; "
        "  if(sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) gl_FragColor = vec4(0.0,0.0,0.0,1.0); "
        "  else gl_FragColor = texture2D(s_tex, sample_uv); "
        "}";
    GLuint programObject = glCreateProgram(); glAttachShader(programObject, compileShader(GL_VERTEX_SHADER, vShaderStr)); glAttachShader(programObject, compileShader(GL_FRAGMENT_SHADER, fShaderStr)); glLinkProgram(programObject);

    cv::VideoCapture boot_cap("c8a38f809278a782619bea9682dacb80.mp4");
    cv::Mat boot_frame, boot_rgba;
    int nvg_boot_img = -1;
    double video_fps = 30.0;
    float frame_duration = 1.0f / 30.0f;
    float last_video_time = 0.0f;
    bool has_boot_video = false;
    
    if (boot_cap.isOpened()) {
        video_fps = boot_cap.get(cv::CAP_PROP_FPS);
        if (video_fps <= 0) video_fps = 30.0;
        frame_duration = 1.0f / video_fps;
        
        boot_cap.read(boot_frame);
        if (!boot_frame.empty()) {
            cv::cvtColor(boot_frame, boot_rgba, cv::COLOR_BGR2RGBA);
            nvg_boot_img = nvgCreateImageRGBA(vg, boot_frame.cols, boot_frame.rows, 0, boot_rgba.data);
            has_boot_video = true;
        }
    } else {
        std::cout << "[WARNING] 找不到开机视频 c8a38f809278a782619bea9682dacb80.mp4, 将跳过开机动画。\n";
        g_target_state = SysState::STANDBY;
    }

    struct gbm_bo *prev_bo = nullptr; uint32_t prev_fb_id = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    SysState current_state = g_target_state.load();
    float state_enter_time = 0.0f;

    set_sign_translate_background_capture(false);
    std::cout << ">>> 显示引擎启动成功，开始渲染循环...\n";

    while (true) {
        float anim_time = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start_time).count();

        if (current_state != g_target_state.load()) {
            current_state = g_target_state.load();
            state_enter_time = anim_time; 
            last_video_time = 0.0f; 
        }
        
        float state_time = anim_time - state_enter_time; 

        if (current_state == SysState::BOOT_PLAY && has_boot_video) {
            while (state_time - last_video_time >= frame_duration) {
                cv::Mat temp_frame;
                boot_cap >> temp_frame;
                if (temp_frame.empty()) {
                    g_target_state = SysState::BOOT_FADE;
                    break;
                } else {
                    boot_frame = temp_frame.clone(); 
                    last_video_time += frame_duration;
                }
            }
        } else if (current_state == SysState::BOOT_PLAY && !has_boot_video) {
            g_target_state = SysState::BOOT_WAIT;
        }
        
        if (current_state == SysState::BOOT_FADE) {
            if (state_time > 0.2f) g_target_state = SysState::BOOT_WAIT;
        } else if (current_state == SysState::BOOT_WAIT) {
            if (state_time > 0.5f) g_target_state = SysState::DESKTOP;
        }

        const unsigned long long pending_shear_revision =
            g_symmetric_shear_revision.load(std::memory_order_relaxed);
        if (pending_shear_revision != applied_shear_revision) {
            if (generateDistortionMeshesHud50cmFromCalibration(hud_calibration, leftMesh, rightMesh) &&
                generateDistortionMeshesHudFromCalibration(
                    hud_calibration,
                    leftMeshRiding,
                    rightMeshRiding,
                    kRidingHudPanelScale)) {
                glBindBuffer(GL_ARRAY_BUFFER, left_vbo);
                glBufferSubData(
                    GL_ARRAY_BUFFER,
                    0,
                    leftMesh.vertices.size() * sizeof(VertexData),
                    leftMesh.vertices.data());
                glBindBuffer(GL_ARRAY_BUFFER, right_vbo);
                glBufferSubData(
                    GL_ARRAY_BUFFER,
                    0,
                    rightMesh.vertices.size() * sizeof(VertexData),
                    rightMesh.vertices.data());
                glBindBuffer(GL_ARRAY_BUFFER, left_vbo_riding);
                glBufferSubData(
                    GL_ARRAY_BUFFER,
                    0,
                    leftMeshRiding.vertices.size() * sizeof(VertexData),
                    leftMeshRiding.vertices.data());
                glBindBuffer(GL_ARRAY_BUFFER, right_vbo_riding);
                glBufferSubData(
                    GL_ARRAY_BUFFER,
                    0,
                    rightMeshRiding.vertices.size() * sizeof(VertexData),
                    rightMeshRiding.vertices.data());
                applied_shear_revision = pending_shear_revision;
            }
        }

        const bool expanded_mode_state =
            current_state == SysState::APP_RIDING_MODE ||
            current_state == SysState::APP_HEARING_ASSIST;
        const GLuint active_fbo = expanded_mode_state ? riding_fbo : fbo;
        const GLuint active_fbo_texture = expanded_mode_state ? riding_fbo_texture : fbo_texture;
        const int active_ui_width = expanded_mode_state ? riding_ui_width : ui_width;
        const int active_ui_height = expanded_mode_state ? riding_ui_height : ui_height;

        glBindFramebuffer(GL_FRAMEBUFFER, active_fbo); glViewport(0, 0, active_ui_width, active_ui_height);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        nvgBeginFrame(vg, active_ui_width, active_ui_height, 1.0f); 
        nvgTranslate(vg, active_ui_width * 0.5f, active_ui_height * 0.5f);
        nvgRotate(vg, static_cast<float>(M_PI));
        nvgTranslate(vg, -active_ui_width * 0.5f, -active_ui_height * 0.5f);
        
        if (current_state == SysState::STANDBY) {
        }
        else if (current_state == SysState::BOOT_PLAY || current_state == SysState::BOOT_FADE) {
            if (has_boot_video && !boot_frame.empty()) {
                cv::cvtColor(boot_frame, boot_rgba, cv::COLOR_BGR2RGBA);
                nvgUpdateImage(vg, nvg_boot_img, boot_rgba.data);
                
                float scale = std::min((float)ui_width / boot_frame.cols, (float)ui_height / boot_frame.rows) * 1.8f;
                float draw_w = boot_frame.cols * scale;
                float draw_h = boot_frame.rows * scale;
                float draw_x = (ui_width - draw_w) / 2.0f;
                float draw_y = (ui_height - draw_h) / 2.0f;
                
                NVGpaint imgPaint = nvgImagePattern(vg, draw_x, draw_y, draw_w, draw_h, 0.0f, nvg_boot_img, 1.0f);
                nvgBeginPath(vg); nvgRect(vg, draw_x, draw_y, draw_w, draw_h); nvgFillPaint(vg, imgPaint); nvgFill(vg);
                
                if (current_state == SysState::BOOT_FADE) {
                    float fade_alpha = std::min(1.0f, state_time / 0.2f);
                    nvgBeginPath(vg); nvgRect(vg, 0, 0, ui_width, ui_height); nvgFillColor(vg, nvgRGBAf(0.0f, 0.0f, 0.0f, fade_alpha)); nvgFill(vg);
                }
            }
        } 
        else if (current_state == SysState::DESKTOP) { render_desktop_nvg(vg, active_ui_width, active_ui_height, fontNormal, state_time); }
        else if (current_state == SysState::APP_RADAR) { render_sound_radar_nvg(vg, anim_time, active_ui_width / 2.0f, active_ui_height / 2.0f, state_time); }
        else if (current_state == SysState::APP_SIGN_TRANSLATE) { render_sign_translate_nvg(vg, active_ui_width, active_ui_height, fontNormal, state_time); }
        else if (current_state == SysState::APP_REAR_CAMERA) { render_rear_camera_nvg(vg, active_ui_width, active_ui_height, fontNormal, state_time); }
        else if (current_state == SysState::APP_NAVIGATION) { render_navigation_nvg(vg, active_ui_width, active_ui_height, fontNormal, anim_time); }
        else if (current_state == SysState::APP_SPEECH_TO_TEXT) { render_speech_to_text_nvg(vg, active_ui_width, active_ui_height, fontNormal, state_time); }
        else if (current_state == SysState::APP_HEARING_ASSIST) { render_hearing_assist_mode_nvg(vg, active_ui_width, active_ui_height, fontNormal, state_time); }
        else if (current_state == SysState::APP_RIDING_MODE) { render_riding_mode_nvg(vg, active_ui_width, active_ui_height, fontNormal, anim_time, state_time); }
        else if (current_state == SysState::APP_PLACEHOLDER) { render_placeholder_nvg(vg, active_ui_width, active_ui_height, fontNormal, state_time); }

        nvgEndFrame(vg);

        glBindFramebuffer(GL_FRAMEBUFFER, 0); glViewport(0, 0, mode_info.hdisplay, mode_info.vdisplay);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(programObject);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, active_fbo_texture);
        glUniform1i(glGetUniformLocation(programObject, "s_tex"), 0);
        GLint uvCenterLoc = glGetUniformLocation(programObject, "u_uv_center");
        GLint uvScaleLoc = glGetUniformLocation(programObject, "u_uv_scale");
        glUniform2f(uvCenterLoc, 0.5f, 0.5f);
        if (current_state == SysState::BOOT_PLAY || current_state == SysState::BOOT_FADE) {
            glUniform2f(uvScaleLoc, kBootUvScale, kBootUvScale);
        } else {
            glUniform2f(uvScaleLoc, 1.0f, 1.0f);
        }

        const GLuint active_left_vbo = expanded_mode_state ? left_vbo_riding : left_vbo;
        const GLuint active_left_ebo = expanded_mode_state ? left_ebo_riding : left_ebo;
        const GLuint active_right_vbo = expanded_mode_state ? right_vbo_riding : right_vbo;
        const GLuint active_right_ebo = expanded_mode_state ? right_ebo_riding : right_ebo;
        const GLsizei active_left_index_count = static_cast<GLsizei>(
            expanded_mode_state ? leftMeshRiding.indices.size() : leftMesh.indices.size());
        const GLsizei active_right_index_count = static_cast<GLsizei>(
            expanded_mode_state ? rightMeshRiding.indices.size() : rightMesh.indices.size());

        glBindBuffer(GL_ARRAY_BUFFER, active_left_vbo);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, x));
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, u));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, active_left_ebo);
        glDrawElements(GL_TRIANGLES, active_left_index_count, GL_UNSIGNED_SHORT, 0);

        glBindBuffer(GL_ARRAY_BUFFER, active_right_vbo);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, x));
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, u));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, active_right_ebo);
        glDrawElements(GL_TRIANGLES, active_right_index_count, GL_UNSIGNED_SHORT, 0);

        eglSwapBuffers(display, surface);
        struct gbm_bo *bo = gbm_surface_lock_front_buffer(gs);
        if (!bo) break;

        uint32_t handle = gbm_bo_get_handle(bo).u32, pitch = gbm_bo_get_stride(bo), fb_id;
        drmModeAddFB(fd, mode_info.hdisplay, mode_info.vdisplay, 24, 32, pitch, handle, &fb_id);
        drmModeSetCrtc(fd, crtc_id, fb_id, 0, 0, &connector_id, 1, &mode_info);

        if (prev_bo) { drmModeRmFB(fd, prev_fb_id); gbm_surface_release_buffer(gs, prev_bo); }
        prev_bo = bo; prev_fb_id = fb_id;
    }
    
    if (boot_cap.isOpened()) boot_cap.release();
    std::remove(kArHudPidFile);
    nvgDeleteGLES2(vg);
    return 0;
}
