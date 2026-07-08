param(
    [string]$TargetPath = "D:\Flutter_projects\ar_app1_hud_fixed_v2.cpp"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $TargetPath)) {
    throw "Target file not found: $TargetPath"
}

$content = Get-Content -LiteralPath $TargetPath -Raw -Encoding UTF8

if ($content -notmatch "#include <bluetooth/bluetooth.h>") {
    $content = [System.Text.RegularExpressions.Regex]::Replace(
        $content,
        "#include <arpa/inet.h>\s*#include <nlohmann/json.hpp>",
        "#include <arpa/inet.h>`r`n#include <bluetooth/bluetooth.h>`r`n#include <bluetooth/rfcomm.h>`r`n#include <bluetooth/sdp.h>`r`n#include <bluetooth/sdp_lib.h>`r`n#include <nlohmann/json.hpp>",
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )
}

$replacementBlock = @'
constexpr uint8_t kBluetoothNavChannel = 1;

float monotonic_time_sec() {
    return std::chrono::duration<float>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
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

void gps_to_local(double lng, double lat, float& x, float& y) {
    double rad_lat = g_nav.origin_lat * M_PI / 180.0;
    x = static_cast<float>((lng - g_nav.origin_lng) * 111320.0 * std::cos(rad_lat));
    y = static_cast<float>((lat - g_nav.origin_lat) * 111320.0);
}

void apply_navigation_payload(const json& payload) {
    std::lock_guard<std::mutex> lock(g_nav.mtx);

    float t_now = monotonic_time_sec();
    g_nav.last_packet_time = t_now;

    std::string status = json_to_string(payload, "status", "standby");
    if (status == "standby") {
        if (g_nav.state != "STANDBY") {
            g_nav.state = "STANDBY";
            g_nav.state_timer = t_now;
        }
        g_nav.has_origin = false;
        g_nav.route_pts.clear();
        g_nav.last_path_str.clear();
        g_nav.current_dist = 0.0f;
        g_nav.total_dist = 1.0f;
        g_nav.dist_to_turn = 9999.0f;
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
        g_nav.state_timer = t_now;
        g_nav.smooth_heading = g_nav.raw_heading;
    }

    if (g_nav.state == "NAVIGATING" && remain < 15.0f) {
        g_nav.state = "ARRIVED";
        g_nav.state_timer = t_now;
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
'@

$regexOptions = [System.Text.RegularExpressions.RegexOptions]::Singleline
$content = [System.Text.RegularExpressions.Regex]::Replace(
    $content,
    "(?:constexpr uint8_t kBluetoothNavChannel = 1;.*?|void gps_to_local\(.*?)(?=void terminal_input_thread\(\))",
    $replacementBlock + "`r`n",
    $regexOptions
)

$content = $content.Replace(
    "    std::thread udp_thread(udp_navigation_thread);",
    "    std::thread bluetooth_thread(bluetooth_navigation_thread);"
)
$content = $content.Replace(
    "    udp_thread.detach();",
    "    bluetooth_thread.detach();"
)

$content = $content.Replace(
    "void render_navigation_nvg(NVGcontext* vg, float w, float h, int fontNormal, float t_now) {`r`n    std::lock_guard<std::mutex> lock(g_nav.mtx);",
    "void render_navigation_nvg(NVGcontext* vg, float w, float h, int fontNormal, float t_now) {`r`n    std::lock_guard<std::mutex> lock(g_nav.mtx);`r`n    float nav_now = monotonic_time_sec();"
)
$content = $content.Replace(
    "    if (g_nav.state != ""STANDBY"" && t_now - g_nav.last_packet_time > 5.0f) {`r`n        g_nav.state = ""STANDBY"";`r`n        g_nav.state_timer = t_now;",
    "    if (g_nav.state != ""STANDBY"" && nav_now - g_nav.last_packet_time > 5.0f) {`r`n        g_nav.state = ""STANDBY"";`r`n        g_nav.state_timer = nav_now;"
)
$content = $content.Replace(
    "    float trans_t = t_now - g_nav.state_timer;",
    "    float trans_t = nav_now - g_nav.state_timer;"
)

Copy-Item -LiteralPath $TargetPath -Destination ($TargetPath + ".bak_bt") -Force
Set-Content -LiteralPath $TargetPath -Value $content -Encoding UTF8
