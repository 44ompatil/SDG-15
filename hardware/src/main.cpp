#include <cinttypes>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "board_config.h"

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef SDG15_WIFI_SSID
#define SDG15_WIFI_SSID ""
#endif
#ifndef SDG15_WIFI_PASSWORD
#define SDG15_WIFI_PASSWORD ""
#endif
#ifndef SDG15_UPLOAD_URL
#define SDG15_UPLOAD_URL ""
#endif
#ifndef SDG15_UPLOAD_TOKEN
#define SDG15_UPLOAD_TOKEN ""
#endif

#include "esp_camera.h"
#include "driver/gpio.h"
#include "esp_event.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_sntp.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/err.h"
#include "lwip/sys.h"
#include "nvs_flash.h"
namespace {

constexpr char kTag[] = "sdg15";
constexpr EventBits_t kWifiConnectedBit = BIT0;
constexpr int64_t kUnixTimeMinimum = 1700000000;

enum class ThreatLevel {
    kUnknown,
    kLow,
    kMedium,
    kHigh,
};

struct HttpResponseBody {
    char data[256] = {};
    size_t length = 0;
    bool overflow = false;
};

EventGroupHandle_t wifi_event_group = nullptr;
bool wifi_connected = false;
bool sntp_started = false;
bool pir_warmed_up = false;
bool previous_pir_state = false;
bool high_threat_alerted_for_motion = false;
uint32_t event_counter = 0;
uint64_t boot_time_ms = 0;

void setOutputLevel(gpio_num_t pin, int level) {
    const esp_err_t err = gpio_set_level(pin, level);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "GPIO %d output update failed: %s", static_cast<int>(pin), esp_err_to_name(err));
    }
}

void updateThreatIndicators(ThreatLevel level) {
    setOutputLevel(LED_LOW_THREAT_GPIO, level == ThreatLevel::kLow ? 1 : 0);
    setOutputLevel(LED_MEDIUM_THREAT_GPIO, level == ThreatLevel::kMedium ? 1 : 0);
    setOutputLevel(LED_HIGH_THREAT_GPIO, level == ThreatLevel::kHigh ? 1 : 0);
}

bool initializeIndicators() {
    gpio_config_t config = {};
    config.pin_bit_mask =
        (1ULL << LED_LOW_THREAT_GPIO) |
        (1ULL << LED_MEDIUM_THREAT_GPIO) |
        (1ULL << LED_HIGH_THREAT_GPIO) |
        (1ULL << BUZZER_GPIO);
    config.mode = GPIO_MODE_OUTPUT;
    config.pull_up_en = GPIO_PULLUP_DISABLE;
    config.pull_down_en = GPIO_PULLDOWN_DISABLE;
    config.intr_type = GPIO_INTR_DISABLE;
    const esp_err_t err = gpio_config(&config);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "Indicator GPIO setup failed: %s", esp_err_to_name(err));
        return false;
    }

    updateThreatIndicators(ThreatLevel::kUnknown);
    setOutputLevel(BUZZER_GPIO, 0);
    return true;
}

void beepForHighThreat() {
    setOutputLevel(BUZZER_GPIO, 1);
    vTaskDelay(pdMS_TO_TICKS(100));
    setOutputLevel(BUZZER_GPIO, 0);
}

esp_err_t captureResponseHandler(esp_http_client_event_t *event) {
    if (event == nullptr || event->event_id != HTTP_EVENT_ON_DATA ||
        event->user_data == nullptr || event->data_len <= 0) {
        return ESP_OK;
    }

    auto *response = static_cast<HttpResponseBody *>(event->user_data);
    const size_t data_length = static_cast<size_t>(event->data_len);
    if (data_length >= sizeof(response->data) - response->length) {
        response->overflow = true;
        return ESP_OK;
    }

    std::memcpy(response->data + response->length, event->data, data_length);
    response->length += data_length;
    response->data[response->length] = '\0';
    return ESP_OK;
}

bool parseThreatLevel(const HttpResponseBody &response, ThreatLevel &level) {
    if (response.overflow || response.length == 0) {
        ESP_LOGE(kTag, "Classifier response is empty or exceeds the response buffer");
        return false;
    }

    const char *field = std::strstr(response.data, "\"threat_level\"");
    const char *value = field == nullptr ? nullptr : std::strchr(field, ':');
    if (value == nullptr) {
        ESP_LOGE(kTag, "Classifier response has no threat level");
        return false;
    }

    ++value;
    while (*value != '\0' && std::isspace(static_cast<unsigned char>(*value))) {
        ++value;
    }
    if (*value != '"') {
        ESP_LOGE(kTag, "Classifier response has an invalid threat level");
        return false;
    }
    ++value;

    bool valid = true;
    if (std::strncmp(value, "low\"", sizeof("low\"") - 1) == 0) {
        level = ThreatLevel::kLow;
    } else if (std::strncmp(value, "medium\"", sizeof("medium\"") - 1) == 0) {
        level = ThreatLevel::kMedium;
    } else if (std::strncmp(value, "high\"", sizeof("high\"") - 1) == 0) {
        level = ThreatLevel::kHigh;
    } else {
        valid = false;
    }

    if (!valid) {
        ESP_LOGE(kTag, "Classifier response has no valid threat level");
    }
    return valid;
}

void wifiEventHandler(void *, esp_event_base_t event_base, int32_t event_id, void *) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        const esp_err_t err = esp_wifi_connect();
        if (err != ESP_OK) {
            ESP_LOGE(kTag, "Initial Wi-Fi connect failed: %s", esp_err_to_name(err));
        }
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_connected = false;
        xEventGroupClearBits(wifi_event_group, kWifiConnectedBit);
        const esp_err_t err = esp_wifi_connect();
        if (err != ESP_OK) {
            ESP_LOGE(kTag, "Wi-Fi reconnect failed: %s", esp_err_to_name(err));
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        wifi_connected = true;
        xEventGroupSetBits(wifi_event_group, kWifiConnectedBit);
        ESP_LOGI(kTag, "Wi-Fi connected");
        if (!sntp_started) {
            esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
            esp_sntp_setservername(0, "pool.ntp.org");
            esp_sntp_init();
            sntp_started = true;
        }
    }
}

bool initializeWifi() {
    if (SDG15_WIFI_SSID[0] == '\0' || SDG15_UPLOAD_URL[0] == '\0' ||
        SDG15_UPLOAD_TOKEN[0] == '\0') {
        ESP_LOGE(
            kTag,
            "Configure Wi-Fi, upload URL, and token in include/secrets.h; "
            "camera captures cannot reach the host without these settings");
        return false;
    }

    esp_err_t err = nvs_flash_init();
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "NVS init failed: %s", esp_err_to_name(err));
        return false;
    }
    err = esp_netif_init();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(kTag, "Network interface init failed: %s", esp_err_to_name(err));
        return false;
    }
    err = esp_event_loop_create_default();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(kTag, "Event loop init failed: %s", esp_err_to_name(err));
        return false;
    }
    if (esp_netif_create_default_wifi_sta() == nullptr) {
        ESP_LOGE(kTag, "Could not create default Wi-Fi station interface");
        return false;
    }
    wifi_init_config_t init_config = WIFI_INIT_CONFIG_DEFAULT();
    err = esp_wifi_init(&init_config);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "Wi-Fi init failed: %s", esp_err_to_name(err));
        return false;
    }

    wifi_event_group = xEventGroupCreate();
    if (wifi_event_group == nullptr) {
        ESP_LOGE(kTag, "Could not create Wi-Fi event group");
        return false;
    }

    err = esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifiEventHandler, nullptr);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "Wi-Fi event registration failed: %s", esp_err_to_name(err));
        return false;
    }
    err = esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifiEventHandler, nullptr);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "IP event registration failed: %s", esp_err_to_name(err));
        return false;
    }

    wifi_config_t station_config = {};
    std::strncpy(
        reinterpret_cast<char *>(station_config.sta.ssid),
        SDG15_WIFI_SSID,
        sizeof(station_config.sta.ssid) - 1);
    std::strncpy(
        reinterpret_cast<char *>(station_config.sta.password),
        SDG15_WIFI_PASSWORD,
        sizeof(station_config.sta.password) - 1);

    err = esp_wifi_set_mode(WIFI_MODE_STA);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "Setting station mode failed: %s", esp_err_to_name(err));
        return false;
    }
    err = esp_wifi_set_config(WIFI_IF_STA, &station_config);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "Setting Wi-Fi credentials failed: %s", esp_err_to_name(err));
        return false;
    }
    err = esp_wifi_start();
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "Wi-Fi start failed: %s", esp_err_to_name(err));
        return false;
    }

    const EventBits_t bits = xEventGroupWaitBits(
        wifi_event_group,
        kWifiConnectedBit,
        pdFALSE,
        pdFALSE,
        pdMS_TO_TICKS(20000));
    if ((bits & kWifiConnectedBit) == 0) {
        ESP_LOGE(kTag, "Wi-Fi connection timed out; will keep retrying in the background");
        return false;
    }

    setenv("TZ", "UTC0", 1);
    tzset();
    return true;
}

bool initializeCamera() {
    camera_config_t config = {};
    config.pin_pwdn = CAMERA_PIN_PWDN;
    config.pin_reset = CAMERA_PIN_RESET;
    config.pin_xclk = CAMERA_PIN_XCLK;
    config.pin_sccb_sda = CAMERA_PIN_SIOD;
    config.pin_sccb_scl = CAMERA_PIN_SIOC;
    config.pin_d0 = CAMERA_PIN_D0;
    config.pin_d1 = CAMERA_PIN_D1;
    config.pin_d2 = CAMERA_PIN_D2;
    config.pin_d3 = CAMERA_PIN_D3;
    config.pin_d4 = CAMERA_PIN_D4;
    config.pin_d5 = CAMERA_PIN_D5;
    config.pin_d6 = CAMERA_PIN_D6;
    config.pin_d7 = CAMERA_PIN_D7;
    config.pin_vsync = CAMERA_PIN_VSYNC;
    config.pin_href = CAMERA_PIN_HREF;
    config.pin_pclk = CAMERA_PIN_PCLK;
    config.xclk_freq_hz = 20000000;
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    const esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "Camera init failed: %s", esp_err_to_name(err));
        return false;
    }
    return true;
}

bool initializePir() {
    gpio_config_t config = {};
    config.pin_bit_mask = 1ULL << PIR_GPIO;
    config.mode = GPIO_MODE_INPUT;
    config.pull_up_en = GPIO_PULLUP_DISABLE;
    config.pull_down_en = GPIO_PULLDOWN_ENABLE;
    config.intr_type = GPIO_INTR_DISABLE;
    const esp_err_t err = gpio_config(&config);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "PIR GPIO setup failed: %s", esp_err_to_name(err));
        return false;
    }
    return true;
}

void makeTimestamp(char *buffer, size_t size) {
    const std::time_t now = std::time(nullptr);
    if (now < kUnixTimeMinimum) {
        buffer[0] = '\0';
        return;
    }
    std::tm utc_time = {};
    gmtime_r(&now, &utc_time);
    std::strftime(buffer, size, "%Y-%m-%dT%H:%M:%SZ", &utc_time);
}

void makeEventId(char *buffer, size_t size, const char *timestamp, uint64_t uptime_ms) {
    ++event_counter;
    if (timestamp[0] != '\0') {
        std::tm utc_time = {};
        const std::time_t now = std::time(nullptr);
        gmtime_r(&now, &utc_time);
        char date[20] = {};
        std::strftime(date, sizeof(date), "%Y%m%dT%H%M%S", &utc_time);
        std::snprintf(buffer, size, "%s-%03" PRIu32, date, event_counter % 1000);
    } else {
        std::snprintf(
            buffer,
            size,
            "boot-%" PRIu64 "-%" PRIu32,
            uptime_ms,
            event_counter);
    }
}

bool uploadCapture(
    camera_fb_t *frame,
    const char *event_id,
    const char *timestamp,
    uint64_t uptime_ms,
    ThreatLevel &threat_level) {
    if (!wifi_connected) {
        ESP_LOGE(kTag, "Capture %s was not uploaded: Wi-Fi is disconnected", event_id);
        return false;
    }

    char metadata[256] = {};
    const int written = std::snprintf(
        metadata,
        sizeof(metadata),
        "{\"event_id\":\"%s\",\"captured_at\":\"%s\","
        "\"uptime_ms\":%" PRIu64 ",\"frame_width\":%u,\"frame_height\":%u,"
        "\"pir_gpio\":%d,\"trigger_source\":\"HC-SR501\","
        "\"camera_sensor\":\"OV3660\"}",
        event_id,
        timestamp,
        uptime_ms,
        frame->width,
        frame->height,
        static_cast<int>(PIR_GPIO));
    if (written < 0 || static_cast<size_t>(written) >= sizeof(metadata)) {
        ESP_LOGE(kTag, "Could not serialize capture metadata");
        return false;
    }

    char authorization[160] = {};
    std::snprintf(authorization, sizeof(authorization), "Bearer %s", SDG15_UPLOAD_TOKEN);
    esp_http_client_config_t http_config = {};
    http_config.url = SDG15_UPLOAD_URL;
    http_config.timeout_ms = 15000;
    http_config.buffer_size = 1024;
    http_config.buffer_size_tx = 1024;
    HttpResponseBody response = {};
    http_config.event_handler = captureResponseHandler;
    http_config.user_data = &response;
    esp_http_client_handle_t client = esp_http_client_init(&http_config);
    if (client == nullptr) {
        ESP_LOGE(kTag, "Could not initialize HTTP client");
        return false;
    }

    esp_err_t err = esp_http_client_set_method(client, HTTP_METHOD_POST);
    if (err == ESP_OK) {
        err = esp_http_client_set_header(client, "Content-Type", "image/jpeg");
    }
    if (err == ESP_OK) {
        err = esp_http_client_set_header(client, "Authorization", authorization);
    }
    if (err == ESP_OK) {
        err = esp_http_client_set_header(client, "X-Capture", metadata);
    }
    if (err == ESP_OK) {
        err = esp_http_client_set_post_field(
            client,
            reinterpret_cast<const char *>(frame->buf),
            static_cast<int>(frame->len));
    }
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "Could not prepare capture upload: %s", esp_err_to_name(err));
        esp_http_client_cleanup(client);
        return false;
    }

    err = esp_http_client_perform(client);
    const int status = esp_http_client_get_status_code(client);
    esp_http_client_cleanup(client);
    if (err != ESP_OK || status < 200 || status >= 300) {
        ESP_LOGE(
            kTag,
            "Upload for %s failed (HTTP %d): %s",
            event_id,
            status,
            esp_err_to_name(err));
        return false;
    }

    if (!parseThreatLevel(response, threat_level)) {
        return false;
    }

    ESP_LOGI(
        kTag,
        "Uploaded capture %s (%u bytes); threat level=%s",
        event_id,
        frame->len,
        threat_level == ThreatLevel::kHigh
            ? "high"
            : threat_level == ThreatLevel::kMedium ? "medium" : "low");
    return true;
}

bool uploadMotionAlert(const char *event_id, const char *timestamp, uint64_t uptime_ms) {
    if (!wifi_connected) {
        ESP_LOGE(kTag, "Motion alert %s was not uploaded: Wi-Fi is disconnected", event_id);
        return false;
    }

    constexpr char kCapturePath[] = "/capture";
    constexpr char kMotionPath[] = "/motion";
    const size_t upload_url_length = std::strlen(SDG15_UPLOAD_URL);
    const size_t capture_path_length = sizeof(kCapturePath) - 1;
    if (upload_url_length < capture_path_length ||
        std::strcmp(
            SDG15_UPLOAD_URL + upload_url_length - capture_path_length,
            kCapturePath) != 0) {
        ESP_LOGE(kTag, "Upload URL must end in /capture to derive the motion alert endpoint");
        return false;
    }

    char motion_url[sizeof(SDG15_UPLOAD_URL)] = {};
    const size_t base_length = upload_url_length - capture_path_length;
    std::memcpy(motion_url, SDG15_UPLOAD_URL, base_length);
    std::memcpy(motion_url + base_length, kMotionPath, sizeof(kMotionPath));

    char metadata[192] = {};
    const int written = std::snprintf(
        metadata,
        sizeof(metadata),
        "{\"event_id\":\"%s\",\"captured_at\":\"%s\","
        "\"uptime_ms\":%" PRIu64 ",\"pir_gpio\":%d,"
        "\"trigger_source\":\"HC-SR501\"}",
        event_id,
        timestamp,
        uptime_ms,
        static_cast<int>(PIR_GPIO));
    if (written < 0 || static_cast<size_t>(written) >= sizeof(metadata)) {
        ESP_LOGE(kTag, "Could not serialize motion alert metadata");
        return false;
    }

    char authorization[160] = {};
    std::snprintf(authorization, sizeof(authorization), "Bearer %s", SDG15_UPLOAD_TOKEN);
    esp_http_client_config_t http_config = {};
    http_config.url = motion_url;
    http_config.timeout_ms = 1000;
    http_config.buffer_size = 1024;
    http_config.buffer_size_tx = 1024;
    esp_http_client_handle_t client = esp_http_client_init(&http_config);
    if (client == nullptr) {
        ESP_LOGE(kTag, "Could not initialize motion alert HTTP client");
        return false;
    }

    esp_err_t err = esp_http_client_set_method(client, HTTP_METHOD_POST);
    if (err == ESP_OK) {
        err = esp_http_client_set_header(client, "Content-Type", "application/json");
    }
    if (err == ESP_OK) {
        err = esp_http_client_set_header(client, "Authorization", authorization);
    }
    if (err == ESP_OK) {
        err = esp_http_client_set_post_field(client, metadata, written);
    }
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "Could not prepare motion alert upload: %s", esp_err_to_name(err));
        esp_http_client_cleanup(client);
        return false;
    }

    err = esp_http_client_perform(client);
    const int status = esp_http_client_get_status_code(client);
    esp_http_client_cleanup(client);
    if (err != ESP_OK || status < 200 || status >= 300) {
        ESP_LOGE(
            kTag,
            "Motion alert upload failed (HTTP %d): %s",
            status,
            esp_err_to_name(err));
        return false;
    }

    ESP_LOGI(kTag, "Motion alert %s saved by receiver", event_id);
    return true;
}

void captureAndUpload(uint64_t uptime_ms) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) {
        ESP_LOGE(kTag, "Camera returned no frame");
        return;
    }
    if (frame->format != PIXFORMAT_JPEG || frame->width == 0 || frame->height == 0) {
        ESP_LOGE(kTag, "Camera returned an invalid JPEG frame");
        esp_camera_fb_return(frame);
        return;
    }

    char timestamp[32] = {};
    char event_id[48] = {};
    makeTimestamp(timestamp, sizeof(timestamp));
    makeEventId(event_id, sizeof(event_id), timestamp, uptime_ms);
    ESP_LOGI(kTag, "Capturing motion frame event=%s", event_id);
    ThreatLevel threat_level = ThreatLevel::kUnknown;
    if (uploadCapture(frame, event_id, timestamp, uptime_ms, threat_level)) {
        updateThreatIndicators(threat_level);
        if (threat_level == ThreatLevel::kHigh && !high_threat_alerted_for_motion) {
            beepForHighThreat();
            high_threat_alerted_for_motion = true;
        }
    }
    esp_camera_fb_return(frame);
}

}  // namespace

extern "C" void app_main(void) {
    boot_time_ms = static_cast<uint64_t>(esp_timer_get_time() / 1000);

    if (!initializeIndicators()) {
        ESP_LOGE(kTag, "Indicator setup failed");
        while (true) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    if (!initializeCamera()) {
        ESP_LOGE(kTag, "Camera setup failed; PIR capture is disabled");
        while (true) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    if (!initializePir()) {
        ESP_LOGE(kTag, "PIR setup failed; motion capture is disabled");
        while (true) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    initializeWifi();
    ESP_LOGI(kTag, "PIR warm-up for %u ms; camera captures are saved by the Wi-Fi receiver", PIR_WARMUP_MS);

    while (true) {
        const uint64_t now_ms = static_cast<uint64_t>(esp_timer_get_time() / 1000);
        const bool pir_high = gpio_get_level(PIR_GPIO) != 0;
        const bool warmed_up = now_ms - boot_time_ms >= PIR_WARMUP_MS;

        if (warmed_up && !pir_warmed_up) {
            pir_warmed_up = true;
            previous_pir_state = false;
            ESP_LOGI(kTag, "PIR armed");
        }

        if (pir_warmed_up && pir_high && !previous_pir_state) {
            char timestamp[32] = {};
            char motion_event_id[48] = {};
            makeTimestamp(timestamp, sizeof(timestamp));
            makeEventId(motion_event_id, sizeof(motion_event_id), timestamp, now_ms - boot_time_ms);
            ESP_LOGW(kTag, "MOTION ALERT event=%s; starting continuous capture", motion_event_id);
            high_threat_alerted_for_motion = false;
            uploadMotionAlert(motion_event_id, timestamp, now_ms - boot_time_ms);
        } else if (pir_warmed_up && !pir_high && previous_pir_state) {
            ESP_LOGI(kTag, "Motion ended; continuous capture stopped");
        }

        previous_pir_state = pir_high;
        if (pir_warmed_up && pir_high) {
            captureAndUpload(now_ms - boot_time_ms);
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}
