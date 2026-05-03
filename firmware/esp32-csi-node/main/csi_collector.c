#include "csi_collector.h"
#include "csi_packet.h"
#include "udp_streamer.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_mac.h"
#include <string.h>

#ifndef CONFIG_RFPOSE_NODE_ID
#define CONFIG_RFPOSE_NODE_ID 1
#endif
#ifndef CONFIG_RFPOSE_FIRMWARE_VERSION
#define CONFIG_RFPOSE_FIRMWARE_VERSION 1
#endif
#ifndef CONFIG_RFPOSE_GATEWAY_HOST
#define CONFIG_RFPOSE_GATEWAY_HOST "192.168.1.100"
#endif
#ifndef CONFIG_RFPOSE_GATEWAY_PORT
#define CONFIG_RFPOSE_GATEWAY_PORT 5006
#endif
#ifndef CONFIG_RFPOSE_MAX_CSI_SUBCARRIERS
#define CONFIG_RFPOSE_MAX_CSI_SUBCARRIERS 128
#endif

static const char *TAG = "rfpose_csi";
static uint32_t g_seq = 0;
static bool g_running = false;
static int16_t g_iq[RFPOSE_CSI_MAX_SUBCARRIERS * 2];
static uint8_t g_packet[RFPOSE_CSI_HEADER_LEN + RFPOSE_CSI_MAX_SUBCARRIERS * 2 * sizeof(int16_t) + sizeof(uint32_t)];

static uint16_t copy_csi_iq(const wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len <= 0) return 0;
    uint16_t n_pairs = (uint16_t)(info->len / 2); // ESP-IDF CSI buf is I/Q signed bytes interleaved in common configs.
    if (n_pairs > CONFIG_RFPOSE_MAX_CSI_SUBCARRIERS) n_pairs = CONFIG_RFPOSE_MAX_CSI_SUBCARRIERS;
    for (uint16_t i = 0; i < n_pairs; ++i) {
        int8_t imag = (int8_t)info->buf[i * 2];
        int8_t real = (int8_t)info->buf[i * 2 + 1];
        g_iq[i * 2] = (int16_t)real;
        g_iq[i * 2 + 1] = (int16_t)imag;
    }
    return n_pairs;
}

static void rfpose_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
    (void)ctx;
    if (!g_running || !info) return;
    const uint16_t n_sub = copy_csi_iq(info);
    if (n_sub == 0) return;

    rfpose_csi_frame_t frame = {
        .node_id = (uint8_t)CONFIG_RFPOSE_NODE_ID,
        .seq = g_seq++,
        .timestamp_us = (uint64_t)esp_timer_get_time(),
        .rssi = (int8_t)info->rx_ctrl.rssi,
        .noise_floor = (int8_t)info->rx_ctrl.noise_floor,
        .channel = (uint8_t)info->rx_ctrl.channel,
        .n_subcarriers = n_sub,
        .firmware_version = (uint16_t)CONFIG_RFPOSE_FIRMWARE_VERSION,
        .iq_pairs = g_iq,
    };
    size_t packet_len = 0;
    if (!rfpose_encode_csi_packet(&frame, g_packet, sizeof(g_packet), &packet_len)) {
        ESP_LOGW(TAG, "failed to encode CSI packet");
        return;
    }
    if (!rfpose_udp_streamer_send(g_packet, packet_len)) {
        ESP_LOGW(TAG, "failed to send CSI packet seq=%lu", (unsigned long)frame.seq);
    }
}

bool rfpose_csi_collector_start(void) {
    if (g_running) return true;
    if (!rfpose_udp_streamer_init(CONFIG_RFPOSE_GATEWAY_HOST, CONFIG_RFPOSE_GATEWAY_PORT)) {
        ESP_LOGE(TAG, "UDP streamer init failed: %s:%d", CONFIG_RFPOSE_GATEWAY_HOST, CONFIG_RFPOSE_GATEWAY_PORT);
        return false;
    }

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = false,
    };
    esp_err_t err = esp_wifi_set_csi_config(&csi_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_csi_config failed: %s", esp_err_to_name(err));
        return false;
    }
    err = esp_wifi_set_csi_rx_cb(rfpose_csi_rx_cb, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_csi_rx_cb failed: %s", esp_err_to_name(err));
        return false;
    }
    err = esp_wifi_set_csi(true);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_csi(true) failed: %s", esp_err_to_name(err));
        return false;
    }
    g_running = true;
    ESP_LOGI(TAG, "CSI collector started node=%d gateway=%s:%d", CONFIG_RFPOSE_NODE_ID, CONFIG_RFPOSE_GATEWAY_HOST, CONFIG_RFPOSE_GATEWAY_PORT);
    return true;
}

void rfpose_csi_collector_stop(void) {
    if (!g_running) return;
    esp_wifi_set_csi(false);
    rfpose_udp_streamer_close();
    g_running = false;
    ESP_LOGI(TAG, "CSI collector stopped");
}
