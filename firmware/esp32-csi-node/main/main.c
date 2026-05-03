#include "csi_collector.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

static const char *TAG = "rfpose_main";

void app_main(void) {
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Production TODO: provision WiFi SSID/password via NVS/BLE/serial provisioning.
    // CSI can be enabled after WiFi init; association/beacon traffic strategy is deployment-specific.
    if (!rfpose_csi_collector_start()) {
        ESP_LOGE(TAG, "failed to start RF-WorldPose CSI collector");
    }
}
