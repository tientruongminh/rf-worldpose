#include "provisioning.h"
#include "nvs.h"
#include "nvs_flash.h"
#include <string.h>
#define NS "rfpose"
bool rfpose_load_config(rfpose_provisioning_config_t *out){ if(!out) return false; nvs_handle_t h; if(nvs_open(NS,NVS_READONLY,&h)!=ESP_OK) return false; size_t ss=sizeof(out->ssid), ps=sizeof(out->password), hs=sizeof(out->gateway_host); int32_t port=0,node=0; bool ok=nvs_get_str(h,"ssid",out->ssid,&ss)==ESP_OK && nvs_get_str(h,"password",out->password,&ps)==ESP_OK && nvs_get_str(h,"gateway",out->gateway_host,&hs)==ESP_OK && nvs_get_i32(h,"port",&port)==ESP_OK && nvs_get_i32(h,"node",&node)==ESP_OK; out->gateway_port=port; out->node_id=node; nvs_close(h); return ok; }
bool rfpose_save_config(const rfpose_provisioning_config_t *cfg){ if(!cfg) return false; nvs_handle_t h; if(nvs_open(NS,NVS_READWRITE,&h)!=ESP_OK) return false; bool ok=nvs_set_str(h,"ssid",cfg->ssid)==ESP_OK && nvs_set_str(h,"password",cfg->password)==ESP_OK && nvs_set_str(h,"gateway",cfg->gateway_host)==ESP_OK && nvs_set_i32(h,"port",cfg->gateway_port)==ESP_OK && nvs_set_i32(h,"node",cfg->node_id)==ESP_OK && nvs_commit(h)==ESP_OK; nvs_close(h); return ok; }
