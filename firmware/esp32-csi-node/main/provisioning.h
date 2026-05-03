#pragma once
#include <stdbool.h>
typedef struct { char ssid[33]; char password[65]; char gateway_host[64]; int gateway_port; int node_id; } rfpose_provisioning_config_t;
bool rfpose_load_config(rfpose_provisioning_config_t *out);
bool rfpose_save_config(const rfpose_provisioning_config_t *cfg);
