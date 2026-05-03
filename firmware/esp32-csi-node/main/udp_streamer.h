#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

bool rfpose_udp_streamer_init(const char *host, uint16_t port);
bool rfpose_udp_streamer_send(const uint8_t *packet, size_t len);
void rfpose_udp_streamer_close(void);
