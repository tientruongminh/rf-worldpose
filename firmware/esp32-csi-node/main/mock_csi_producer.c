#include "csi_packet.h"
#include "udp_streamer.h"
#include <stdint.h>
#include <stdlib.h>

#ifndef RFPOSE_NODE_ID
#define RFPOSE_NODE_ID 1
#endif
#ifndef RFPOSE_FIRMWARE_VERSION
#define RFPOSE_FIRMWARE_VERSION 1
#endif

static uint32_t g_seq = 0;

bool rfpose_send_mock_csi(uint64_t timestamp_us, int8_t rssi, uint8_t channel) {
    enum { N_SUB = 56, MAX_PACKET = RFPOSE_CSI_HEADER_LEN + N_SUB * 2 * 2 + 4 };
    static int16_t iq[N_SUB * 2];
    static uint8_t packet[MAX_PACKET];
    for (int i = 0; i < N_SUB; ++i) {
        iq[i * 2] = (int16_t)((i % 17) - 8);
        iq[i * 2 + 1] = (int16_t)((i % 23) - 11);
    }
    rfpose_csi_frame_t frame = {
        .node_id = RFPOSE_NODE_ID,
        .seq = g_seq++,
        .timestamp_us = timestamp_us,
        .rssi = rssi,
        .noise_floor = -90,
        .channel = channel,
        .n_subcarriers = N_SUB,
        .firmware_version = RFPOSE_FIRMWARE_VERSION,
        .iq_pairs = iq,
    };
    size_t len = 0;
    if (!rfpose_encode_csi_packet(&frame, packet, sizeof(packet), &len)) return false;
    return rfpose_udp_streamer_send(packet, len);
}
