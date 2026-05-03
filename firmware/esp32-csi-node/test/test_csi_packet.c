#include "../main/csi_packet.h"
#include <assert.h>
#include <stdio.h>

static uint32_t read_le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int main(void) {
    int16_t iq[4] = {3, 4, 5, 12};
    uint8_t packet[128];
    size_t len = 0;
    rfpose_csi_frame_t frame = {
        .node_id = 7,
        .seq = 42,
        .timestamp_us = 123456,
        .rssi = -50,
        .noise_floor = -90,
        .channel = 6,
        .n_subcarriers = 2,
        .firmware_version = 100,
        .iq_pairs = iq,
    };
    assert(rfpose_encode_csi_packet(&frame, packet, sizeof(packet), &len));
    assert(len == RFPOSE_CSI_HEADER_LEN + sizeof(iq) + 4);
    assert(read_le32(packet) == RFPOSE_CSI_MAGIC);
    uint32_t supplied = read_le32(packet + len - 4);
    uint32_t computed = rfpose_crc32(packet, len - 4);
    assert(supplied == computed);
    puts("csi_packet tests passed");
    return 0;
}
