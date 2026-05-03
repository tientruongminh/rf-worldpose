#include "csi_packet.h"
#include <string.h>

static void write_le16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)((v >> 8) & 0xffu);
}
static void write_le32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)((v >> 8) & 0xffu);
    p[2] = (uint8_t)((v >> 16) & 0xffu);
    p[3] = (uint8_t)((v >> 24) & 0xffu);
}
static void write_le64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)((v >> (8 * i)) & 0xffu);
}

uint32_t rfpose_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

bool rfpose_encode_csi_packet(const rfpose_csi_frame_t *frame, uint8_t *out, size_t out_cap, size_t *out_len) {
    if (!frame || !out || !out_len || !frame->iq_pairs) return false;
    if (frame->n_subcarriers == 0 || frame->n_subcarriers > RFPOSE_CSI_MAX_SUBCARRIERS) return false;

    const uint32_t payload_len = (uint32_t)frame->n_subcarriers * 2u * sizeof(int16_t);
    const size_t total_len = RFPOSE_CSI_HEADER_LEN + payload_len + sizeof(uint32_t);
    if (out_cap < total_len) return false;

    memset(out, 0, total_len);
    write_le32(out + 0, RFPOSE_CSI_MAGIC);
    out[4] = RFPOSE_CSI_PROTOCOL_VERSION;
    out[5] = frame->node_id;
    write_le16(out + 6, RFPOSE_CSI_HEADER_LEN);
    write_le32(out + 8, frame->seq);
    write_le64(out + 12, frame->timestamp_us);
    out[20] = (uint8_t)frame->rssi;
    out[21] = (uint8_t)frame->noise_floor;
    out[22] = frame->channel;
    out[23] = 0;
    write_le16(out + 24, frame->n_subcarriers);
    write_le16(out + 26, frame->firmware_version);
    write_le32(out + 28, payload_len);

    uint8_t *payload = out + RFPOSE_CSI_HEADER_LEN;
    for (uint32_t i = 0; i < frame->n_subcarriers * 2u; ++i) {
        write_le16(payload + i * 2u, (uint16_t)frame->iq_pairs[i]);
    }

    const uint32_t crc = rfpose_crc32(out, RFPOSE_CSI_HEADER_LEN + payload_len);
    write_le32(out + RFPOSE_CSI_HEADER_LEN + payload_len, crc);
    *out_len = total_len;
    return true;
}
