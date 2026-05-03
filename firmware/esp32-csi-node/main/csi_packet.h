#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define RFPOSE_CSI_MAGIC 0xC5110001u
#define RFPOSE_CSI_HEADER_LEN 32u
#define RFPOSE_CSI_PROTOCOL_VERSION 1u
#define RFPOSE_CSI_MAX_SUBCARRIERS 256u

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint8_t protocol_version;
    uint8_t node_id;
    uint16_t header_len;
    uint32_t seq;
    uint64_t timestamp_us;
    int8_t rssi;
    int8_t noise_floor;
    uint8_t channel;
    uint8_t flags;
    uint16_t n_subcarriers;
    uint16_t firmware_version;
    uint32_t payload_len;
} rfpose_csi_header_t;
#pragma pack(pop)

_Static_assert(sizeof(rfpose_csi_header_t) == RFPOSE_CSI_HEADER_LEN, "CSI header must be 32 bytes");

typedef struct {
    uint8_t node_id;
    uint32_t seq;
    uint64_t timestamp_us;
    int8_t rssi;
    int8_t noise_floor;
    uint8_t channel;
    uint16_t n_subcarriers;
    uint16_t firmware_version;
    const int16_t *iq_pairs; // length = n_subcarriers * 2, I/Q interleaved
} rfpose_csi_frame_t;

uint32_t rfpose_crc32(const uint8_t *data, size_t len);
bool rfpose_encode_csi_packet(const rfpose_csi_frame_t *frame, uint8_t *out, size_t out_cap, size_t *out_len);
