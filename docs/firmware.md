# Firmware Guide

## Modules
```text
csi_packet.c/h       binary packet + CRC32
udp_streamer.c/h     UDP transport
csi_collector.c/h    ESP-IDF CSI callback
provisioning.c/h     NVS config load/save
mock_csi_producer.c  test producer
```

## Packet contract
```text
magic u32, protocol_version u8, node_id u8, header_len u16, seq u32, timestamp_us u64, rssi i8, noise_floor i8, channel u8, flags u8, n_subcarriers u16, firmware_version u16, payload_len u32, payload int16 I/Q pairs, crc32 u32
```

## CSI capture path
```text
wifi_csi_info_t → copy I/Q → rfpose_csi_frame_t → rfpose_encode_csi_packet → UDP gateway
```

## OTA
Use signed OTA, canary rollout, heartbeat observation, and rollback partition.
