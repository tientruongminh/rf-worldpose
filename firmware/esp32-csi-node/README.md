# ESP32-S3 CSI Node Firmware

Production firmware target: ESP-IDF C/C++.

## Implemented modules

- `csi_packet.c/h` — binary CSI packet encoder + CRC32.
- `udp_streamer.c/h` — UDP sender to Rust gateway.
- `csi_collector.c/h` — ESP-IDF CSI callback integration.
- `mock_csi_producer.c` — mock CSI packet generation for end-to-end tests.

## Packet flow

```text
esp_wifi_set_csi_rx_cb()
→ wifi_csi_info_t
→ rfpose_csi_frame_t
→ rfpose_encode_csi_packet()
→ rfpose_udp_streamer_send()
→ Rust gateway UDP decoder
```

## Menuconfig

```text
RFPOSE_NODE_ID
RFPOSE_FIRMWARE_VERSION
RFPOSE_GATEWAY_HOST
RFPOSE_GATEWAY_PORT
RFPOSE_MAX_CSI_SUBCARRIERS
```

## Production TODO

- WiFi provisioning via NVS/BLE/serial.
- Signed OTA + rollback partition table.
- Device identity keys.
- Heartbeat packets independent from CSI stream.
- Deployment-specific traffic/beacon strategy for stable CSI capture.
