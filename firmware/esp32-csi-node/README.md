# ESP32-S3 CSI Node Firmware

Production firmware target: ESP-IDF C/C++.

Responsibilities:
- CSI capture via ESP-IDF CSI API
- binary packet encode with seq/timestamp/CRC
- NVS provisioning
- heartbeat and node health
- watchdog and crash logs
- signed OTA update

Packet schema lives in `libs/rfpose-schemas/proto/csi.proto`.
