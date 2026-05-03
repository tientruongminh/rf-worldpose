idf_component_register(
  SRCS "csi_packet.c" "udp_streamer.c" "mock_csi_producer.c" "csi_collector.c"
  INCLUDE_DIRS "."
  REQUIRES esp_wifi esp_timer nvs_flash esp_netif
)
