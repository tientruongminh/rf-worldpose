mod buffer;
mod nats;
mod packet;

use anyhow::Result;
use std::collections::HashMap;
use tokio::net::UdpSocket;
use tracing::{debug, error, info, warn};
use buffer::LocalBuffer;
use nats::NatsPublisher;
use packet::{decode_csi_packet, NodeHealth};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let bind = std::env::var("RFPOSE_GATEWAY_BIND").unwrap_or_else(|_| "0.0.0.0:5006".to_string());
    let deployment_id = std::env::var("RFPOSE_DEPLOYMENT_ID").unwrap_or_else(|_| "room01".to_string());
    let sqlite_path = std::env::var("RFPOSE_GATEWAY_SQLITE").unwrap_or_else(|_| "rf-gateway-buffer.sqlite".to_string());
    let nats_url = std::env::var("NATS_URL").ok();

    let sock = UdpSocket::bind(&bind).await?;
    let buffer = LocalBuffer::open(&sqlite_path)?;
    let publisher = NatsPublisher::connect(nats_url, deployment_id.clone()).await?;
    let mut health: HashMap<u8, NodeHealth> = HashMap::new();

    info!(%bind, %deployment_id, %sqlite_path, "rf-gateway listening for CSI packets");
    let mut buf = vec![0u8; 8192];
    loop {
        let (n, addr) = sock.recv_from(&mut buf).await?;
        match decode_csi_packet(&buf[..n]) {
            Ok(pkt) => {
                let h = health.entry(pkt.node_id).or_insert(NodeHealth {
                    node_id: pkt.node_id,
                    last_seq: pkt.seq,
                    packets_received: 0,
                    packets_dropped_est: 0,
                    last_rssi: pkt.rssi,
                    channel: pkt.channel,
                    last_timestamp_us: pkt.timestamp_us,
                });
                if h.packets_received > 0 && pkt.seq > h.last_seq + 1 {
                    h.packets_dropped_est += (pkt.seq - h.last_seq - 1) as u64;
                }
                h.last_seq = pkt.seq;
                h.packets_received += 1;
                h.last_rssi = pkt.rssi;
                h.channel = pkt.channel;
                h.last_timestamp_us = pkt.timestamp_us;

                buffer.insert_packet(&pkt)?;
                publisher.publish_csi(&pkt).await?;
                if h.packets_received % 50 == 0 {
                    publisher.publish_health(h).await?;
                    info!(node_id = pkt.node_id, seq = pkt.seq, received = h.packets_received, dropped = h.packets_dropped_est, "node health");
                } else {
                    debug!(%addr, node_id = pkt.node_id, seq = pkt.seq, "packet accepted");
                }
            }
            Err(e) => {
                warn!(%addr, bytes = n, error = %e, "dropping invalid CSI packet");
            }
        }
    }
}
