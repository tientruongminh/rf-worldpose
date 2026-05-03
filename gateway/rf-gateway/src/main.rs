mod buffer;
mod nats;
mod packet;
mod uploader;

use anyhow::Result;
use std::{collections::HashMap, sync::{Arc, Mutex}, time::Duration};
use tokio::net::UdpSocket;
use tracing::{debug, info, warn};
use buffer::LocalBuffer;
use nats::NatsPublisher;
use packet::{decode_csi_packet, NodeHealth};
use uploader::{s3_client, BronzeUploader};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let bind = std::env::var("RFPOSE_GATEWAY_BIND").unwrap_or_else(|_| "0.0.0.0:5006".to_string());
    let deployment_id = std::env::var("RFPOSE_DEPLOYMENT_ID").unwrap_or_else(|_| "room01".to_string());
    let sqlite_path = std::env::var("RFPOSE_GATEWAY_SQLITE").unwrap_or_else(|_| "rf-gateway-buffer.sqlite".to_string());
    let nats_url = std::env::var("NATS_URL").ok();
    let s3_bucket = std::env::var("S3_BUCKET").ok();
    let s3_endpoint = std::env::var("S3_ENDPOINT_URL").ok();
    let upload_interval_secs: u64 = std::env::var("RFPOSE_UPLOAD_INTERVAL_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    let upload_batch_size: usize = std::env::var("RFPOSE_UPLOAD_BATCH_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(500);

    let sock = UdpSocket::bind(&bind).await?;
    let buffer = Arc::new(Mutex::new(LocalBuffer::open(&sqlite_path)?));
    let publisher = NatsPublisher::connect(nats_url, deployment_id.clone()).await?;
    let mut health: HashMap<u8, NodeHealth> = HashMap::new();

    if let Some(bucket) = s3_bucket.clone() {
        let client = s3_client(s3_endpoint).await;
        let uploader = BronzeUploader::new(client, bucket, deployment_id.clone());
        let upload_buffer = Arc::clone(&buffer);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(upload_interval_secs));
            loop {
                interval.tick().await;
                let packets = {
                    let guard = upload_buffer.lock().expect("buffer mutex poisoned");
                    match guard.pending_packets(upload_batch_size) {
                        Ok(p) => p,
                        Err(e) => {
                            warn!(error = %e, "failed reading pending packets");
                            continue;
                        }
                    }
                };
                if packets.is_empty() {
                    debug!("no pending packets to upload");
                    continue;
                }
                match uploader.upload_packets(packets).await {
                    Ok(ids) => {
                        let uploaded = ids.len();
                        let guard = upload_buffer.lock().expect("buffer mutex poisoned");
                        if let Err(e) = guard.mark_uploaded(&ids) {
                            warn!(error = %e, "uploaded Bronze batch but failed to mark local buffer");
                        } else {
                            info!(uploaded, "uploaded Bronze CSI batch");
                        }
                    }
                    Err(e) => warn!(error = %e, "Bronze upload failed; packets remain buffered"),
                }
            }
        });
    } else {
        info!("S3_BUCKET not set; Bronze uploader disabled");
    }

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

                {
                    let guard = buffer.lock().expect("buffer mutex poisoned");
                    guard.insert_packet(&pkt)?;
                }
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
