use anyhow::Result;
use tokio::net::UdpSocket;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let bind = std::env::var("RFPOSE_GATEWAY_BIND").unwrap_or_else(|_| "0.0.0.0:5006".to_string());
    let sock = UdpSocket::bind(&bind).await?;
    info!(%bind, "rf-gateway listening for CSI packets");
    let mut buf = vec![0u8; 8192];
    loop {
        let (n, addr) = sock.recv_from(&mut buf).await?;
        if n < 16 {
            warn!(%addr, bytes = n, "dropping short packet");
            continue;
        }
        // TODO: decode libs/rfpose-schemas csi packet, validate CRC, persist local buffer, publish NATS.
        info!(%addr, bytes = n, "received CSI packet");
    }
}
