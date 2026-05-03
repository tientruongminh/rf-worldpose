use std::sync::atomic::{AtomicU64, Ordering};
#[derive(Default)]
pub struct GatewayMetrics { pub packets_ok: AtomicU64, pub packets_bad: AtomicU64, pub packets_uploaded: AtomicU64 }
impl GatewayMetrics { pub fn render_prometheus(&self) -> String { format!("rfpose_packets_ok {}\nrfpose_packets_bad {}\nrfpose_packets_uploaded {}\n", self.packets_ok.load(Ordering::Relaxed), self.packets_bad.load(Ordering::Relaxed), self.packets_uploaded.load(Ordering::Relaxed)) } }
