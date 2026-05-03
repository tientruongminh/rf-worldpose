use anyhow::Result;
use rusqlite::{params, Connection};
use crate::packet::CsiPacket;

pub struct LocalBuffer {
    conn: Connection,
}

impl LocalBuffer {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS csi_packets (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              received_at_ms INTEGER NOT NULL,
              node_id INTEGER NOT NULL,
              seq INTEGER NOT NULL,
              timestamp_us INTEGER NOT NULL,
              rssi INTEGER NOT NULL,
              channel INTEGER NOT NULL,
              n_subcarriers INTEGER NOT NULL,
              packet_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_csi_node_seq ON csi_packets(node_id, seq);
            "#,
        )?;
        Ok(Self { conn })
    }

    pub fn insert_packet(&self, pkt: &CsiPacket) -> Result<()> {
        let now_ms = chrono::Utc::now().timestamp_millis();
        let json = serde_json::to_string(pkt)?;
        self.conn.execute(
            "INSERT INTO csi_packets(received_at_ms,node_id,seq,timestamp_us,rssi,channel,n_subcarriers,packet_json) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![now_ms, pkt.node_id as i64, pkt.seq as i64, pkt.timestamp_us as i64, pkt.rssi as i64, pkt.channel as i64, pkt.n_subcarriers as i64, json],
        )?;
        Ok(())
    }
}
