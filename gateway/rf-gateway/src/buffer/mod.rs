use anyhow::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use crate::packet::CsiPacket;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferedPacket {
    pub id: i64,
    pub received_at_ms: i64,
    pub node_id: i64,
    pub seq: i64,
    pub timestamp_us: i64,
    pub packet_json: String,
}

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
              packet_json TEXT NOT NULL,
              uploaded_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_csi_node_seq ON csi_packets(node_id, seq);
            CREATE INDEX IF NOT EXISTS idx_csi_uploaded ON csi_packets(uploaded_at_ms, id);
            "#,
        )?;
        // Migration for older local buffers.
        let _ = conn.execute("ALTER TABLE csi_packets ADD COLUMN uploaded_at_ms INTEGER", []);
        Ok(Self { conn })
    }

    pub fn insert_packet(&self, pkt: &CsiPacket) -> Result<()> {
        let now_ms = chrono::Utc::now().timestamp_millis();
        let json = serde_json::to_string(pkt)?;
        self.conn.execute(
            "INSERT INTO csi_packets(received_at_ms,node_id,seq,timestamp_us,rssi,channel,n_subcarriers,packet_json,uploaded_at_ms) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,NULL)",
            params![now_ms, pkt.node_id as i64, pkt.seq as i64, pkt.timestamp_us as i64, pkt.rssi as i64, pkt.channel as i64, pkt.n_subcarriers as i64, json],
        )?;
        Ok(())
    }

    pub fn pending_packets(&self, limit: usize) -> Result<Vec<BufferedPacket>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, received_at_ms, node_id, seq, timestamp_us, packet_json FROM csi_packets WHERE uploaded_at_ms IS NULL ORDER BY id LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], |row| {
            Ok(BufferedPacket {
                id: row.get(0)?,
                received_at_ms: row.get(1)?,
                node_id: row.get(2)?,
                seq: row.get(3)?,
                timestamp_us: row.get(4)?,
                packet_json: row.get(5)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn mark_uploaded(&self, ids: &[i64]) -> Result<()> {
        if ids.is_empty() { return Ok(()); }
        let now_ms = chrono::Utc::now().timestamp_millis();
        let tx = self.conn.unchecked_transaction()?;
        for id in ids {
            tx.execute("UPDATE csi_packets SET uploaded_at_ms=?1 WHERE id=?2", params![now_ms, id])?;
        }
        tx.commit()?;
        Ok(())
    }
}
