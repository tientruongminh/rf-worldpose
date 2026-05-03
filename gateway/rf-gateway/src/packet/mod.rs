use anyhow::{anyhow, bail, Result};
use serde::Serialize;

pub const MAGIC_CSI_RAW: u32 = 0xC511_0001;
pub const HEADER_LEN: usize = 32;

#[derive(Debug, Clone, Serialize)]
pub struct CsiPacket {
    pub protocol_version: u8,
    pub node_id: u8,
    pub seq: u32,
    pub timestamp_us: u64,
    pub rssi: i8,
    pub noise_floor: i8,
    pub channel: u8,
    pub n_subcarriers: u16,
    pub firmware_version: u16,
    pub iq: Vec<i16>,
    pub amplitude: Vec<f32>,
    pub crc32: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeHealth {
    pub node_id: u8,
    pub last_seq: u32,
    pub packets_received: u64,
    pub packets_dropped_est: u64,
    pub last_rssi: i8,
    pub channel: u8,
    pub last_timestamp_us: u64,
}

fn le_u16(data: &[u8], off: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(data.get(off..off + 2).ok_or_else(|| anyhow!("u16 out of bounds"))?.try_into()?))
}
fn le_u32(data: &[u8], off: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(data.get(off..off + 4).ok_or_else(|| anyhow!("u32 out of bounds"))?.try_into()?))
}
fn le_u64(data: &[u8], off: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(data.get(off..off + 8).ok_or_else(|| anyhow!("u64 out of bounds"))?.try_into()?))
}

/// Canonical production packet layout, little-endian:
/// magic u32, protocol u8, node_id u8, header_len u16, seq u32, timestamp_us u64,
/// rssi i8, noise i8, channel u8, flags u8, n_subcarriers u16, firmware u16,
/// payload_len u32, payload i16 IQ pairs, crc32 u32 over all bytes before crc.
pub fn decode_csi_packet(data: &[u8]) -> Result<CsiPacket> {
    if data.len() < HEADER_LEN + 4 {
        bail!("packet too short: {}", data.len());
    }
    let magic = le_u32(data, 0)?;
    if magic != MAGIC_CSI_RAW {
        bail!("bad magic: 0x{magic:08x}");
    }
    let protocol_version = data[4];
    let node_id = data[5];
    let header_len = le_u16(data, 6)? as usize;
    if header_len != HEADER_LEN {
        bail!("unsupported header_len: {header_len}");
    }
    let seq = le_u32(data, 8)?;
    let timestamp_us = le_u64(data, 12)?;
    let rssi = data[20] as i8;
    let noise_floor = data[21] as i8;
    let channel = data[22];
    let n_subcarriers = le_u16(data, 24)?;
    let firmware_version = le_u16(data, 26)?;
    let payload_len = le_u32(data, 28)? as usize;
    let expected = HEADER_LEN + payload_len + 4;
    if data.len() != expected {
        bail!("length mismatch: got {}, expected {}", data.len(), expected);
    }
    let supplied_crc = le_u32(data, HEADER_LEN + payload_len)?;
    let computed_crc = crc32fast::hash(&data[..HEADER_LEN + payload_len]);
    if supplied_crc != computed_crc {
        bail!("crc mismatch: supplied={supplied_crc:08x} computed={computed_crc:08x}");
    }
    if payload_len % 2 != 0 {
        bail!("payload_len not i16 aligned");
    }
    let samples = payload_len / 2;
    let mut iq = Vec::with_capacity(samples);
    for i in 0..samples {
        let off = HEADER_LEN + i * 2;
        iq.push(i16::from_le_bytes(data[off..off + 2].try_into()?));
    }
    if iq.len() != n_subcarriers as usize * 2 {
        bail!("iq sample count does not match n_subcarriers");
    }
    let mut amplitude = Vec::with_capacity(n_subcarriers as usize);
    for pair in iq.chunks_exact(2) {
        let re = pair[0] as f32;
        let im = pair[1] as f32;
        amplitude.push((re * re + im * im).sqrt());
    }
    Ok(CsiPacket { protocol_version, node_id, seq, timestamp_us, rssi, noise_floor, channel, n_subcarriers, firmware_version, iq, amplitude, crc32: supplied_crc })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_packet() -> Vec<u8> {
        let mut data = vec![];
        data.extend_from_slice(&MAGIC_CSI_RAW.to_le_bytes());
        data.push(1);
        data.push(7);
        data.extend_from_slice(&(HEADER_LEN as u16).to_le_bytes());
        data.extend_from_slice(&42u32.to_le_bytes());
        data.extend_from_slice(&123_456u64.to_le_bytes());
        data.push((-50i8) as u8);
        data.push((-90i8) as u8);
        data.push(6);
        data.push(0);
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&100u16.to_le_bytes());
        let payload: [i16; 4] = [3, 4, 5, 12];
        data.extend_from_slice(&(payload.len() as u32 * 2).to_le_bytes());
        for x in payload { data.extend_from_slice(&x.to_le_bytes()); }
        let crc = crc32fast::hash(&data);
        data.extend_from_slice(&crc.to_le_bytes());
        data
    }

    #[test]
    fn decodes_valid_packet() {
        let pkt = decode_csi_packet(&build_packet()).unwrap();
        assert_eq!(pkt.node_id, 7);
        assert_eq!(pkt.seq, 42);
        assert_eq!(pkt.amplitude, vec![5.0, 13.0]);
    }

    #[test]
    fn rejects_bad_crc() {
        let mut data = build_packet();
        data[10] ^= 1;
        assert!(decode_csi_packet(&data).is_err());
    }
}
