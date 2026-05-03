use anyhow::Result;
use aws_sdk_s3::{primitives::ByteStream, Client};
use chrono::{Datelike, Utc};
use serde::Serialize;
use uuid::Uuid;
use crate::buffer::BufferedPacket;

#[derive(Clone)]
pub struct BronzeUploader {
    client: Client,
    bucket: String,
    deployment_id: String,
}

#[derive(Serialize)]
struct BronzeBatch<'a> {
    schema: &'static str,
    deployment_id: &'a str,
    uploaded_at: String,
    packets: &'a [BufferedPacket],
}

impl BronzeUploader {
    pub fn new(client: Client, bucket: String, deployment_id: String) -> Self {
        Self { client, bucket, deployment_id }
    }

    pub async fn upload_packets(&self, packets: Vec<BufferedPacket>) -> Result<Vec<i64>> {
        if packets.is_empty() { return Ok(vec![]); }
        let now = Utc::now();
        let key = format!(
            "bronze/deployment={}/date={:04}-{:02}-{:02}/csi_raw/batch-{}-{}.json",
            self.deployment_id,
            now.year(), now.month(), now.day(),
            now.timestamp_millis(),
            Uuid::new_v4()
        );
        let batch = BronzeBatch {
            schema: "rfpose.bronze.csi_batch.v1",
            deployment_id: &self.deployment_id,
            uploaded_at: now.to_rfc3339(),
            packets: &packets,
        };
        let body = serde_json::to_vec(&batch)?;
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(ByteStream::from(body))
            .content_type("application/json")
            .send()
            .await?;
        let ids: Vec<i64> = packets.iter().map(|p| p.id).collect();
        Ok(ids)
    }
}

pub async fn s3_client(endpoint_url: Option<String>) -> Client {
    let mut loader = aws_config::from_env();
    if let Some(endpoint) = endpoint_url {
        loader = loader.endpoint_url(endpoint);
    }
    let config = loader.load().await;
    Client::new(&config)
}
