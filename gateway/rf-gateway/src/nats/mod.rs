use anyhow::Result;
use crate::packet::{CsiPacket, NodeHealth};

#[derive(Clone)]
pub struct NatsPublisher {
    client: Option<async_nats::Client>,
    deployment_id: String,
}

impl NatsPublisher {
    pub async fn connect(url: Option<String>, deployment_id: String) -> Result<Self> {
        let client = match url {
            Some(u) if !u.is_empty() => Some(async_nats::connect(u).await?),
            _ => None,
        };
        Ok(Self { client, deployment_id })
    }

    pub async fn publish_csi(&self, pkt: &CsiPacket) -> Result<()> {
        if let Some(client) = &self.client {
            let subject = format!("csi.raw.{}.node-{:02}", self.deployment_id, pkt.node_id);
            client.publish(subject, serde_json::to_vec(pkt)?.into()).await?;
        }
        Ok(())
    }

    pub async fn publish_health(&self, health: &NodeHealth) -> Result<()> {
        if let Some(client) = &self.client {
            let subject = format!("node.health.{}", self.deployment_id);
            client.publish(subject, serde_json::to_vec(health)?.into()).await?;
        }
        Ok(())
    }
}
