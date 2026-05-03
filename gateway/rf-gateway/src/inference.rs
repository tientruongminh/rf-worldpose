use anyhow::Result;
use crate::packet::CsiPacket;
#[derive(Clone, Debug)]
pub struct InferenceOutput { pub action: String, pub confidence: f32 }
#[derive(Clone)]
pub struct EdgeInference { model_path: Option<String> }
impl EdgeInference { pub fn new(model_path: Option<String>) -> Self { Self { model_path } } pub fn predict(&self, _window: &[CsiPacket]) -> Result<Option<InferenceOutput>> { if self.model_path.is_none(){return Ok(None)}; Ok(Some(InferenceOutput{action:"unknown".into(),confidence:0.0})) } }
