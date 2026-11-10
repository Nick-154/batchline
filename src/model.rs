//! Adapter that lets the batcher drive an `onyx` session.

use crate::batcher::Model;
use onyx::{Session, SessionOptions, Tensor};
use std::path::Path;

pub struct OnnxModel {
    session: Session,
    /// Shape of one item, without the batch dimension.
    item_shape: Vec<usize>,
}

impl OnnxModel {
    /// Load a model with one input whose first dimension is a dynamic batch
    /// size and whose other dimensions are fixed. `threads` = 0 uses every core.
    pub fn load(path: impl AsRef<Path>, threads: usize) -> Result<Self, String> {
        let options = SessionOptions {
            threads,
            ..Default::default()
        };
        let session = Session::from_file(path, options).map_err(|e| e.to_string())?;
        let inputs = session.inputs();
        if inputs.len() != 1 || session.output_names().len() != 1 {
            return Err("model must have exactly one input and one output".into());
        }
        let shape = &inputs[0].shape;
        if shape.first() != Some(&None) {
            return Err(format!(
                "input '{}' must have a dynamic first (batch) dimension",
                inputs[0].name
            ));
        }
        let item_shape = shape[1..]
            .iter()
            .map(|d| d.ok_or_else(|| "only the batch dimension may be dynamic".to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(OnnxModel { session, item_shape })
    }

    pub fn item_shape(&self) -> &[usize] {
        &self.item_shape
    }
}

impl Model for OnnxModel {
    fn input_len(&self) -> usize {
        self.item_shape.iter().product()
    }

    fn run_batch(&self, inputs: Vec<f32>, batch: usize) -> Result<Vec<f32>, String> {
        let mut shape = Vec::with_capacity(self.item_shape.len() + 1);
        shape.push(batch);
        shape.extend_from_slice(&self.item_shape);
        let input = Tensor::new(shape, inputs).map_err(|e| e.to_string())?;
        let output = self.session.run_one(input).map_err(|e| e.to_string())?;
        Ok(output.into_data())
    }
}
