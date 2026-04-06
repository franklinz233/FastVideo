# LTX-2.3 (22B) Fine-tuning Examples

These are e2e example scripts for fine-tuning LTX-2.3 (22B) on the crush-smol dataset.

## Key Differences from LTX-2 (19B)

LTX-2.3 (22B) introduces three new architectural features that are
automatically handled by the pipeline:

1. **Caption projection before connector** — the text projection lives in
   the Gemma text-encoder connector instead of inside the transformer.
2. **Cross-attention AdaLN** — the text cross-attention path uses
   adaptive LayerNorm modulation (scale\_shift\_table grows from 6→9).
3. **Gated attention** — optional gating on attention outputs.

The existing `LTX2TrainingPipeline` detects these features from the
converted checkpoint's `config.json` and enables them transparently.

## Prerequisites

### 1. Convert LTX-2.3 checkpoint to FastVideo format

```bash
python scripts/checkpoint_conversion/convert_ltx2_weights.py \
  --source /path/to/ltx-2.3-22b-dev.safetensors \
  --output converted_weights/ltx23-base \
  --gemma-path /path/to/gemma-3-12b-it-qat-q4_0-unquantized
```

### 2. Prepare data (same as LTX-2)

```bash
bash examples/training/finetune/ltx2/preprocess_ltx2_data_t2v_new.sh
```

## Run training

### LoRA fine-tuning (recommended for most use cases)

```bash
bash examples/training/finetune/ltx23/finetune_t2v_lora.sh
```

### Full fine-tuning

```bash
bash examples/training/finetune/ltx23/finetune_t2v.sh
```

## Notes

- Update `MODEL_PATH` to point to your converted LTX-2.3 checkpoint.
- Update `DATA_DIR` to point to your preprocessed dataset.
- The data preprocessing pipeline is shared with LTX-2 (19B).
- 22B model requires more GPU memory; use gradient checkpointing and
  consider reducing batch size or resolution if needed.
- For 8-GPU LoRA training, ~40GB per GPU is typical at 480×832 resolution.
