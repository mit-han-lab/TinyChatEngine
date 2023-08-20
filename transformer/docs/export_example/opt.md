# This doc provides examples to show how to export a OPT model to TinyChatEngine

## OPT 6.7B

### Export model and quantize weights to int4

```bash
python opt_smooth_exporter.py --model_name facebook/opt-6.7b --output_path FP32/models/OPT_6.7B --no-int8_smooth
```

This will export the OPT model to FP32.

```bash
python model_quantizer.py --model_path FP32/models/OPT_6.7B --method QM_ARM --output_path INT4
```

This will quantize the `FP32/models/OPT_6.7B` model into 4-bit weights for `QM_ARM`
