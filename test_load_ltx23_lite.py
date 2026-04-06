import torch
from fastvideo.configs.models.dits.ltx2 import LTX23VideoConfig
from fastvideo.models.dits.ltx2 import LTXModel, LTXModelType

def test_load_ltx23_weights():
    print("1. Creating LTX-2.3 22B Base Model architecture...")
    model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_layers=48,
        cross_attention_adaln=True,
        caption_proj_before_connector=True,
        apply_gated_attention=True,
    )
    
    print("✅ Model architecture created successfully")

    # 检查 Block 0 里面的 AdaLN 和 Gated Attention
    block = model.transformer_blocks[0]
    print(f"Block 0 scale_shift_table shape: {block.scale_shift_table.shape}")
    assert block.scale_shift_table.shape[0] == 9, f"Should be 9, got {block.scale_shift_table.shape[0]}"
    print("✅ 22B Specific Features Verified: AdaLN uses 9 params per block")

    print("\n2. Reading weights...")
    model_path = "/root/data_root/model/converted_ltx23/transformer/model.safetensors"
    from safetensors.torch import load_file
    state_dict = load_file(model_path)
    
    print("\n3. Loading weights into model...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"\nMissing keys count: {len(missing_keys)}")
    print(f"Unexpected keys count: {len(unexpected_keys)}")
    
    if len(unexpected_keys) > 0:
        print("\nSome unexpected keys:")
        for k in unexpected_keys[:5]:
            print(f"  - {k}")

if __name__ == "__main__":
    test_load_ltx23_weights()
