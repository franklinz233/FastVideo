import torch
from fastvideo.configs.models.dits.ltx2 import LTX23VideoConfig
from fastvideo.models.dits.ltx2 import LTX2Transformer3DModel

def test_load_ltx23_weights():
    # 创建 LTX-2.3 22B 配置
    config = LTX23VideoConfig()

    print("1. Creating LTX-2.3 model architecture...")
    model = LTX2Transformer3DModel(config, hf_config={})

    # 验证 22B 新特性的架构是否正确创建
    assert model.model.caption_proj_before_connector == True
    assert model.model.cross_attention_adaln == True
    assert model.model.apply_gated_attention == True

    # 检查 Block 0 里面的 AdaLN 和 Gated Attention
    block = model.model.transformer_blocks[0]
    assert block.scale_shift_table.shape[0] == 9  # 19B 是 6
    assert hasattr(block, "prompt_scale_shift_table")
    assert hasattr(block, "attn1_gate") if hasattr(block, "attn1_gate") else False # 如果你在之前实现了

    print("✅ Model architecture verified")

    print("2. Loading converted weights from disk...")
    model_path = "/root/data_root/model/converted_ltx23/transformer/model.safetensors"
    from safetensors.torch import load_file
    state_dict = load_file(model_path)

    print("3. Loading weights into model...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"\nMissing keys count: {len(missing_keys)}")
    print(f"Unexpected keys count: {len(unexpected_keys)}")

    if len(unexpected_keys) > 0:
        print("\nSome unexpected keys:")
        for k in unexpected_keys[:10]:
            print(f"  - {k}")

    print("\n✅ Weight loading test completed!")

if __name__ == "__main__":
    test_load_ltx23_weights()
