from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import WanT2V480PConfig

MODEL_PATH = "/pfs_root/sc/outputs/wan2.1_1.3B_161f_spatialvid_diffusers"
OUTPUT_PATH = "/root/data_root/outputs/inference_spatialvid_test"

if __name__ == '__main__':
    # Use WanT2V480PConfig with flow_shift=5.0 to match training config
    pipeline_config = WanT2V480PConfig()
    pipeline_config.flow_shift = 5.0

    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=4,
        sp_size=4,
        text_encoder_cpu_offload=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        pin_cpu_memory=True,
        pipeline_config=pipeline_config,
    )

    prompts = [
        "Walking through an indoor shopping mall with stores and people on multiple floors.",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n=== Generating video {i} ===")
        print(f"Prompt: {prompt}")
        video = generator.generate_video(
            prompt,
            num_frames=157,
            height=480,
            width=832,
            num_inference_steps=50,
            guidance_scale=5.0,
            seed=42 + i,
            output_path=OUTPUT_PATH,
            save_video=True,
        )
        print(f"Video {i} saved to {OUTPUT_PATH}")

    print("\nDone!")
