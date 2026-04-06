from fastvideo import VideoGenerator
import torch
import os

def main():
    # Use the base model path
    base_model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    # Path to the finetuned transformer weights
    finetuned_weights = "/root/data_root/outputs/finetune_wan2.1_spatialvid_1.3B_161f_4x8_merged/checkpoint-1200-diffusers/transformer/model.safetensors"
    
    if not os.path.exists(finetuned_weights):
        print(f"Error: Finetuned weights not found at {finetuned_weights}")
        return

    print(f"Loading base model {base_model_path} with finetuned weights...")
    generator = VideoGenerator.from_pretrained(
        base_model_path,
        num_gpus=1,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        init_weights_from_safetensors=finetuned_weights
    )

    prompt = "A first-person perspective of walking through a rainy urban neighborhood at dusk."
    
    print(f"Generating 81 frames for prompt: {prompt}")
    # Generate 81 frames
    generator.generate_video(
        prompt=prompt,
        width=832,
        height=480,
        num_frames=161,
        output_path="spatialvid_81f_test.mp4"
    )
    print("Generation complete. Saved to spatialvid_81f_test.mp4")

if __name__ == "__main__":
    main()
