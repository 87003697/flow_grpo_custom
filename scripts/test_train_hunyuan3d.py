#!/usr/bin/env python3
"""
Test script for Hunyuan3D GRPO training implementation.
Verifies that all components work together correctly.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from flow_grpo.trainer_3d import Hunyuan3DGRPOTrainer, create_3d_reward_function
from flow_grpo.diffusers_patch.hunyuan3d_pipeline_with_logprob import hunyuan3d_pipeline_with_logprob


def create_test_data(temp_dir: str, num_samples: int = 3):
    """Create test data for training."""
    train_dir = os.path.join(temp_dir, "train")
    test_dir = os.path.join(temp_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create some dummy images
    for i in range(num_samples):
        # Create a simple colored image
        image = Image.new('RGB', (512, 512), color=(255 * i // num_samples, 128, 255 - 255 * i // num_samples))
        
        # Save to both train and test
        image.save(os.path.join(train_dir, f"image_{i}.png"))
        image.save(os.path.join(test_dir, f"image_{i}.png"))
    
    # Create prompt files
    prompts = [
        "A simple 3D cube object",
        "A spherical 3D object",
        "A cylindrical 3D shape",
    ]
    
    with open(os.path.join(temp_dir, "train_prompts.txt"), 'w') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")
    
    with open(os.path.join(temp_dir, "test_prompts.txt"), 'w') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")
    
    return train_dir, test_dir


def test_pipeline_with_logprob():
    """Test pipeline with log probability computation."""
    print("Testing pipeline with log probability computation...")
    
    # Create mock pipeline
    pipeline = MockPipeline()
    
    # Test image input
    image_path = "test_image.png"
    
    try:
        # Test the pipeline with log probability function
        meshes, latents, log_probs, kl = hunyuan3d_pipeline_with_logprob(
            pipeline,  # Pass pipeline as first parameter (self)
            image=image_path,
            num_inference_steps=10,
            guidance_scale=5.0,
            output_type="trimesh",
        )
        
        print(f"✅ Pipeline with logprob succeeded")
        print(f"  - Meshes: {len(meshes) if isinstance(meshes, list) else 'single mesh'}")
        print(f"  - Latents: {len(latents)} steps")
        print(f"  - Log probs: {len(log_probs)} steps")
        print(f"  - KL: {len(kl)} steps")
        
        # Test with KL reward
        meshes, latents, log_probs, kl = hunyuan3d_pipeline_with_logprob(
            pipeline,  # Pass pipeline as first parameter (self)
            image=image_path,
            num_inference_steps=10,
            guidance_scale=5.0,
            kl_reward=0.1,
            output_type="trimesh",
        )
        
        print(f"✅ Pipeline with KL reward succeeded")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline with logprob failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_components():
    """Test the trainer components."""
    print("\n🧪 Testing trainer components...")
    
    # Create test image
    test_image = Image.new('RGB', (512, 512), color=(255, 128, 64))
    temp_image_path = "/tmp/test_image.png"
    test_image.save(temp_image_path)
    
    try:
        # Initialize components
        pipeline = Hunyuan3DPipeline()
        
        # Setup reward configuration
        reward_config = {
            "geometric_quality": 0.3,
            "uni3d": 0.7
        }
        
        trainer = Hunyuan3DGRPOTrainer(
            pipeline=pipeline,
            reward_config=reward_config,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Test mesh generation with rewards
        results = trainer.sample_meshes_with_rewards(
            images=[temp_image_path],
            prompts=["A simple 3D test object"],
            batch_size=1,
            num_inference_steps=5,
            guidance_scale=5.0,
            deterministic=True,
            kl_reward=0.0,
        )
        
        print(f"✅ Mesh generation successful!")
        print(f"   - Generated {len(results['meshes'])} meshes")
        print(f"   - Latents shape: {results['latents'].shape}")
        print(f"   - Log probs shape: {results['log_probs'].shape}")
        
        # Test reward computation
        rewards = trainer._compute_rewards_sync(
            results['meshes'], 
            results['images'], 
            results['prompts']
        )
        
        print(f"✅ Reward computation successful!")
        print(f"   - Rewards keys: {list(rewards.keys())}")
        for key, value in rewards.items():
            print(f"   - {key} score: {value[0]:.3f}")
        
        # Test reward function
        reward_fn = create_3d_reward_function(reward_config)
        rewards_dict, metadata = reward_fn(
            results['meshes'],
            results['images'],
            results['prompts']
        )
        
        print(f"✅ Reward function successful!")
        print(f"   - Metadata: {metadata}")
        
        # Cleanup
        os.remove(temp_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading and data preparation."""
    print("\n🧪 Testing dataset loading...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create test data
            train_dir, test_dir = create_test_data(temp_dir, num_samples=3)
            
            # Test dataset import
            sys.path.insert(0, str(project_root / "scripts"))
            from train_hunyuan3d import Image3DDataset
            
            # Temporarily replace logger with print for testing
            import train_hunyuan3d
            original_logger = train_hunyuan3d.logger
            
            class MockLogger:
                def info(self, msg):
                    print(f"INFO: {msg}")
            
            train_hunyuan3d.logger = MockLogger()
            
            # Create datasets
            train_dataset = Image3DDataset(
                image_dir=train_dir,
                prompts_file=os.path.join(temp_dir, "train_prompts.txt"),
                split="train"
            )
            
            test_dataset = Image3DDataset(
                image_dir=test_dir,
                prompts_file=os.path.join(temp_dir, "test_prompts.txt"),
                split="test"
            )
            
            # Restore original logger
            train_hunyuan3d.logger = original_logger
            
            print(f"✅ Dataset creation successful!")
            print(f"   - Train dataset: {len(train_dataset)} samples")
            print(f"   - Test dataset: {len(test_dataset)} samples")
            
            # Test data loading
            from torch.utils.data import DataLoader
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=2,
                shuffle=True,
                collate_fn=Image3DDataset.collate_fn,
                num_workers=0,  # Use 0 to avoid multiprocessing issues in test
            )
            
            # Test batch loading
            batch = next(iter(train_loader))
            image_paths, prompts, metadata = batch
            
            print(f"✅ Data loading successful!")
            print(f"   - Batch size: {len(image_paths)}")
            print(f"   - First prompt: {prompts[0]}")
            print(f"   - First image: {Path(image_paths[0]).name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_training_config():
    """Test training configuration and setup."""
    print("\n🧪 Testing training configuration...")
    
    try:
        # Import configuration
        sys.path.insert(0, str(project_root / "scripts"))
        from train_hunyuan3d import create_config
        
        config = create_config()
        
        print(f"✅ Configuration creation successful!")
        print(f"   - Epochs: {config.num_epochs}")
        print(f"   - Learning rate: {config.train.learning_rate}")
        print(f"   - Batch size: {config.sample.train_batch_size}")
        print(f"   - Inference steps: {config.sample.num_steps}")
        print(f"   - Mixed precision: {config.mixed_precision}")
        
        # Test configuration access
        assert hasattr(config, 'train')
        assert hasattr(config, 'sample')
        assert hasattr(config.train, 'learning_rate')
        assert hasattr(config.sample, 'num_steps')
        
        print(f"✅ Configuration validation successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("🚀 Running Hunyuan3D GRPO Training Tests\n")
    
    tests = [
        ("Configuration", test_training_config),
        ("Dataset Loading", test_dataset_loading),
        ("Pipeline with LogProb", test_pipeline_with_logprob),
        ("Trainer Components", test_trainer_components),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{'='*60}")
        print(f"Running {test_name} Test")
        print(f"{'='*60}")
        
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} test PASSED\n")
        else:
            print(f"❌ {test_name} test FAILED\n")
    
    # Summary
    print(f"{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The 3D training implementation is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
