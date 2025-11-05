#!/usr/bin/env python3
"""
Simple checkpoint structure checker
"""

import torch
import sys

def check_checkpoint_structure(model_path):
    """Check the structure of a checkpoint file"""

    print(f"Loading checkpoint: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Checkpoint type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print(f"Dictionary with {len(checkpoint)} keys")
            print("Keys:")
            for i, key in enumerate(checkpoint.keys()):
                value = checkpoint[key]
                if hasattr(value, 'shape'):
                    print(f"  {i}. {key}: {type(value)} shape={value.shape}")
                else:
                    print(f"  {i}. {key}: {type(value)}")

        elif isinstance(checkpoint, tuple) or isinstance(checkpoint, list):
            print(f"Tuple/List with {len(checkpoint)} elements")
            print("Elements:")
            for i, elem in enumerate(checkpoint):
                if hasattr(elem, 'shape'):
                    print(f"  {i}. {type(elem)} shape={elem.shape}")
                else:
                    print(f"  {i}. {type(elem)}")

        else:
            print(f"Unknown format: {type(checkpoint)}")

        # Try to find xyz to infer number of Gaussians
        num_gaussians = None

        if isinstance(checkpoint, dict):
            if '_xyz' in checkpoint:
                num_gaussians = checkpoint['_xyz'].shape[0]
        elif isinstance(checkpoint, tuple) and len(checkpoint) > 1:
            if hasattr(checkpoint[1], 'shape'):
                num_gaussians = checkpoint[1].shape[0]

        if num_gaussians:
            print(f"\nNumber of Gaussians: {num_gaussians:,}")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_checkpoint.py path/to/chkpntXXXX.pth")
    else:
        check_checkpoint_structure(sys.argv[1])