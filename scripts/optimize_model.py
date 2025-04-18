#!/usr/bin/env python
"""
Model optimization script for edge deployment.
"""
import argparse
import yaml
import os
import torch
from pathlib import Path
import time

from src.models.mobilenerf import MobileNeRF, load_pretrained_mobilenerf
from src.models.optimization import optimize_for_edge, benchmark_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimize MobileNeRF model for edge deployment')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained PyTorch model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, help='Directory to save optimized models')
    parser.add_argument('--target', type=str, 
                        choices=['generic', 'jetson', 'raspberrypi', 'intel'],
                        help='Target device for optimization')
    parser.add_argument('--benchmark', action='store_true', 
                        help='Run benchmark on optimized models')
    parser.add_argument('--input-size', type=str, default='1,3,256,256',
                        help='Input tensor size for the model as comma-separated values')
    
    return parser.parse_args()


def main():
    """Main optimization function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.output_dir:
        config['optimization']['output_dir'] = args.output_dir
    if args.target:
        config['optimization']['target_device'] = args.target
    
    # Create output directory
    output_dir = Path(config['optimization']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Parse input size
    input_size = tuple(map(int, args.input_size.split(',')))
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_pretrained_mobilenerf(args.model_path, 'cpu')
    
    # Optimize model for edge deployment
    print(f"Optimizing model for {config['optimization']['target_device']}...")
    optimized_paths = optimize_for_edge(
        model=model,
        target_device=config['optimization']['target_device'],
        output_dir=str(output_dir),
        example_input_shape=input_size
    )
    
    # Save optimization config
    optimization_info = {
        'original_model': args.model_path,
        'target_device': config['optimization']['target_device'],
        'input_size': list(input_size),
        'optimized_models': {k: str(v) for k, v in optimized_paths.items()}
    }
    
    with open(output_dir / 'optimization_info.yaml', 'w') as f:
        yaml.dump(optimization_info, f)
    
    # Run benchmark if requested
    if args.benchmark:
        print("Running benchmark...")
        
        # Benchmark each optimized model
        benchmark_results = {}
        
        for model_type, model_path in optimized_paths.items():
            print(f"Benchmarking {model_type} model...")
            
            benchmark_results[model_type] = benchmark_model(
                model_path=model_path,
                input_shape=input_size,
                num_iterations=100,
                warmup_iterations=10
            )
        
        # Save benchmark results
        with open(output_dir / 'benchmark_results.yaml', 'w') as f:
            yaml.dump(benchmark_results, f)
            
        # Print summary
        print("\nBenchmark Summary:")
        for model_type, results in benchmark_results.items():
            if 'avg_inference_time_ms' in results:
                print(f"  {model_type}: {results['avg_inference_time_ms']:.2f} ms/inference " +
                      f"({results['fps']:.2f} FPS)")
    
    print("\nOptimization complete!")
    print(f"Optimized models saved to {output_dir}")


if __name__ == '__main__':
    main()
