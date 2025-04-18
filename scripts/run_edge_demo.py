#!/usr/bin/env python
"""
Demo script for running MobileNeRF on edge devices.
"""
import argparse
import yaml
import os
from pathlib import Path
import time
import cv2

from src.deployment.edge_pipeline import run_edge_pipeline
from src.deployment.utils import get_device_info, ResourceMonitor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run MobileNeRF Edge demo')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to optimized model file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--output-video', type=str,
                        help='Path to save output video')
    parser.add_argument('--config', type=str, default='configs/edge_device_configs/generic.yaml',
                        help='Path to edge device configuration file')
    parser.add_argument('--monitor-resources', action='store_true',
                        help='Monitor system resources during inference')
    parser.add_argument('--max-frames', type=int, default=1000,
                        help='Maximum number of frames to process')
    
    return parser.parse_args()


def main():
    """Main demo function."""
    args = parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Load configuration if available
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Print device information
    print("Device Information:")
    device_info = get_device_info()
    print(f"  CPU: {device_info['cpu']['cores']} cores, {device_info['cpu']['threads']} threads")
    print(f"  Memory: {device_info['memory']['total'] / (1024**3):.1f} GB total, " +
          f"{device_info['memory']['percent_used']}% used")
    
    if 'gpu' in device_info:
        for i, gpu in enumerate(device_info['gpu']['devices']):
            print(f"  GPU {i}: {gpu['name']}, {gpu['memory_total']}, Temperature: {gpu['temperature']}")
    
    # Create output directory if needed
    if args.output_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
    
    # Start resource monitoring if requested
    monitor = None
    if args.monitor_resources:
        log_dir = Path('logs') / time.strftime('%Y%m%d_%H%M%S')
        log_dir.mkdir(exist_ok=True, parents=True)
        
        monitor = ResourceMonitor(log_dir=str(log_dir))
        monitor.start()
        print(f"Resource monitoring started. Logs will be saved to {log_dir}")
    
    try:
        # Run edge pipeline
        print(f"Running edge pipeline with model: {args.model_path}")
        print(f"Camera ID: {args.camera}")
        print(f"Using GPU: {args.use_gpu}")
        
        run_edge_pipeline(
            model_path=args.model_path,
            camera_id=args.camera,
            use_gpu=args.use_gpu,
            output_path=args.output_video,
            max_frames=args.max_frames
        )
        
    finally:
        # Stop resource monitoring if started
        if monitor:
            stats = monitor.stop()
            print("Resource monitoring stopped")
            
            # Print summary
            cpu_avg = sum(data['percent'] for data in stats['cpu']) / len(stats['cpu'])
            memory_avg = sum(data['percent'] for data in stats['memory']) / len(stats['memory'])
            
            print("\nResource Usage Summary:")
            print(f"  Average CPU Usage: {cpu_avg:.1f}%")
            print(f"  Average Memory Usage: {memory_avg:.1f}%")
            
            if stats['gpu'] and stats['gpu'][0]:
                gpu_util = []
                for gpu_data in stats['gpu']:
                    if gpu_data:
                        try:
                            util = float(gpu_data[0]['utilization'].replace('%', ''))
                            gpu_util.append(util)
                        except (ValueError, KeyError, IndexError):
                            pass
                
                if gpu_util:
                    gpu_avg = sum(gpu_util) / len(gpu_util)
                    print(f"  Average GPU Usage: {gpu_avg:.1f}%")
    
    print("Demo completed!")


if __name__ == '__main__':
    main()
