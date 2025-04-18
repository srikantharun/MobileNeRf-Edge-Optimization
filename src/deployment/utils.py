"""
Utility functions for edge deployment.
"""
import cv2
import numpy as np
import os
import time
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import psutil
import threading
from datetime import datetime


def get_device_info() -> Dict[str, Any]:
    """
    Get information about the device.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cpu': {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'usage_percent': psutil.cpu_percent(interval=0.1),
            'frequency': {
                'current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'min': psutil.cpu_freq().min if psutil.cpu_freq() else None,
                'max': psutil.cpu_freq().max if psutil.cpu_freq() else None
            }
        },
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'percent_used': psutil.virtual_memory().percent
        },
        'disk': {
            'total': psutil.disk_usage('/').total,
            'free': psutil.disk_usage('/').free,
            'percent_used': psutil.disk_usage('/').percent
        },
        'platform': {
            'system': os.uname().sysname if hasattr(os, 'uname') else 'Unknown',
            'release': os.uname().release if hasattr(os, 'uname') else 'Unknown',
            'version': os.uname().version if hasattr(os, 'uname') else 'Unknown',
            'machine': os.uname().machine if hasattr(os, 'uname') else 'Unknown'
        }
    }
    
    # Check for GPU information (NVIDIA)
    try:
        nvidia_smi_output = os.popen('nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader').read().strip()
        if nvidia_smi_output:
            gpu_info = []
            for line in nvidia_smi_output.split('\n'):
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 4:
                    gpu_info.append({
                        'name': parts[0],
                        'memory_total': parts[1],
                        'memory_used': parts[2],
                        'temperature': parts[3]
                    })
            info['gpu'] = {
                'vendor': 'NVIDIA',
                'devices': gpu_info
            }
    except:
        pass
    
    # Check for OpenCL devices
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        opencl_devices = []
        
        for platform in platforms:
            devices = platform.get_devices()
            for device in devices:
                opencl_devices.append({
                    'name': device.name,
                    'vendor': device.vendor,
                    'type': 'GPU' if device.type == cl.device_type.GPU else 'CPU',
                    'max_compute_units': device.max_compute_units,
                    'global_mem_size': device.global_mem_size
                })
                
        if opencl_devices:
            info['opencl'] = {
                'devices': opencl_devices
            }
    except:
        pass
    
    return info


def monitor_resource_usage(interval: float = 1.0, duration: float = 60.0) -> Dict[str, List[Dict[str, Any]]]:
    """
    Monitor CPU, memory, and GPU usage over time.
    
    Args:
        interval: Sampling interval in seconds
        duration: Total duration to monitor in seconds
        
    Returns:
        Dictionary with usage statistics
    """
    stats = {
        'timestamps': [],
        'cpu': [],
        'memory': [],
        'gpu': []
    }
    
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        # Record timestamp
        current_time = time.time()
        stats['timestamps'].append(current_time - start_time)
        
        # CPU usage
        stats['cpu'].append({
            'percent': psutil.cpu_percent(interval=0.1)
        })
        
        # Memory usage
        memory = psutil.virtual_memory()
        stats['memory'].append({
            'percent': memory.percent,
            'used': memory.used,
            'available': memory.available
        })
        
        # GPU usage (NVIDIA)
        gpu_stats = []
        try:
            nvidia_smi_output = os.popen('nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader').read().strip()
            if nvidia_smi_output:
                for line in nvidia_smi_output.split('\n'):
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 3:
                        gpu_stats.append({
                            'utilization': parts[0],
                            'memory_utilization': parts[1],
                            'memory_used': parts[2]
                        })
        except:
            pass
            
        stats['gpu'].append(gpu_stats)
        
        # Sleep until next sampling point
        sleep_time = max(0, interval - (time.time() - current_time))
        time.sleep(sleep_time)
        
    return stats


class ResourceMonitor:
    """
    Monitor system resources during model inference.
    """
    
    def __init__(self, log_dir: Optional[str] = None, interval: float = 0.5):
        """
        Initialize the resource monitor.
        
        Args:
            log_dir: Directory to save log files
            interval: Sampling interval in seconds
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.interval = interval
        self.stats = {
            'timestamps': [],
            'cpu': [],
            'memory': [],
            'gpu': []
        }
        self.is_running = False
        self.monitor_thread = None
        
        # Create log directory if specified
        if self.log_dir:
            self.log_dir.mkdir(exist_ok=True, parents=True)
        
    def start(self):
        """
        Start monitoring resources.
        """
        if self.is_running:
            return
            
        self.is_running = True
        self.start_time = time.time()
        self.stats = {
            'timestamps': [],
            'cpu': [],
            'memory': [],
            'gpu': []
        }
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_worker)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self) -> Dict[str, Any]:
        """
        Stop monitoring and return statistics.
        
        Returns:
            Dictionary with resource usage statistics
        """
        if not self.is_running:
            return self.stats
            
        self.is_running = False
        
        # Wait for monitoring thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        # Save statistics if log directory is specified
        if self.log_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = self.log_dir / f'resource_usage_{timestamp}.json'
            
            with open(log_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
                
        return self.stats
        
    def _monitor_worker(self):
        """
        Worker function for monitoring thread.
        """
        while self.is_running:
            # Record timestamp
            current_time = time.time()
            self.stats['timestamps'].append(current_time - self.start_time)
            
            # CPU usage
            self.stats['cpu'].append({
                'percent': psutil.cpu_percent(interval=0.1)
            })
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.stats['memory'].append({
                'percent': memory.percent,
                'used': memory.used,
                'available': memory.available
            })
            
            # GPU usage (NVIDIA)
            gpu_stats = []
            try:
                nvidia_smi_output = os.popen('nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader').read().strip()
                if nvidia_smi_output:
                    for line in nvidia_smi_output.split('\n'):
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) >= 3:
                            gpu_stats.append({
                                'utilization': parts[0],
                                'memory_utilization': parts[1],
                                'memory_used': parts[2]
                            })
            except:
                pass
                
            self.stats['gpu'].append(gpu_stats)
            
            # Sleep until next sampling point
            time.sleep(self.interval)


def visualize_model_architecture(model_path: str, output_path: str) -> str:
    """
    Visualize the architecture of an ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        output_path: Path to save the visualization
        
    Returns:
        Path to the visualization image
    """
    try:
        import onnx
        from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
        
        # Load the ONNX model
        model = onnx.load(model_path)
        
        # Create graph
        pydot_graph = GetPydotGraph(
            model.graph,
            name=model.graph.name,
            rankdir="TB",
            node_producer=GetOpNodeProducer("docstring")
        )
        
        # Save visualization
        pydot_graph.write_png(output_path)
        
        return output_path
        
    except ImportError:
        print("Cannot visualize model architecture: onnx or pydot not installed.")
        print("Install with: pip install onnx pydot")
        return ""
