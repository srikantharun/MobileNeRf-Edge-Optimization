"""
Model optimization utilities for edge deployment.
"""
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import cv2
from pathlib import Path

from .mobilenerf import MobileNeRF


def quantize_model(model: nn.Module, 
                   example_input: torch.Tensor, 
                   dynamic_quantization: bool = True) -> nn.Module:
    """
    Quantize model for efficient deployment.
    
    Args:
        model: PyTorch model to quantize
        example_input: Example input tensor
        dynamic_quantization: Whether to use dynamic quantization
        
    Returns:
        Quantized model
    """
    model.eval()
    
    if dynamic_quantization:
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # Quantize only linear layers
            dtype=torch.qint8
        )
    else:
        # Static quantization requires calibration
        # Set up quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with example data
        with torch.no_grad():
            model(example_input)
            
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
    
    return quantized_model


def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Prune model for efficient deployment.
    
    Args:
        model: PyTorch model to prune
        amount: Amount of weights to prune (0.0 to 1.0)
        
    Returns:
        Pruned model
    """
    # Clone model to avoid modifying original
    pruned_model = type(model)()
    pruned_model.load_state_dict(model.state_dict())
    
    # Iterate through layers and apply pruning
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # L1 unstructured pruning
            torch.nn.utils.prune.l1_unstructured(
                module, 
                name='weight', 
                amount=amount
            )
            
            # Make pruning permanent
            torch.nn.utils.prune.remove(module, 'weight')
    
    return pruned_model


def convert_to_onnx(model: nn.Module, 
                    output_path: str, 
                    example_input: torch.Tensor,
                    input_names: List[str] = ['input'],
                    output_names: List[str] = ['output'],
                    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None) -> str:
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to convert
        output_path: Path to save ONNX model
        example_input: Example input tensor
        input_names: Names of input tensors
        output_names: Names of output tensors
        dynamic_axes: Dynamic axes configuration
        
    Returns:
        Path to the exported ONNX model
    """
    model.eval()
    
    # Set default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 1: 'height', 2: 'width'}
        }
    
    # Export model to ONNX
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Model exported to {output_path}")
    return output_path


def optimize_onnx_model(model_path: str, target: str = 'generic') -> str:
    """
    Optimize ONNX model for specific targets.
    
    Args:
        model_path: Path to ONNX model
        target: Target device ('generic', 'jetson', 'raspberrypi', 'intel')
        
    Returns:
        Path to optimized model
    """
    # Load ONNX model
    onnx_model = onnx.load(model_path)
    
    # Apply optimizations
    from onnxruntime.transformers import optimizer
    
    # Configure optimization level
    opt_options = optimizer.OptimizationOptions()
    opt_options.enable_gelu_approximation = True
    opt_options.enable_layer_norm_optimization = True
    
    # Device-specific optimizations
    if target == 'jetson':
        # Optimize for NVIDIA Jetson devices
        opt_options.enable_tensorrt_acceleration = True
    elif target == 'raspberrypi':
        # Optimize for ARM CPUs
        opt_options.optimize_for_arm = True
    elif target == 'intel':
        # Optimize for Intel CPUs
        opt_options.enable_openvino_acceleration = True
    
    # Apply optimizations
    optimized_model = optimizer.optimize_model(
        model_path,
        'bert',  # Model type is irrelevant here
        opt_options,
        opt_level=99
    )
    
    # Save optimized model
    optimized_path = os.path.splitext(model_path)[0] + f"_optimized_{target}.onnx"
    optimized_model.save_model_to_file(optimized_path)
    
    return optimized_path


def convert_to_openvino(onnx_model_path: str) -> str:
    """
    Convert ONNX model to OpenVINO IR format.
    
    Args:
        onnx_model_path: Path to ONNX model
        
    Returns:
        Path to OpenVINO model
    """
    try:
        from openvino.tools import mo
        
        model_name = Path(onnx_model_path).stem
        output_dir = Path(onnx_model_path).parent
        
        # Run model optimizer
        model_xml = output_dir / f"{model_name}.xml"
        
        # Convert to IR format
        mo.convert_model(
            onnx_model_path,
            model_name=model_name,
            output_dir=str(output_dir),
            compress_to_fp16=True  # Use FP16 for better performance on edge
        )
        
        return str(model_xml)
    
    except ImportError:
        print("OpenVINO not available. Please install it with:")
        print("pip install openvino-dev")
        return onnx_model_path


def convert_for_opencv(model_path: str, opencv_output_path: str) -> str:
    """
    Convert model to a format optimized for OpenCV DNN.
    
    Args:
        model_path: Path to input model (ONNX or OpenVINO)
        opencv_output_path: Path to save OpenCV compatible model
        
    Returns:
        Path to OpenCV model
    """
    # For ONNX models, we can use them directly with OpenCV DNN
    if model_path.endswith('.onnx'):
        # Just copy the file since OpenCV DNN can read ONNX directly
        import shutil
        shutil.copy(model_path, opencv_output_path)
        
        # Test loading with OpenCV
        net = cv2.dnn.readNetFromONNX(opencv_output_path)
        
        # Set backend and target
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        print(f"Model converted for OpenCV DNN: {opencv_output_path}")
        return opencv_output_path
    
    # For OpenVINO IR models
    elif model_path.endswith('.xml'):
        # OpenCV can directly use OpenVINO IR models
        import shutil
        
        # Copy both .xml and .bin files
        shutil.copy(model_path, opencv_output_path)
        
        bin_path = model_path.replace('.xml', '.bin')
        opencv_bin_path = opencv_output_path.replace('.xml', '.bin')
        shutil.copy(bin_path, opencv_bin_path)
        
        # Test loading with OpenCV
        net = cv2.dnn.readNetFromModelOptimizer(
            opencv_output_path,
            opencv_bin_path
        )
        
        print(f"Model converted for OpenCV DNN: {opencv_output_path}")
        return opencv_output_path
    
    else:
        raise ValueError(f"Unsupported model format: {model_path}")


def optimize_for_edge(model: MobileNeRF, 
                     target_device: str,
                     output_dir: str,
                     example_input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256)) -> Dict[str, str]:
    """
    Optimize a MobileNeRF model for edge deployment.
    
    Args:
        model: MobileNeRF model to optimize
        target_device: Target device ('jetson', 'raspberrypi', 'intel')
        output_dir: Directory to save optimized models
        example_input_shape: Shape of example input for conversion
        
    Returns:
        Dictionary with paths to optimized models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create example input
    example_input = torch.randn(example_input_shape)
    
    # Step 1: Quantize
    print("Quantizing model...")
    quantized_model = quantize_model(model, example_input)
    
    # Step 2: Prune
    print("Pruning model...")
    pruned_model = prune_model(quantized_model)
    
    # Step 3: Export to ONNX
    print("Exporting to ONNX...")
    onnx_path = str(output_dir / "mobilenerf_optimized.onnx")
    convert_to_onnx(pruned_model, onnx_path, example_input)
    
    # Step 4: Optimize ONNX for target
    print(f"Optimizing ONNX for {target_device}...")
    optimized_onnx_path = optimize_onnx_model(onnx_path, target_device)
    
    # Step 5: Convert to target specific format if needed
    result_paths = {
        'onnx': onnx_path,
        'optimized_onnx': optimized_onnx_path
    }
    
    if target_device == 'intel':
        # Convert to OpenVINO for Intel devices
        print("Converting to OpenVINO...")
        openvino_path = convert_to_openvino(optimized_onnx_path)
        result_paths['openvino'] = openvino_path
    
    # Step 6: Create OpenCV DNN compatible model
    print("Creating OpenCV DNN compatible model...")
    opencv_path = str(output_dir / "mobilenerf_opencv.onnx")
    opencv_model_path = convert_for_opencv(
        optimized_onnx_path, 
        opencv_path
    )
    result_paths['opencv'] = opencv_model_path
    
    print("Optimization complete!")
    return result_paths


def benchmark_model(model_path: str, 
                   input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
                   num_iterations: int = 100,
                   warmup_iterations: int = 10) -> Dict[str, float]:
    """
    Benchmark a model on the current device.
    
    Args:
        model_path: Path to model file
        input_shape: Input tensor shape
        num_iterations: Number of iterations for benchmarking
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    # Create random input data
    input_data = np.random.rand(*input_shape).astype(np.float32)
    
    # ONNX Runtime benchmark
    if model_path.endswith('.onnx'):
        # Create ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, session_options)
        
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(warmup_iterations):
            session.run(None, {input_name: input_data})
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(num_iterations):
            session.run(None, {input_name: input_data})
            
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = num_iterations / total_time
        
        results['runtime'] = 'onnxruntime'
        results['avg_inference_time_ms'] = avg_time * 1000
        results['fps'] = fps
    
    # OpenCV DNN benchmark
    if model_path.endswith('.onnx') or model_path.endswith('.xml'):
        # Load model with OpenCV DNN
        if model_path.endswith('.onnx'):
            net = cv2.dnn.readNetFromONNX(model_path)
        else:
            bin_path = model_path.replace('.xml', '.bin')
            net = cv2.dnn.readNetFromModelOptimizer(model_path, bin_path)
        
        # Set backend and target
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            input_data.transpose(0, 2, 3, 1)[0], 
            1.0, 
            (input_shape[2], input_shape[3]),
            (0, 0, 0),
            swapRB=False,
            crop=False
        )
        
        # Warmup
        for _ in range(warmup_iterations):
            net.setInput(blob)
            net.forward()
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(num_iterations):
            net.setInput(blob)
            net.forward()
            
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = num_iterations / total_time
        
        results['opencv_avg_inference_time_ms'] = avg_time * 1000
        results['opencv_fps'] = fps
    
    return results
