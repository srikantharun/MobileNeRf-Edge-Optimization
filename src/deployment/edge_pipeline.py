"""
Edge deployment pipeline for MobileNeRF scene recreation.
"""
import cv2
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from pathlib import Path


class EdgeInferenceEngine:
    """
    Inference engine for running MobileNeRF on edge devices.
    """
    
    def __init__(self, 
                 model_path: str,
                 input_size: Tuple[int, int] = (256, 256),
                 use_gpu: bool = False,
                 optimized_for_opencv: bool = True):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model file (ONNX or OpenVINO IR)
            input_size: Input image size for the model
            use_gpu: Whether to use GPU acceleration if available
            optimized_for_opencv: Whether the model is optimized for OpenCV DNN
        """
        self.input_size = input_size
        self.model_path = model_path
        self.optimized_for_opencv = optimized_for_opencv
        
        # Load model with OpenCV DNN
        if model_path.endswith('.onnx'):
            self.net = cv2.dnn.readNetFromONNX(model_path)
        elif model_path.endswith('.xml'):
            bin_path = model_path.replace('.xml', '.bin')
            self.net = cv2.dnn.readNetFromModelOptimizer(model_path, bin_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
        
        # Configure backend and target
        if use_gpu:
            # Try to use GPU via different backends
            backends_targets = [
                (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA),
                (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL),
                (cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_OPENCL_FP16)
            ]
            
            for backend, target in backends_targets:
                try:
                    self.net.setPreferableBackend(backend)
                    self.net.setPreferableTarget(target)
                    # Test if this configuration works
                    dummy_input = np.zeros((1, 3, *input_size), dtype=np.float32)
                    self.net.setInput(cv2.dnn.blobFromImage(dummy_input))
                    self.net.forward()
                    print(f"Using backend {backend} with target {target}")
                    break
                except Exception as e:
                    print(f"Failed to use {backend} with {target}: {e}")
        else:
            # Use CPU
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU backend")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image blob
        """
        # Resize image
        resized = cv2.resize(image, self.input_size)
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            resized, 
            1.0 / 255.0,  # Scale factor
            self.input_size,
            (0.5, 0.5, 0.5),  # Mean subtraction
            swapRB=True,  # BGR to RGB
            crop=False
        )
        
        return blob
    
    def postprocess(self, outputs: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess network outputs.
        
        Args:
            outputs: Network outputs
            original_size: Original image size (height, width)
            
        Returns:
            Processed output image
        """
        # Extract the first output (assuming batch size 1)
        output = outputs[0]
        
        # Reshape and scale to 0-255 range
        output = output.transpose(1, 2, 0)  # CHW to HWC
        output = np.clip(output * a 255.0, 0, 255).astype(np.uint8)
        
        # Resize to original dimensions
        if output.shape[:2] != original_size:
            output = cv2.resize(output, (original_size[1], original_size[0]))
            
        return output
    
    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on an image.
        
        Args:
            image: Input image
            
        Returns:
            Output image
        """
        # Get original dimensions
        original_size = image.shape[:2]
        
        # Preprocess image
        blob = self.preprocess(image)
        
        # Set input and run inference
        self.net.setInput(blob)
        
        # Get outputs
        outputs = self.net.forward()
        
        # Postprocess outputs
        result = self.postprocess(outputs, original_size)
        
        return result


class EdgePipeline:
    """
    Real-time processing pipeline for edge deployment.
    """
    
    def __init__(self, 
                 model_path: str,
                 camera_id: int = 0,
                 input_size: Tuple[int, int] = (256, 256),
                 use_gpu: bool = False,
                 detection_threshold: float = 0.2,
                 max_queue_size: int = 10):
        """
        Initialize the edge pipeline.
        
        Args:
            model_path: Path to the model file
            camera_id: Camera device ID
            input_size: Input size for the model
            use_gpu: Whether to use GPU acceleration
            detection_threshold: Threshold for scene change detection
            max_queue_size: Maximum size of the frame queue
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.input_size = input_size
        self.detection_threshold = detection_threshold
        
        # Initialize the inference engine
        self.engine = EdgeInferenceEngine(
            model_path=model_path,
            input_size=input_size,
            use_gpu=use_gpu
        )
        
        # Initialize the camera
        self.cap = None
        
        # Create queues for multi-threaded processing
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        
        # Create threading events
        self.stop_event = threading.Event()
        
        # Previous frame for change detection
        self.prev_frame = None
        
    def start(self):
        """
        Start the processing pipeline.
        """
        # Open the camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start worker threads
        self.capture_thread = threading.Thread(target=self._capture_worker)
        self.inference_thread = threading.Thread(target=self._inference_worker)
        
        self.capture_thread.daemon = True
        self.inference_thread.daemon = True
        
        self.stop_event.clear()
        self.capture_thread.start()
        self.inference_thread.start()
        
        print("Pipeline started")
        
    def stop(self):
        """
        Stop the processing pipeline.
        """
        self.stop_event.set()
        
        # Wait for threads to finish
        self.capture_thread.join(timeout=1.0)
        self.inference_thread.join(timeout=1.0)
        
        # Release the camera
        if self.cap:
            self.cap.release()
            
        print("Pipeline stopped")
        
    def _capture_worker(self):
        """
        Worker thread for capturing frames.
        """
        while not self.stop_event.is_set():
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            # Detect significant changes
            if self.prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(frame, self.prev_frame)
                diff_mean = np.mean(diff)
                
                # Skip processing if minimal change
                if diff_mean < self.detection_threshold:
                    time.sleep(0.01)  # Short sleep to reduce CPU usage
                    continue
            
            # Update previous frame
            self.prev_frame = frame.copy()
            
            try:
                # Add to processing queue (non-blocking)
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Skip frame if queue is full
                pass
                
    def _inference_worker(self):
        """
        Worker thread for running inference.
        """
        while not self.stop_event.is_set():
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Run inference
                start_time = time.time()
                result = self.engine.infer(frame)
                inference_time = time.time() - start_time
                
                # Add result to output queue
                self.result_queue.put({
                    'original': frame,
                    'result': result,
                    'inference_time': inference_time
                })
                
                # Mark task as done
                self.frame_queue.task_done()
                
            except queue.Empty:
                # No frames available
                continue
                
    def get_latest_result(self, block: bool = False, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Get the latest processing result.
        
        Args:
            block: Whether to block until a result is available
            timeout: Timeout for blocking
            
        Returns:
            Dictionary with original frame, result and inference time,
            or None if no result is available
        """
        try:
            return self.result_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
            
    def run_demo(self, output_path: Optional[str] = None, max_frames: int = 1000):
        """
        Run a demo of the pipeline with visualization.
        
        Args:
            output_path: Path to save the output video (optional)
            max_frames: Maximum number of frames to process
        """
        # Start the pipeline
        self.start()
        
        # Create video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                30.0, 
                (640, 480)
            )
        
        try:
            frame_count = 0
            
            while frame_count < max_frames and not self.stop_event.is_set():
                # Get latest result
                result_data = self.get_latest_result(block=True, timeout=1.0)
                
                if result_data:
                    original = result_data['original']
                    result = result_data['result']
                    inference_time = result_data['inference_time']
                    
                    # Create visualization
                    # Display original and result side by side
                    viz = np.hstack([original, result])
                    
                    # Add inference time text
                    fps = 1.0 / inference_time if inference_time > 0 else 0
                    cv2.putText(
                        viz, 
                        f"Inference: {inference_time*1000:.1f}ms ({fps:.1f} FPS)",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Display result
                    cv2.imshow("MobileNeRF Edge Demo", viz)
                    
                    # Write to video if enabled
                    if writer:
                        writer.write(viz)
                        
                    frame_count += 1
                    
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    break
                    
        finally:
            # Stop the pipeline
            self.stop()
            
            # Release video writer
            if writer:
                writer.release()
                
            # Close all windows
            cv2.destroyAllWindows()


def run_edge_pipeline(model_path: str, 
                     camera_id: int = 0,
                     use_gpu: bool = False,
                     output_path: Optional[str] = None,
                     max_frames: int = 1000):
    """
    Run the edge pipeline with the specified model.
    
    Args:
        model_path: Path to the model file
        camera_id: Camera device ID
        use_gpu: Whether to use GPU acceleration
        output_path: Path to save the output video (optional)
        max_frames: Maximum number of frames to process
    """
    pipeline = EdgePipeline(
        model_path=model_path,
        camera_id=camera_id,
        use_gpu=use_gpu
    )
    
    pipeline.run_demo(output_path=output_path, max_frames=max_frames)
