"""
Benchmark System Performance
Test inference speed and system resource usage
"""

import cv2
import time
import yaml
import argparse
import numpy as np
from tensorrt_detector import VehicleDetector, LicensePlateDetector
from performance_monitor import PerformanceMonitor

def benchmark_detector(model_path, use_tensorrt=True, num_frames=100):
    """Benchmark detector performance"""
    print(f"\nBenchmarking detector: {model_path}")
    print(f"TensorRT: {use_tensorrt}")
    print(f"Frames: {num_frames}")
    
    # Initialize detector
    detector = VehicleDetector(model_path, use_tensorrt=use_tensorrt)
    
    # Create test frames
    test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        detector.detect_vehicles(test_frame)
    
    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    
    for i in range(num_frames):
        detections = detector.detect_vehicles(test_frame)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{num_frames} - {fps:.1f} FPS")
    
    total_time = time.time() - start_time
    avg_fps = num_frames / total_time
    avg_inference_ms = (total_time / num_frames) * 1000
    
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average inference time: {avg_inference_ms:.2f}ms")
    print("="*50)
    
    return {
        'fps': avg_fps,
        'inference_ms': avg_inference_ms,
        'total_time': total_time
    }

def benchmark_system(config_path='config.yaml'):
    """Benchmark full system"""
    print("\nBenchmarking full system...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    vehicle_detector = VehicleDetector(
        model_path=config['detection']['vehicle_model'],
        use_tensorrt=config['detection']['use_tensorrt']
    )
    
    plate_detector = LicensePlateDetector(
        model_path=config['detection']['plate_model'],
        use_tensorrt=config['detection']['use_tensorrt']
    )
    
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)
    
    # Benchmark
    num_frames = 100
    print(f"Processing {num_frames} frames...")
    
    for i in range(num_frames):
        start = time.time()
        
        # Vehicle detection
        vehicles = vehicle_detector.detect_vehicles(test_frame)
        
        # Plate detection (if vehicles found)
        if vehicles:
            bbox = vehicles[0]['bbox']
            plates = plate_detector.detect_plate_in_vehicle(test_frame, bbox)
        
        # Record metrics
        inference_time = time.time() - start
        monitor.record_frame()
        monitor.record_inference(inference_time)
        
        if (i + 1) % 20 == 0:
            metrics = monitor.get_metrics()
            print(f"  {i+1}/{num_frames} - FPS: {metrics['fps']:.1f}, "
                  f"Inference: {metrics['avg_inference_ms']:.1f}ms")
    
    # Results
    monitor.stop()
    print(monitor.get_summary())

def compare_tensorrt_pytorch(model_path):
    """Compare TensorRT vs PyTorch performance"""
    print("\n" + "="*60)
    print("TensorRT vs PyTorch Comparison")
    print("="*60)
    
    # Benchmark PyTorch
    print("\n1. PyTorch Model:")
    pytorch_results = benchmark_detector(model_path, use_tensorrt=False, num_frames=50)
    
    # Benchmark TensorRT
    print("\n2. TensorRT Engine:")
    tensorrt_results = benchmark_detector(model_path, use_tensorrt=True, num_frames=50)
    
    # Comparison
    speedup = tensorrt_results['fps'] / pytorch_results['fps']
    
    print("\n" + "="*60)
    print("Comparison Summary:")
    print("="*60)
    print(f"PyTorch FPS: {pytorch_results['fps']:.2f}")
    print(f"TensorRT FPS: {tensorrt_results['fps']:.2f}")
    print(f"Speedup: {speedup:.2f}x")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Benchmark System Performance')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--model', type=str, help='Model to benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare TensorRT vs PyTorch')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames')
    
    args = parser.parse_args()
    
    if args.compare and args.model:
        compare_tensorrt_pytorch(args.model)
    elif args.model:
        benchmark_detector(args.model, use_tensorrt=True, num_frames=args.frames)
    else:
        benchmark_system(args.config)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
