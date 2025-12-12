"""
TensorRT Model Converter
Converts PyTorch YOLO models to TensorRT engines for optimized inference on Jetson Nano
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TensorRTConverter:
    """Convert YOLO models to TensorRT format"""
    
    def __init__(self, precision='fp16', workspace=4):
        """
        Initialize TensorRT converter
        
        Args:
            precision: Model precision ('fp32' or 'fp16')
            workspace: Maximum workspace size in GB
        """
        self.precision = precision
        self.workspace = workspace
        
    def convert_model(self, model_path, output_path=None, imgsz=640, batch_size=1):
        """
        Convert PyTorch model to TensorRT engine
        
        Args:
            model_path: Path to PyTorch model (.pt file)
            output_path: Path to save TensorRT engine (optional)
            imgsz: Input image size
            batch_size: Batch size for inference
            
        Returns:
            Path to generated TensorRT engine
        """
        try:
            logger.info(f"Loading model from {model_path}")
            model = YOLO(model_path)
            
            # Generate output path if not provided
            if output_path is None:
                model_name = Path(model_path).stem
                output_path = f"models/{model_name}.engine"
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logger.info(f"Converting to TensorRT with precision={self.precision}")
            logger.info(f"Image size: {imgsz}, Batch size: {batch_size}")
            logger.info("This may take 10-15 minutes...")
            
            # Export to TensorRT
            # Note: This requires TensorRT to be installed
            model.export(
                format='engine',
                imgsz=imgsz,
                half=(self.precision == 'fp16'),
                workspace=self.workspace,
                batch=batch_size,
                device=0  # GPU 0
            )
            
            # The export creates a .engine file in the same directory as the model
            source_engine = str(Path(model_path).with_suffix('.engine'))
            
            # Move to desired location if different
            if source_engine != output_path:
                import shutil
                shutil.move(source_engine, output_path)
            
            logger.info(f"✓ TensorRT engine saved to {output_path}")
            
            # Validate the engine
            self._validate_engine(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert model: {e}")
            raise
    
    def _validate_engine(self, engine_path):
        """Validate that the TensorRT engine works"""
        try:
            logger.info("Validating TensorRT engine...")
            
            # Try to load the engine
            model = YOLO(engine_path)
            
            # Run a test inference
            import numpy as np
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = model(test_image, verbose=False)
            
            logger.info("✓ Engine validation successful")
            
        except Exception as e:
            logger.error(f"Engine validation failed: {e}")
            raise
    
    def convert_batch(self, model_configs):
        """
        Convert multiple models in batch
        
        Args:
            model_configs: List of dicts with 'model_path', 'output_path', 'imgsz'
        """
        results = []
        
        for config in model_configs:
            try:
                engine_path = self.convert_model(**config)
                results.append({'success': True, 'path': engine_path})
            except Exception as e:
                results.append({'success': False, 'error': str(e)})
        
        return results


def main():
    """Command-line interface for model conversion"""
    parser = argparse.ArgumentParser(
        description='Convert YOLO models to TensorRT engines'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to PyTorch model (.pt file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for TensorRT engine'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp32', 'fp16'],
        default='fp16',
        help='Model precision (default: fp16)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=1,
        help='Batch size (default: 1)'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4,
        help='Maximum workspace size in GB (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = TensorRTConverter(
        precision=args.precision,
        workspace=args.workspace
    )
    
    # Convert model
    try:
        engine_path = converter.convert_model(
            model_path=args.model,
            output_path=args.output,
            imgsz=args.imgsz,
            batch_size=args.batch
        )
        logger.info(f"✓ Conversion complete: {engine_path}")
        return 0
    except Exception as e:
        logger.error(f"✗ Conversion failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
