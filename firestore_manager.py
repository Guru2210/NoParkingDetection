"""
Firestore Database Manager
Handles cloud storage of parking violations with offline queue support
"""

import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import firebase_admin
from firebase_admin import credentials, firestore, storage
from collections import deque
import threading

logger = logging.getLogger(__name__)


class FirestoreManager:
    """Manage Firestore database operations for parking violations"""
    
    def __init__(self, credentials_path, collection_name='parking_violations',
                 storage_bucket=None, offline_queue=True):
        """
        Initialize Firestore manager
        
        Args:
            credentials_path: Path to Firebase service account JSON
            collection_name: Firestore collection name
            storage_bucket: Firebase storage bucket name
            offline_queue: Enable offline queueing
        """
        self.credentials_path = credentials_path
        self.collection_name = collection_name
        self.storage_bucket = storage_bucket
        self.offline_queue_enabled = offline_queue
        
        self.db = None
        self.bucket = None
        self.is_connected = False
        
        # Offline queue
        self.offline_queue = deque()
        self.queue_lock = threading.Lock()
        
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            # Check if credentials file exists
            if not os.path.exists(self.credentials_path):
                logger.error(f"Firebase credentials not found: {self.credentials_path}")
                logger.warning("⚠ Firestore disabled. Violations will be saved locally only.")
                return
            
            # Initialize Firebase Admin SDK
            cred = credentials.Certificate(self.credentials_path)
            
            # Check if already initialized
            try:
                firebase_admin.get_app()
                logger.info("Firebase already initialized")
            except ValueError:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': self.storage_bucket
                })
                logger.info("Firebase initialized successfully")
            
            # Get Firestore client
            self.db = firestore.client()
            
            # Get Storage bucket if configured
            if self.storage_bucket:
                self.bucket = storage.bucket()
            
            self.is_connected = True
            logger.info(f"✓ Connected to Firestore collection: {self.collection_name}")
            
            # Process any queued violations
            if self.offline_queue_enabled:
                self._process_offline_queue()
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            logger.warning("⚠ Firestore disabled. Violations will be saved locally only.")
            self.is_connected = False
    
    def save_violation(self, violation_data: Dict, vehicle_image=None, 
                      plate_image=None) -> Optional[str]:
        """
        Save parking violation to Firestore
        
        Args:
            violation_data: Dict containing violation information
            vehicle_image: Vehicle image (numpy array or path)
            plate_image: License plate image (numpy array or path)
            
        Returns:
            Violation ID if successful, None otherwise
        """
        try:
            # Generate violation ID
            violation_id = self._generate_violation_id()
            
            # Prepare violation record
            record = {
                'violation_id': violation_id,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'vehicle_id': violation_data.get('vehicle_id'),
                'license_plate': violation_data.get('license_plate', ''),
                'confidence': violation_data.get('confidence', 0.0),
                'parking_duration': violation_data.get('parking_duration', 0),
                'location': violation_data.get('location', 'Unknown'),
                'zone': violation_data.get('zone', 'Default'),
                'status': 'pending'
            }
            
            # Upload images if provided
            if vehicle_image is not None:
                vehicle_url = self._upload_image(
                    vehicle_image,
                    f"violations/{violation_id}/vehicle.jpg"
                )
                record['vehicle_image_url'] = vehicle_url
            
            if plate_image is not None:
                plate_url = self._upload_image(
                    plate_image,
                    f"violations/{violation_id}/plate.jpg"
                )
                record['plate_image_url'] = plate_url
            
            # Save to Firestore
            if self.is_connected:
                self.db.collection(self.collection_name).document(violation_id).set(record)
                logger.info(f"✓ Violation saved to Firestore: {violation_id}")
                return violation_id
            else:
                # Queue for later if offline
                if self.offline_queue_enabled:
                    self._queue_violation(record)
                    logger.info(f"⚠ Violation queued (offline): {violation_id}")
                return violation_id
                
        except Exception as e:
            logger.error(f"Failed to save violation: {e}")
            
            # Queue for retry
            if self.offline_queue_enabled:
                self._queue_violation(record)
            
            return None
    
    def _upload_image(self, image, destination_path):
        """
        Upload image to Firebase Storage
        
        Args:
            image: Image data (numpy array or file path)
            destination_path: Storage path
            
        Returns:
            Public URL of uploaded image
        """
        try:
            import cv2
            import tempfile
            
            # If image is numpy array, save to temp file
            if hasattr(image, 'shape'):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                cv2.imwrite(temp_file.name, image)
                image_path = temp_file.name
            else:
                image_path = image
            
            # Upload to storage
            if self.bucket:
                blob = self.bucket.blob(destination_path)
                blob.upload_from_filename(image_path)
                blob.make_public()
                
                # Clean up temp file
                if hasattr(image, 'shape'):
                    os.remove(image_path)
                
                return blob.public_url
            else:
                logger.warning("Storage bucket not configured")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
            return ""
    
    def _generate_violation_id(self):
        """Generate unique violation ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import random
        random_suffix = random.randint(1000, 9999)
        return f"VIO_{timestamp}_{random_suffix}"
    
    def _queue_violation(self, record):
        """Add violation to offline queue"""
        with self.queue_lock:
            self.offline_queue.append(record)
            logger.debug(f"Queued violation. Queue size: {len(self.offline_queue)}")
    
    def _process_offline_queue(self):
        """Process queued violations"""
        if not self.is_connected:
            return
        
        with self.queue_lock:
            processed = 0
            failed = []
            
            while self.offline_queue:
                record = self.offline_queue.popleft()
                
                try:
                    violation_id = record['violation_id']
                    self.db.collection(self.collection_name).document(violation_id).set(record)
                    processed += 1
                except Exception as e:
                    logger.error(f"Failed to process queued violation: {e}")
                    failed.append(record)
            
            # Re-queue failed items
            for record in failed:
                self.offline_queue.append(record)
            
            if processed > 0:
                logger.info(f"✓ Processed {processed} queued violations")
    
    def get_violation(self, violation_id):
        """Retrieve violation by ID"""
        try:
            if not self.is_connected:
                return None
            
            doc = self.db.collection(self.collection_name).document(violation_id).get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve violation: {e}")
            return None
    
    def query_violations(self, start_date=None, end_date=None, limit=100):
        """
        Query violations with filters
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            limit: Maximum results
            
        Returns:
            List of violation records
        """
        try:
            if not self.is_connected:
                return []
            
            query = self.db.collection(self.collection_name)
            
            if start_date:
                query = query.where('timestamp', '>=', start_date)
            
            if end_date:
                query = query.where('timestamp', '<=', end_date)
            
            query = query.limit(limit)
            
            results = []
            for doc in query.stream():
                results.append(doc.to_dict())
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query violations: {e}")
            return []
    
    def update_violation_status(self, violation_id, status):
        """Update violation status"""
        try:
            if not self.is_connected:
                return False
            
            self.db.collection(self.collection_name).document(violation_id).update({
                'status': status,
                'updated_at': firestore.SERVER_TIMESTAMP
            })
            
            logger.info(f"✓ Updated violation {violation_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update violation: {e}")
            return False
    
    def get_statistics(self):
        """Get violation statistics"""
        try:
            if not self.is_connected:
                return {}
            
            # Get total count
            docs = self.db.collection(self.collection_name).stream()
            total = sum(1 for _ in docs)
            
            # Get today's count
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_docs = self.db.collection(self.collection_name)\
                .where('timestamp', '>=', today)\
                .stream()
            today_count = sum(1 for _ in today_docs)
            
            return {
                'total_violations': total,
                'today_violations': today_count,
                'queue_size': len(self.offline_queue)
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


class LocalViolationStorage:
    """Fallback local storage when Firestore is unavailable"""
    
    def __init__(self, storage_dir='violations'):
        """
        Initialize local storage
        
        Args:
            storage_dir: Directory to store violations
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.storage_dir / 'violations.json'
        self.violations = self._load_metadata()
    
    def _load_metadata(self):
        """Load violations metadata from JSON"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_metadata(self):
        """Save violations metadata to JSON"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.violations, f, indent=2, default=str)
    
    def save_violation(self, violation_data, vehicle_image=None, plate_image=None):
        """Save violation locally"""
        import cv2
        
        violation_id = violation_data.get('violation_id', 
                                         f"VIO_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create violation directory
        violation_dir = self.storage_dir / violation_id
        violation_dir.mkdir(exist_ok=True)
        
        # Save images
        if vehicle_image is not None:
            vehicle_path = violation_dir / 'vehicle.jpg'
            if hasattr(vehicle_image, 'shape'):
                cv2.imwrite(str(vehicle_path), vehicle_image)
            violation_data['vehicle_image_path'] = str(vehicle_path)
        
        if plate_image is not None:
            plate_path = violation_dir / 'plate.jpg'
            if hasattr(plate_image, 'shape'):
                cv2.imwrite(str(plate_path), plate_image)
            violation_data['plate_image_path'] = str(plate_path)
        
        # Add to metadata
        self.violations.append(violation_data)
        self._save_metadata()
        
        logger.info(f"✓ Violation saved locally: {violation_id}")
        return violation_id
