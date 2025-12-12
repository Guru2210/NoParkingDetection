"""
Test Firestore Connection
Verify Firestore connectivity and operations
"""

import sys
import argparse
from firestore_manager import FirestoreManager
from datetime import datetime
import numpy as np

def test_connection(credentials_path):
    """Test Firestore connection"""
    print("Testing Firestore connection...")
    
    try:
        # Initialize Firestore
        firestore = FirestoreManager(
            credentials_path=credentials_path,
            collection_name='test_violations'
        )
        
        if not firestore.is_connected:
            print("✗ Failed to connect to Firestore")
            return False
        
        print("✓ Successfully connected to Firestore")
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def test_write(credentials_path):
    """Test writing to Firestore"""
    print("\nTesting write operation...")
    
    try:
        firestore = FirestoreManager(
            credentials_path=credentials_path,
            collection_name='test_violations'
        )
        
        # Create test violation
        test_data = {
            'vehicle_id': 999,
            'license_plate': 'TEST123',
            'confidence': 0.95,
            'parking_duration': 15.5,
            'location': 'Test Zone',
            'zone': 'Test'
        }
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # Save violation
        violation_id = firestore.save_violation(
            test_data,
            vehicle_image=test_image,
            plate_image=test_image
        )
        
        if violation_id:
            print(f"✓ Successfully wrote test violation: {violation_id}")
            return violation_id
        else:
            print("✗ Failed to write violation")
            return None
            
    except Exception as e:
        print(f"✗ Write failed: {e}")
        return None

def test_read(credentials_path, violation_id):
    """Test reading from Firestore"""
    print("\nTesting read operation...")
    
    try:
        firestore = FirestoreManager(
            credentials_path=credentials_path,
            collection_name='test_violations'
        )
        
        # Read violation
        data = firestore.get_violation(violation_id)
        
        if data:
            print("✓ Successfully read violation:")
            print(f"  License Plate: {data.get('license_plate')}")
            print(f"  Duration: {data.get('parking_duration')}s")
            return True
        else:
            print("✗ Failed to read violation")
            return False
            
    except Exception as e:
        print(f"✗ Read failed: {e}")
        return False

def test_query(credentials_path):
    """Test querying Firestore"""
    print("\nTesting query operation...")
    
    try:
        firestore = FirestoreManager(
            credentials_path=credentials_path,
            collection_name='test_violations'
        )
        
        # Query violations
        results = firestore.query_violations(limit=5)
        
        print(f"✓ Query returned {len(results)} results")
        
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. {result.get('license_plate', 'N/A')} - {result.get('parking_duration', 0)}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False

def test_statistics(credentials_path):
    """Test statistics retrieval"""
    print("\nTesting statistics...")
    
    try:
        firestore = FirestoreManager(
            credentials_path=credentials_path,
            collection_name='test_violations'
        )
        
        stats = firestore.get_statistics()
        
        print("✓ Statistics:")
        print(f"  Total violations: {stats.get('total_violations', 0)}")
        print(f"  Today's violations: {stats.get('today_violations', 0)}")
        print(f"  Queue size: {stats.get('queue_size', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Statistics failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Firestore Connection')
    parser.add_argument(
        '--credentials',
        type=str,
        default='firebase-credentials.json',
        help='Path to Firebase credentials JSON'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Firestore Connection Test")
    print("="*60)
    
    # Run tests
    tests_passed = 0
    tests_total = 5
    
    # Test 1: Connection
    if test_connection(args.credentials):
        tests_passed += 1
    
    # Test 2: Write
    violation_id = test_write(args.credentials)
    if violation_id:
        tests_passed += 1
    
    # Test 3: Read
    if violation_id and test_read(args.credentials, violation_id):
        tests_passed += 1
    
    # Test 4: Query
    if test_query(args.credentials):
        tests_passed += 1
    
    # Test 5: Statistics
    if test_statistics(args.credentials):
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print("="*60)
    
    return 0 if tests_passed == tests_total else 1

if __name__ == '__main__':
    sys.exit(main())
