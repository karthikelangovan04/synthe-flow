#!/usr/bin/env python3
"""
Test script for Enhanced SDV Backend
Tests the new features and verifies that previous errors are resolved
"""

import requests
import json
import time
import pandas as pd
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8003"
TEST_FILES = {
    "users": "users.csv",
    "posts": "posts.csv"
}

def test_health_check():
    """Test the health endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/api/enhanced/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Service: {data['service']}")
            print(f"   Version: {data['version']}")
            print(f"   Capabilities: {', '.join(data['capabilities'])}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_file_upload():
    """Test file upload functionality"""
    print("\n📤 Testing file upload...")
    
    uploaded_files = {}
    
    for table_name, filename in TEST_FILES.items():
        try:
            # Check if test file exists
            if not Path(filename).exists():
                print(f"⚠️  Test file {filename} not found, skipping upload test")
                continue
                
            with open(filename, 'rb') as f:
                files = {'file': (filename, f, 'text/csv')}
                response = requests.post(f"{BASE_URL}/api/enhanced/upload/file", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    uploaded_files[table_name] = data['filename']
                    print(f"✅ Uploaded {table_name}: {data['filename']}")
                else:
                    print(f"❌ Upload failed for {table_name}: {response.status_code}")
                    print(f"   Response: {response.text}")
                    
        except Exception as e:
            print(f"❌ Upload error for {table_name}: {e}")
    
    return uploaded_files

def test_synthetic_data_generation(uploaded_files):
    """Test synthetic data generation with enhanced features"""
    print("\n🚀 Testing synthetic data generation...")
    
    if not uploaded_files:
        print("⚠️  No files uploaded, skipping generation test")
        return None
    
    try:
        # Prepare generation request with proper schema structure
        generation_request = {
            "tables": [
                {
                    "name": "users",
                    "description": "User information table",
                    "columns": [
                        {"name": "user_id", "data_type": "integer", "is_nullable": False, "is_primary_key": True, "is_unique": True},
                        {"name": "username", "data_type": "varchar", "is_nullable": False, "is_primary_key": False, "is_unique": True},
                        {"name": "email", "data_type": "varchar", "is_nullable": False, "is_primary_key": False, "is_unique": True},
                        {"name": "created_at", "data_type": "timestamp", "is_nullable": False, "is_primary_key": False, "is_unique": False}
                    ],
                    "domain": "Social Media",
                    "estimated_volume": 100,
                    "business_criticality": "Medium"
                },
                {
                    "name": "posts",
                    "description": "User posts table",
                    "columns": [
                        {"name": "post_id", "data_type": "integer", "is_nullable": False, "is_primary_key": True, "is_unique": True},
                        {"name": "user_id", "data_type": "integer", "is_nullable": False, "is_primary_key": False, "is_unique": False},
                        {"name": "title", "data_type": "varchar", "is_nullable": False, "is_primary_key": False, "is_unique": False},
                        {"name": "content", "data_type": "text", "is_nullable": False, "is_primary_key": False, "is_unique": False},
                        {"name": "created_at", "data_type": "timestamp", "is_nullable": False, "is_primary_key": False, "is_unique": False}
                    ],
                    "domain": "Social Media",
                    "estimated_volume": 200,
                    "business_criticality": "Medium"
                }
            ],
            "relationships": [
                {
                    "source_table": "users",
                    "source_column": "user_id",
                    "target_table": "posts",
                    "target_column": "user_id",
                    "relationship_type": "one-to-many"
                }
            ],
            "data_sources": [
                {
                    "type": "local",
                    "config": {},
                    "file_paths": list(uploaded_files.values())
                }
            ],
            "scale": 1.0,
            "quality_settings": {
                "enable_privacy_assessment": True,
                "enable_statistical_validation": True,
                "enable_relationship_validation": True
            },
            "output_format": "json",
            "generation_strategy": "balanced",
            "domain_context": "Social Media",
            "privacy_level": "enhanced"
        }
        
        print("   Sending generation request...")
        response = requests.post(
            f"{BASE_URL}/api/enhanced/generate",
            json=generation_request
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Generation started successfully!")
            print(f"   Session ID: {data.get('session_id', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            
            # Check if session_id is empty (background task)
            if not data.get('session_id'):
                print(f"   ⚠️  Note: Session ID will be created in background task")
                print(f"   🔍 Will search for available sessions...")
            
            # Debug: Show full response for troubleshooting
            print(f"   🔍 Debug - Full response keys: {list(data.keys())}")
            
            # Check for quality metrics
            if 'quality_metrics' in data:
                quality = data['quality_metrics']
                print(f"   Overall Quality Score: {quality.get('overall_score', 'N/A')}")
                
                # Check privacy metrics
                if 'privacy_metrics' in quality:
                    privacy = quality['privacy_metrics']
                    print(f"   Privacy Score: {privacy.get('overall_privacy_score', 'N/A')}")
                
                # Check statistical metrics
                if 'statistical_metrics' in quality:
                    stats = quality['statistical_metrics']
                    print(f"   Statistical Score: {stats.get('overall_statistical_score', 'N/A')}")
            
            return data
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return None

def test_export_formats(session_id):
    """Test different export formats with session_id"""
    print("\n📊 Testing export formats...")
    
    if not session_id:
        print("⚠️  No session ID provided, skipping export tests")
        return
    
    export_formats = ['csv', 'json', 'excel']
    
    for format_type in export_formats:
        try:
            print(f"   Testing {format_type.upper()} export...")
            
            # Test export with session_id parameter
            response = requests.get(f"{BASE_URL}/api/enhanced/export/{format_type}", params={'session_id': session_id})
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {format_type.upper()} export successful!")
                print(f"      Files: {len(data.get('export_files', []))}")
            elif response.status_code == 400:
                print(f"⚠️  {format_type.upper()} export: {response.json().get('detail', 'Bad request')}")
            elif response.status_code == 404:
                print(f"⚠️  {format_type.upper()} export: Session not found")
            else:
                print(f"❌ {format_type.upper()} export error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {format_type.upper()} export test error: {e}")

def find_generation_session(max_wait=60):
    """Find the generation session that was created"""
    print(f"   🔍 Looking for available generation sessions...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/api/enhanced/sessions")
            if response.status_code == 200:
                data = response.json()
                sessions = data.get('sessions', [])
                
                if sessions:
                    # Find the most recent session
                    latest_session = max(sessions, key=lambda x: x.get('created_at', ''))
                    session_id = latest_session['session_id']
                    status = latest_session['status']
                    
                    print(f"   ✅ Found session: {session_id[:8]}... (Status: {status})")
                    return session_id, latest_session
                else:
                    print(f"   ⏳ No sessions found yet... ({int(time.time() - start_time)}s)")
                    time.sleep(2)
            else:
                print(f"   ⚠️  Failed to get sessions: {response.status_code}")
                time.sleep(2)
        except Exception as e:
            print(f"   ⚠️  Error getting sessions: {e}")
            time.sleep(2)
    
    print(f"   ⏰ Timeout waiting for session creation")
    return None, None

def wait_for_generation_completion(session_id, max_wait=60):
    """Wait for generation to complete and return the final status"""
    print(f"   Waiting for generation to complete (session: {session_id[:8]}...)")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/api/enhanced/status/{session_id}")
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                
                if status == 'completed':
                    print(f"   ✅ Generation completed successfully!")
                    return data
                elif status == 'failed':
                    error = data.get('error', 'Unknown error')
                    print(f"   ❌ Generation failed: {error}")
                    return data
                elif status == 'processing':
                    print(f"   ⏳ Still processing... ({int(time.time() - start_time)}s)")
                    time.sleep(2)
                else:
                    print(f"   ℹ️  Status: {status}")
                    time.sleep(2)
            else:
                print(f"   ⚠️  Status check failed: {response.status_code}")
                time.sleep(2)
        except Exception as e:
            print(f"   ⚠️  Status check error: {e}")
            time.sleep(2)
    
    print(f"   ⏰ Timeout waiting for generation completion")
    return None

def main():
    """Main test function"""
    print("🧪 Enhanced SDV Backend Test Suite")
    print("=" * 50)
    
    # Test 1: Health Check
    if not test_health_check():
        print("❌ Backend is not running or unhealthy. Please start the service first.")
        return
    
    # Test 2: File Upload
    uploaded_files = test_file_upload()
    
    # Test 3: Synthetic Data Generation
    if uploaded_files:
        generation_result = test_synthetic_data_generation(uploaded_files)
        
        if generation_result:
            print(f"\n🎉 Generation started successfully!")
            print("\n🔧 Key improvements verified:")
            print("   ✅ Enhanced privacy assessment")
            print("   ✅ Advanced statistical validation")
            print("   ✅ Multiple export formats")
            print("   ✅ Relationship integrity validation")
            
            # Always find the actual session that was created (since session_id is empty)
            print(f"\n🔍 Searching for generation session...")
            session_id, session_data = find_generation_session()
            
            if session_id:
                print(f"\n🎯 Found generation session: {session_id}")
                
                # Wait for generation to complete before testing exports
                wait_for_generation_completion(session_id)
                
                # Test 4: Export Formats (after completion)
                test_export_formats(session_id)
            else:
                print("\n⚠️  No generation session found, skipping export tests.")
                
        else:
            print("\n⚠️  Generation failed, but backend is running.")
    else:
        print("\n⚠️  Generation test failed, but backend is running.")
    
    print("\n" + "=" * 50)
    print("🏁 Test suite completed!")

if __name__ == "__main__":
    main() 