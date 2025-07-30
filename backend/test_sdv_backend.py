#!/usr/bin/env python3
"""
Test script for SDV Backend API
"""
import requests
import json
import time

# Test data - similar to what the frontend sends
test_data = {
    "tables": [
        {
            "name": "Accounts",
            "description": "Customer account information",
            "columns": [
                {
                    "name": "account_id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "is_primary_key": True,
                    "is_unique": True,
                    "enhanced_description": "Unique identifier for each account"
                },
                {
                    "name": "customer_name",
                    "data_type": "varchar",
                    "is_nullable": False,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Full name of the customer"
                },
                {
                    "name": "email",
                    "data_type": "varchar",
                    "is_nullable": False,
                    "is_primary_key": False,
                    "is_unique": True,
                    "enhanced_description": "Customer email address"
                },
                {
                    "name": "balance",
                    "data_type": "decimal",
                    "is_nullable": True,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Current account balance"
                }
            ]
        },
        {
            "name": "Customer_data",
            "description": "Additional customer information",
            "columns": [
                {
                    "name": "customer_id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "is_primary_key": True,
                    "is_unique": True,
                    "enhanced_description": "Unique identifier for each customer"
                },
                {
                    "name": "phone",
                    "data_type": "varchar",
                    "is_nullable": True,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Customer phone number"
                },
                {
                    "name": "address",
                    "data_type": "text",
                    "is_nullable": True,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Customer address"
                }
            ]
        }
    ],
    "relationships": [],  # No relationships for now - test single table mode
    "scale": 5.0,  # Generate 5x more data (500 rows per table)
    "quality_settings": {
        "threshold": 0.8,
        "include_relationships": True
    }
}

def test_backend():
    """Test the SDV backend API"""
    base_url = "http://localhost:8002"
    
    print("🧪 Testing SDV Backend API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/api/sdv/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Generate synthetic data
    print("\n2. Testing synthetic data generation...")
    try:
        print(f"   Sending request with scale: {test_data['scale']}")
        print(f"   Expected rows per table: {int(100 * test_data['scale'])}")
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/sdv/generate",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Data generation successful!")
            print(f"   Session ID: {result.get('session_id')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Full response: {json.dumps(result, indent=2)}")
            
            if result.get('status') == 'error':
                print(f"❌ Backend returned error: {result.get('error')}")
                return False
            
            # Check the generated data
            synthetic_data = result.get('synthetic_data', {})
            total_rows = 0
            
            print("\n   Generated data summary:")
            for table_name, rows in synthetic_data.items():
                row_count = len(rows) if isinstance(rows, list) else 0
                total_rows += row_count
                print(f"   - {table_name}: {row_count} rows")
                
                # Show first few rows as sample
                if rows and len(rows) > 0:
                    print(f"     Sample data:")
                    for i, row in enumerate(rows[:3]):  # Show first 3 rows
                        print(f"       Row {i+1}: {row}")
                    if len(rows) > 3:
                        print(f"       ... and {len(rows) - 3} more rows")
            
            print(f"\n   Total synthetic rows generated: {total_rows}")
            
            # Test 3: Export as JSON
            print("\n3. Testing data export...")
            export_filename = f"synthetic_data_test_{int(time.time())}.json"
            with open(export_filename, 'w') as f:
                json.dump(synthetic_data, f, indent=2)
            print(f"✅ Data exported to: {export_filename}")
            
            return True
            
        else:
            print(f"❌ Data generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False

if __name__ == "__main__":
    print("Starting SDV Backend Test...")
    success = test_backend()
    
    if success:
        print("\n🎉 All tests passed! The SDV backend is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the SDV backend.")
    
    print("\nTest completed.") 