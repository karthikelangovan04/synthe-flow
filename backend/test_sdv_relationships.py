#!/usr/bin/env python3
"""
Test script for SDV Backend API with proper relationships
"""
import requests
import json
import time

# Test data with proper relationships - one table has primary key, other references it
test_data_with_relationships = {
    "tables": [
        {
            "name": "Customers",
            "description": "Customer information with primary key",
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
                    "name": "phone",
                    "data_type": "varchar",
                    "is_nullable": True,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Customer phone number"
                }
            ]
        },
        {
            "name": "Orders",
            "description": "Customer orders referencing customers",
            "columns": [
                {
                    "name": "order_id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "is_primary_key": True,
                    "is_unique": True,
                    "enhanced_description": "Unique identifier for each order"
                },
                {
                    "name": "customer_id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Reference to customer who placed the order"
                },
                {
                    "name": "order_date",
                    "data_type": "date",
                    "is_nullable": False,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Date when order was placed"
                },
                {
                    "name": "total_amount",
                    "data_type": "decimal",
                    "is_nullable": True,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Total amount of the order"
                },
                {
                    "name": "status",
                    "data_type": "varchar",
                    "is_nullable": False,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Order status (pending, completed, cancelled)"
                }
            ]
        }
    ],
    "relationships": [
        {
            "source_table": "Customers",
            "source_column": "customer_id",
            "target_table": "Orders", 
            "target_column": "customer_id",
            "relationship_type": "one-to-many"
        }
    ],
    "scale": 3.0,  # Generate 3x more data (300 rows per table)
    "quality_settings": {
        "threshold": 0.8,
        "include_relationships": True
    }
}

# Test data for single table mode (no relationships)
test_data_single_table = {
    "tables": [
        {
            "name": "Products",
            "description": "Product catalog",
            "columns": [
                {
                    "name": "product_id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "is_primary_key": True,
                    "is_unique": True,
                    "enhanced_description": "Unique identifier for each product"
                },
                {
                    "name": "product_name",
                    "data_type": "varchar",
                    "is_nullable": False,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Name of the product"
                },
                {
                    "name": "category",
                    "data_type": "varchar",
                    "is_nullable": True,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Product category"
                },
                {
                    "name": "price",
                    "data_type": "decimal",
                    "is_nullable": False,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Product price"
                },
                {
                    "name": "in_stock",
                    "data_type": "boolean",
                    "is_nullable": False,
                    "is_primary_key": False,
                    "is_unique": False,
                    "enhanced_description": "Whether product is in stock"
                }
            ]
        }
    ],
    "relationships": [],  # No relationships for single table test
    "scale": 2.0,  # Generate 2x more data (200 rows)
    "quality_settings": {
        "threshold": 0.8,
        "include_relationships": False
    }
}

def test_backend_with_relationships():
    """Test the SDV backend API with proper relationships"""
    base_url = "http://localhost:8002"
    
    print("🧪 Testing SDV Backend API with Relationships")
    print("=" * 60)
    
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
    
    # Test 2: Single table mode (no relationships)
    print("\n2. Testing single table mode...")
    try:
        print(f"   Sending single table request with scale: {test_data_single_table['scale']}")
        print(f"   Expected rows: {int(100 * test_data_single_table['scale'])}")
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/sdv/generate",
            json=test_data_single_table,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single table generation successful!")
            print(f"   Session ID: {result.get('session_id')}")
            print(f"   Status: {result.get('status')}")
            
            if result.get('status') == 'error':
                print(f"❌ Backend returned error: {result.get('error')}")
                return False
            
            # Check the generated data
            synthetic_data = result.get('synthetic_data', {})
            total_rows = 0
            
            print("\n   Generated data summary (Single Table):")
            for table_name, rows in synthetic_data.items():
                row_count = len(rows) if isinstance(rows, list) else 0
                total_rows += row_count
                print(f"   - {table_name}: {row_count} rows")
                
                # Show first few rows as sample
                if rows and len(rows) > 0:
                    print(f"     Sample data:")
                    for i, row in enumerate(rows[:2]):  # Show first 2 rows
                        print(f"       Row {i+1}: {row}")
                    if len(rows) > 2:
                        print(f"       ... and {len(rows) - 2} more rows")
            
            print(f"\n   Total synthetic rows generated: {total_rows}")
            
        else:
            print(f"❌ Single table generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Single table generation error: {e}")
        return False
    
    # Test 3: Multi-table with relationships
    print("\n3. Testing multi-table with relationships...")
    try:
        print(f"   Sending multi-table request with scale: {test_data_with_relationships['scale']}")
        print(f"   Expected rows per table: {int(100 * test_data_with_relationships['scale'])}")
        print(f"   Relationships: {len(test_data_with_relationships['relationships'])}")
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/sdv/generate",
            json=test_data_with_relationships,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Multi-table generation successful!")
            print(f"   Session ID: {result.get('session_id')}")
            print(f"   Status: {result.get('status')}")
            
            if result.get('status') == 'error':
                print(f"❌ Backend returned error: {result.get('error')}")
                return False
            
            # Check the generated data
            synthetic_data = result.get('synthetic_data', {})
            total_rows = 0
            
            print("\n   Generated data summary (Multi-Table):")
            for table_name, rows in synthetic_data.items():
                row_count = len(rows) if isinstance(rows, list) else 0
                total_rows += row_count
                print(f"   - {table_name}: {row_count} rows")
                
                # Show first few rows as sample
                if rows and len(rows) > 0:
                    print(f"     Sample data:")
                    for i, row in enumerate(rows[:2]):  # Show first 2 rows
                        print(f"       Row {i+1}: {row}")
                    if len(rows) > 2:
                        print(f"       ... and {len(rows) - 2} more rows")
            
            print(f"\n   Total synthetic rows generated: {total_rows}")
            
            # Test 4: Export as JSON
            print("\n4. Testing data export...")
            export_filename = f"synthetic_data_relationships_{int(time.time())}.json"
            with open(export_filename, 'w') as f:
                json.dump(synthetic_data, f, indent=2)
            print(f"✅ Data exported to: {export_filename}")
            
            return True
            
        else:
            print(f"❌ Multi-table generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Multi-table generation error: {e}")
        return False

def test_relationship_validation():
    """Test relationship validation with invalid relationships"""
    base_url = "http://localhost:8002"
    
    print("\n5. Testing invalid relationship validation...")
    
    # Test with invalid relationship (both columns are primary keys)
    invalid_test_data = {
        "tables": [
            {
                "name": "Table1",
                "description": "First table",
                "columns": [
                    {
                        "name": "id1",
                        "data_type": "integer",
                        "is_nullable": False,
                        "is_primary_key": True,
                        "is_unique": True,
                        "enhanced_description": "Primary key"
                    }
                ]
            },
            {
                "name": "Table2", 
                "description": "Second table",
                "columns": [
                    {
                        "name": "id2",
                        "data_type": "integer",
                        "is_nullable": False,
                        "is_primary_key": True,
                        "is_unique": True,
                        "enhanced_description": "Primary key"
                    }
                ]
            }
        ],
        "relationships": [
            {
                "source_table": "Table1",
                "source_column": "id1",
                "target_table": "Table2",
                "target_column": "id2",
                "relationship_type": "one-to-one"
            }
        ],
        "scale": 1.0
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/sdv/generate",
            json=invalid_test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'error':
                print("✅ Invalid relationship properly rejected")
                print(f"   Error: {result.get('error')}")
                return True
            else:
                print("❌ Invalid relationship should have been rejected")
                return False
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Invalid relationship test error: {e}")
        return False

if __name__ == "__main__":
    print("Starting SDV Backend Relationship Tests...")
    
    # Test 1-4: Valid scenarios
    success1 = test_backend_with_relationships()
    
    # Test 5: Invalid relationship validation
    success2 = test_relationship_validation()
    
    if success1 and success2:
        print("\n🎉 All relationship tests passed! The SDV backend handles relationships correctly.")
    else:
        print("\n❌ Some relationship tests failed. Please check the SDV backend.")
    
    print("\nRelationship tests completed.") 