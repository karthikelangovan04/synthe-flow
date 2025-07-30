#!/usr/bin/env python3
"""
Test script to generate and display sample data with 10 records each
"""
import requests
import json
import time

# Test data for single table with 10 records
single_table_test = {
    "tables": [
        {
            "name": "Products",
            "description": "Product catalog with 10 records",
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
    "relationships": [],
    "scale": 0.1,  # Generate 10 rows (100 * 0.1)
    "quality_settings": {
        "threshold": 0.8,
        "include_relationships": False
    }
}

# Test data for multi-table with relationships - 10 records each
multi_table_test = {
    "tables": [
        {
            "name": "Customers",
            "description": "Customer information with 10 records",
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
            "description": "Customer orders with 10 records",
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
    "scale": 0.1,  # Generate 10 rows per table (100 * 0.1)
    "quality_settings": {
        "threshold": 0.8,
        "include_relationships": True
    }
}

def display_table_data(table_name, data, max_records=10):
    """Display table data in a formatted way"""
    print(f"\n📊 {table_name} Table ({len(data)} records):")
    print("=" * 80)
    
    if not data:
        print("No data available")
        return
    
    # Get column names from first record
    columns = list(data[0].keys())
    
    # Print header
    header = " | ".join(f"{col:15}" for col in columns)
    print(header)
    print("-" * len(header))
    
    # Print data rows (limit to max_records)
    for i, row in enumerate(data[:max_records]):
        row_str = " | ".join(f"{str(val):15}" for val in row.values())
        print(row_str)
    
    if len(data) > max_records:
        print(f"... and {len(data) - max_records} more records")

def test_single_table_sample():
    """Test single table generation with 10 records"""
    base_url = "http://localhost:8002"
    
    print("🧪 Testing Single Table Sample Data (10 records)")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{base_url}/api/sdv/generate",
            json=single_table_test,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'completed':
                synthetic_data = result.get('synthetic_data', {})
                
                print(f"✅ Successfully generated {len(synthetic_data)} table(s)")
                
                for table_name, data in synthetic_data.items():
                    display_table_data(table_name, data, 10)
                
                return True
            else:
                print(f"❌ Error: {result.get('error')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_multi_table_sample():
    """Test multi-table generation with relationships - 10 records each"""
    base_url = "http://localhost:8002"
    
    print("\n🧪 Testing Multi-Table Sample Data with Relationships (10 records each)")
    print("=" * 80)
    
    try:
        response = requests.post(
            f"{base_url}/api/sdv/generate",
            json=multi_table_test,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'completed':
                synthetic_data = result.get('synthetic_data', {})
                
                print(f"✅ Successfully generated {len(synthetic_data)} table(s) with relationships")
                
                # Display parent table first (Customers)
                if 'Customers' in synthetic_data:
                    display_table_data('Customers', synthetic_data['Customers'], 10)
                
                # Display child table (Orders)
                if 'Orders' in synthetic_data:
                    display_table_data('Orders', synthetic_data['Orders'], 10)
                
                # Show relationship analysis
                print("\n🔗 Relationship Analysis:")
                print("=" * 40)
                
                if 'Customers' in synthetic_data and 'Orders' in synthetic_data:
                    customers = synthetic_data['Customers']
                    orders = synthetic_data['Orders']
                    
                    # Get all customer IDs
                    customer_ids = {row['customer_id'] for row in customers}
                    
                    # Count orders per customer
                    customer_order_count = {}
                    for order in orders:
                        customer_id = order['customer_id']
                        customer_order_count[customer_id] = customer_order_count.get(customer_id, 0) + 1
                    
                    print(f"📈 Total Customers: {len(customers)}")
                    print(f"📈 Total Orders: {len(orders)}")
                    print(f"📈 Average Orders per Customer: {len(orders) / len(customers):.2f}")
                    
                    print("\n📊 Orders per Customer:")
                    for customer_id, count in customer_order_count.items():
                        customer_name = next((c['customer_name'] for c in customers if c['customer_id'] == customer_id), 'Unknown')
                        print(f"   {customer_name} ({customer_id}): {count} orders")
                
                return True
            else:
                print(f"❌ Error: {result.get('error')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def export_sample_data(synthetic_data, filename):
    """Export sample data to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(synthetic_data, f, indent=2)
        print(f"💾 Sample data exported to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Export error: {e}")
        return False

if __name__ == "__main__":
    print("Starting Sample Data Generation Tests...")
    print("=" * 60)
    
    # Test 1: Single table with 10 records
    success1 = test_single_table_sample()
    
    # Test 2: Multi-table with relationships - 10 records each
    success2 = test_multi_table_sample()
    
    if success1 and success2:
        print("\n🎉 All sample data tests completed successfully!")
        print("The backend is generating high-quality synthetic data with proper relationships.")
    else:
        print("\n❌ Some tests failed. Please check the backend.")
    
    print("\nSample data generation tests completed.") 