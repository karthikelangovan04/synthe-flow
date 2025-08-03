#!/usr/bin/env python3
"""
Complex HR Dataset Generator for Enhanced Backend Testing
Creates a realistic HR system with multiple tables and complex relationships
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
from typing import Dict, List, Any

def generate_complex_hr_dataset():
    """Generate complex HR dataset with multiple tables and relationships"""
    
    print("Generating complex HR dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate base data
    employees_df = generate_employees()
    departments_df = generate_departments()
    positions_df = generate_positions()
    salaries_df = generate_salaries(employees_df, positions_df)
    performance_df = generate_performance(employees_df)
    projects_df = generate_projects()
    project_assignments_df = generate_project_assignments(employees_df, projects_df)
    skills_df = generate_skills()
    employee_skills_df = generate_employee_skills(employees_df, skills_df)
    training_df = generate_training()
    training_enrollments_df = generate_training_enrollments(employees_df, training_df)
    
    # Save datasets
    datasets = {
        'employees': employees_df,
        'departments': departments_df,
        'positions': positions_df,
        'salaries': salaries_df,
        'performance': performance_df,
        'projects': projects_df,
        'project_assignments': project_assignments_df,
        'skills': skills_df,
        'employee_skills': employee_skills_df,
        'training': training_df,
        'training_enrollments': training_enrollments_df
    }
    
    # Save to CSV files
    for name, df in datasets.items():
        filename = f"complex_hr_{name}.csv"
        df.to_csv(filename, index=False)
        print(f"Generated {filename}: {len(df)} rows, {len(df.columns)} columns")
    
    # Create schema definition
    schema = create_schema_definition(datasets)
    
    # Save schema
    with open('complex_hr_schema.json', 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"\n✅ Generated complex HR dataset with {len(datasets)} tables")
    print(f"📊 Total records: {sum(len(df) for df in datasets.values())}")
    print(f"🔗 Total relationships: {len(schema['relationships'])}")
    
    return datasets, schema

def generate_employees():
    """Generate employee data"""
    n_employees = 500
    
    # Department IDs (will be created in departments table)
    department_ids = list(range(1, 21))  # 20 departments
    
    # Position IDs (will be created in positions table)
    position_ids = list(range(1, 51))  # 50 positions
    
    data = []
    for i in range(n_employees):
        employee_id = i + 1
        hire_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1000))
        
        data.append({
            'employee_id': employee_id,
            'first_name': f"Employee{i+1}",
            'last_name': f"LastName{i+1}",
            'email': f"employee{i+1}@company.com",
            'phone': f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            'hire_date': hire_date.strftime('%Y-%m-%d'),
            'department_id': random.choice(department_ids),
            'position_id': random.choice(position_ids),
            'manager_id': random.choice([None] + list(range(1, employee_id))),
            'salary_grade': random.choice(['A', 'B', 'C', 'D', 'E']),
            'is_active': random.choice([True, True, True, False]),  # 75% active
            'location': random.choice(['HQ', 'Branch1', 'Branch2', 'Remote']),
            'employment_type': random.choice(['Full-time', 'Part-time', 'Contract']),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return pd.DataFrame(data)

def generate_departments():
    """Generate department data"""
    departments = [
        {'department_id': 1, 'name': 'Engineering', 'budget': 5000000, 'location': 'HQ'},
        {'department_id': 2, 'name': 'Sales', 'budget': 3000000, 'location': 'HQ'},
        {'department_id': 3, 'name': 'Marketing', 'budget': 2000000, 'location': 'HQ'},
        {'department_id': 4, 'name': 'Finance', 'budget': 1500000, 'location': 'HQ'},
        {'department_id': 5, 'name': 'HR', 'budget': 1000000, 'location': 'HQ'},
        {'department_id': 6, 'name': 'Operations', 'budget': 2500000, 'location': 'Branch1'},
        {'department_id': 7, 'name': 'Customer Support', 'budget': 1800000, 'location': 'Branch1'},
        {'department_id': 8, 'name': 'Product Management', 'budget': 2200000, 'location': 'HQ'},
        {'department_id': 9, 'name': 'Quality Assurance', 'budget': 1200000, 'location': 'Branch2'},
        {'department_id': 10, 'name': 'Research & Development', 'budget': 4000000, 'location': 'HQ'},
        {'department_id': 11, 'name': 'Legal', 'budget': 800000, 'location': 'HQ'},
        {'department_id': 12, 'name': 'IT Support', 'budget': 900000, 'location': 'Branch1'},
        {'department_id': 13, 'name': 'Business Development', 'budget': 1600000, 'location': 'HQ'},
        {'department_id': 14, 'name': 'Data Science', 'budget': 2800000, 'location': 'HQ'},
        {'department_id': 15, 'name': 'Design', 'budget': 1400000, 'location': 'HQ'},
        {'department_id': 16, 'name': 'Security', 'budget': 1100000, 'location': 'HQ'},
        {'department_id': 17, 'name': 'Facilities', 'budget': 600000, 'location': 'Branch1'},
        {'department_id': 18, 'name': 'Procurement', 'budget': 700000, 'location': 'Branch2'},
        {'department_id': 19, 'name': 'Compliance', 'budget': 500000, 'location': 'HQ'},
        {'department_id': 20, 'name': 'Training', 'budget': 400000, 'location': 'Branch1'}
    ]
    
    return pd.DataFrame(departments)

def generate_positions():
    """Generate position data"""
    positions = []
    position_id = 1
    
    # Engineering positions
    for level in ['Junior', 'Mid', 'Senior', 'Lead', 'Principal']:
        for role in ['Software Engineer', 'Data Engineer', 'DevOps Engineer', 'QA Engineer']:
            positions.append({
                'position_id': position_id,
                'title': f"{level} {role}",
                'department_id': 1,
                'level': level,
                'category': 'Engineering',
                'min_salary': 60000 + (position_id * 5000),
                'max_salary': 80000 + (position_id * 8000)
            })
            position_id += 1
    
    # Sales positions
    for level in ['Junior', 'Senior', 'Lead']:
        for role in ['Sales Representative', 'Account Manager', 'Sales Manager']:
            positions.append({
                'position_id': position_id,
                'title': f"{level} {role}",
                'department_id': 2,
                'level': level,
                'category': 'Sales',
                'min_salary': 50000 + (position_id * 3000),
                'max_salary': 70000 + (position_id * 5000)
            })
            position_id += 1
    
    # Add more positions for other departments
    other_positions = [
        {'title': 'Marketing Manager', 'department_id': 3, 'level': 'Senior', 'category': 'Marketing'},
        {'title': 'Financial Analyst', 'department_id': 4, 'level': 'Mid', 'category': 'Finance'},
        {'title': 'HR Specialist', 'department_id': 5, 'level': 'Mid', 'category': 'HR'},
        {'title': 'Operations Manager', 'department_id': 6, 'level': 'Senior', 'category': 'Operations'},
        {'title': 'Customer Success Manager', 'department_id': 7, 'level': 'Mid', 'category': 'Support'},
        {'title': 'Product Manager', 'department_id': 8, 'level': 'Senior', 'category': 'Product'},
        {'title': 'QA Lead', 'department_id': 9, 'level': 'Lead', 'category': 'Quality'},
        {'title': 'Research Scientist', 'department_id': 10, 'level': 'Senior', 'category': 'Research'},
        {'title': 'Legal Counsel', 'department_id': 11, 'level': 'Senior', 'category': 'Legal'},
        {'title': 'IT Support Specialist', 'department_id': 12, 'level': 'Mid', 'category': 'IT'},
        {'title': 'Business Development Manager', 'department_id': 13, 'level': 'Senior', 'category': 'Business'},
        {'title': 'Data Scientist', 'department_id': 14, 'level': 'Senior', 'category': 'Data'},
        {'title': 'UX Designer', 'department_id': 15, 'level': 'Mid', 'category': 'Design'},
        {'title': 'Security Engineer', 'department_id': 16, 'level': 'Senior', 'category': 'Security'},
        {'title': 'Facilities Coordinator', 'department_id': 17, 'level': 'Junior', 'category': 'Facilities'},
        {'title': 'Procurement Specialist', 'department_id': 18, 'level': 'Mid', 'category': 'Procurement'},
        {'title': 'Compliance Officer', 'department_id': 19, 'level': 'Senior', 'category': 'Compliance'},
        {'title': 'Training Coordinator', 'department_id': 20, 'level': 'Mid', 'category': 'Training'}
    ]
    
    for pos in other_positions:
        positions.append({
            'position_id': position_id,
            'title': pos['title'],
            'department_id': pos['department_id'],
            'level': pos['level'],
            'category': pos['category'],
            'min_salary': 45000 + (position_id * 2000),
            'max_salary': 65000 + (position_id * 4000)
        })
        position_id += 1
    
    return pd.DataFrame(positions)

def generate_salaries(employees_df, positions_df):
    """Generate salary data"""
    salaries = []
    
    for _, employee in employees_df.iterrows():
        # Find matching position
        matching_positions = positions_df[positions_df['position_id'] == employee['position_id']]
        if len(matching_positions) == 0:
            # Use default salary if position not found
            base_salary = random.randint(50000, 100000)
        else:
            position = matching_positions.iloc[0]
            base_salary = random.randint(position['min_salary'], position['max_salary'])
        
        # Add bonuses and adjustments
        bonus = random.randint(0, int(base_salary * 0.2))  # Up to 20% bonus
        total_salary = base_salary + bonus
        
        salaries.append({
            'salary_id': len(salaries) + 1,
            'employee_id': employee['employee_id'],
            'base_salary': base_salary,
            'bonus': bonus,
            'total_salary': total_salary,
            'effective_date': employee['hire_date'],
            'currency': 'USD',
            'review_date': (datetime.strptime(employee['hire_date'], '%Y-%m-%d') + timedelta(days=365)).strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(salaries)

def generate_performance(employees_df):
    """Generate performance data"""
    performance = []
    
    for _, employee in employees_df.iterrows():
        # Generate multiple performance reviews
        hire_date = datetime.strptime(employee['hire_date'], '%Y-%m-%d')
        
        for review_num in range(1, 4):  # 3 reviews per employee
            review_date = hire_date + timedelta(days=review_num * 180)  # Every 6 months
            
            if review_date <= datetime.now():
                performance.append({
                    'review_id': len(performance) + 1,
                    'employee_id': employee['employee_id'],
                    'review_date': review_date.strftime('%Y-%m-%d'),
                    'rating': random.choice([1, 2, 3, 4, 5]),
                    'goals_met': random.choice(['Exceeded', 'Met', 'Partially Met', 'Not Met']),
                    'comments': f"Performance review #{review_num} for {employee['first_name']} {employee['last_name']}",
                    'reviewer_id': random.choice(employees_df['employee_id'].tolist())
                })
    
    return pd.DataFrame(performance)

def generate_projects():
    """Generate project data"""
    projects = []
    
    project_types = ['Development', 'Research', 'Infrastructure', 'Marketing', 'Process Improvement']
    statuses = ['Planning', 'Active', 'On Hold', 'Completed', 'Cancelled']
    
    for i in range(50):
        start_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
        end_date = start_date + timedelta(days=random.randint(30, 365))
        
        projects.append({
            'project_id': i + 1,
            'name': f"Project {chr(65 + (i % 26))}{i+1}",
            'description': f"Description for project {i+1}",
            'type': random.choice(project_types),
            'status': random.choice(statuses),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'budget': random.randint(50000, 500000),
            'department_id': random.randint(1, 20),
            'manager_id': random.randint(1, 500)
        })
    
    return pd.DataFrame(projects)

def generate_project_assignments(employees_df, projects_df):
    """Generate project assignments"""
    assignments = []
    
    # Assign employees to projects
    for _, project in projects_df.iterrows():
        # Assign 3-8 employees per project
        num_assignments = random.randint(3, 8)
        assigned_employees = random.sample(employees_df['employee_id'].tolist(), num_assignments)
        
        for employee_id in assigned_employees:
            start_date = datetime.strptime(project['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(project['end_date'], '%Y-%m-%d')
            
            # Random assignment period within project timeline
            assignment_start = start_date + timedelta(days=random.randint(0, 30))
            assignment_end = end_date - timedelta(days=random.randint(0, 30))
            
            assignments.append({
                'assignment_id': len(assignments) + 1,
                'project_id': project['project_id'],
                'employee_id': employee_id,
                'role': random.choice(['Developer', 'Tester', 'Analyst', 'Lead', 'Manager']),
                'allocation_percentage': random.choice([25, 50, 75, 100]),
                'start_date': assignment_start.strftime('%Y-%m-%d'),
                'end_date': assignment_end.strftime('%Y-%m-%d'),
                'is_active': project['status'] == 'Active'
            })
    
    return pd.DataFrame(assignments)

def generate_skills():
    """Generate skills data"""
    skills = [
        {'skill_id': 1, 'name': 'Python', 'category': 'Programming', 'level': 'Technical'},
        {'skill_id': 2, 'name': 'JavaScript', 'category': 'Programming', 'level': 'Technical'},
        {'skill_id': 3, 'name': 'Java', 'category': 'Programming', 'level': 'Technical'},
        {'skill_id': 4, 'name': 'SQL', 'category': 'Database', 'level': 'Technical'},
        {'skill_id': 5, 'name': 'AWS', 'category': 'Cloud', 'level': 'Technical'},
        {'skill_id': 6, 'name': 'Docker', 'category': 'DevOps', 'level': 'Technical'},
        {'skill_id': 7, 'name': 'Kubernetes', 'category': 'DevOps', 'level': 'Technical'},
        {'skill_id': 8, 'name': 'React', 'category': 'Frontend', 'level': 'Technical'},
        {'skill_id': 9, 'name': 'Node.js', 'category': 'Backend', 'level': 'Technical'},
        {'skill_id': 10, 'name': 'Machine Learning', 'category': 'AI/ML', 'level': 'Technical'},
        {'skill_id': 11, 'name': 'Project Management', 'category': 'Management', 'level': 'Soft'},
        {'skill_id': 12, 'name': 'Leadership', 'category': 'Management', 'level': 'Soft'},
        {'skill_id': 13, 'name': 'Communication', 'category': 'Soft Skills', 'level': 'Soft'},
        {'skill_id': 14, 'name': 'Problem Solving', 'category': 'Analytical', 'level': 'Soft'},
        {'skill_id': 15, 'name': 'Data Analysis', 'category': 'Analytical', 'level': 'Technical'},
        {'skill_id': 16, 'name': 'Sales', 'category': 'Business', 'level': 'Soft'},
        {'skill_id': 17, 'name': 'Marketing', 'category': 'Business', 'level': 'Soft'},
        {'skill_id': 18, 'name': 'Customer Service', 'category': 'Business', 'level': 'Soft'},
        {'skill_id': 19, 'name': 'Financial Analysis', 'category': 'Finance', 'level': 'Technical'},
        {'skill_id': 20, 'name': 'Legal Compliance', 'category': 'Legal', 'level': 'Technical'}
    ]
    
    return pd.DataFrame(skills)

def generate_employee_skills(employees_df, skills_df):
    """Generate employee skills data"""
    employee_skills = []
    
    for _, employee in employees_df.iterrows():
        # Assign 2-6 skills per employee
        num_skills = random.randint(2, 6)
        assigned_skills = random.sample(skills_df['skill_id'].tolist(), num_skills)
        
        for skill_id in assigned_skills:
            employee_skills.append({
                'employee_skill_id': len(employee_skills) + 1,
                'employee_id': employee['employee_id'],
                'skill_id': skill_id,
                'proficiency_level': random.choice(['Beginner', 'Intermediate', 'Advanced', 'Expert']),
                'certified': random.choice([True, False]),
                'acquired_date': employee['hire_date']
            })
    
    return pd.DataFrame(employee_skills)

def generate_training():
    """Generate training data"""
    training = [
        {'training_id': 1, 'name': 'Python Fundamentals', 'category': 'Programming', 'duration_hours': 40, 'cost': 500},
        {'training_id': 2, 'name': 'Advanced JavaScript', 'category': 'Programming', 'duration_hours': 30, 'cost': 400},
        {'training_id': 3, 'name': 'AWS Cloud Practitioner', 'category': 'Cloud', 'duration_hours': 20, 'cost': 300},
        {'training_id': 4, 'name': 'Leadership Skills', 'category': 'Management', 'duration_hours': 16, 'cost': 800},
        {'training_id': 5, 'name': 'Project Management Professional', 'category': 'Management', 'duration_hours': 35, 'cost': 1200},
        {'training_id': 6, 'name': 'Data Science Fundamentals', 'category': 'Analytics', 'duration_hours': 50, 'cost': 600},
        {'training_id': 7, 'name': 'Sales Techniques', 'category': 'Business', 'duration_hours': 24, 'cost': 350},
        {'training_id': 8, 'name': 'Customer Service Excellence', 'category': 'Business', 'duration_hours': 12, 'cost': 200},
        {'training_id': 9, 'name': 'Financial Analysis', 'category': 'Finance', 'duration_hours': 28, 'cost': 450},
        {'training_id': 10, 'name': 'Legal Compliance Training', 'category': 'Legal', 'duration_hours': 8, 'cost': 150}
    ]
    
    return pd.DataFrame(training)

def generate_training_enrollments(employees_df, training_df):
    """Generate training enrollments"""
    enrollments = []
    
    for _, employee in employees_df.iterrows():
        # Enroll in 1-3 training programs
        num_enrollments = random.randint(1, 3)
        enrolled_trainings = random.sample(training_df['training_id'].tolist(), num_enrollments)
        
        for training_id in enrolled_trainings:
            enrollment_date = datetime.strptime(employee['hire_date'], '%Y-%m-%d') + timedelta(days=random.randint(30, 365))
            
            enrollments.append({
                'enrollment_id': len(enrollments) + 1,
                'employee_id': employee['employee_id'],
                'training_id': training_id,
                'enrollment_date': enrollment_date.strftime('%Y-%m-%d'),
                'completion_date': (enrollment_date + timedelta(days=random.randint(30, 90))).strftime('%Y-%m-%d'),
                'status': random.choice(['Enrolled', 'In Progress', 'Completed', 'Dropped']),
                'grade': random.choice(['A', 'B', 'C', 'D', 'F', None]),
                'feedback_score': random.randint(1, 5) if random.choice([True, False]) else None
            })
    
    return pd.DataFrame(enrollments)

def create_schema_definition(datasets):
    """Create schema definition for the complex HR dataset"""
    
    schema = {
        "tables": [],
        "relationships": [
            {
                "source_table": "departments",
                "source_column": "department_id",
                "target_table": "employees",
                "target_column": "department_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "positions",
                "source_column": "position_id",
                "target_table": "employees",
                "target_column": "position_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "employees",
                "source_column": "employee_id",
                "target_table": "employees",
                "target_column": "manager_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "employees",
                "source_column": "employee_id",
                "target_table": "salaries",
                "target_column": "employee_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "employees",
                "source_column": "employee_id",
                "target_table": "performance",
                "target_column": "employee_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "employees",
                "source_column": "employee_id",
                "target_table": "performance",
                "target_column": "reviewer_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "projects",
                "source_column": "project_id",
                "target_table": "project_assignments",
                "target_column": "project_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "employees",
                "source_column": "employee_id",
                "target_table": "project_assignments",
                "target_column": "employee_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "skills",
                "source_column": "skill_id",
                "target_table": "employee_skills",
                "target_column": "skill_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "employees",
                "source_column": "employee_id",
                "target_table": "employee_skills",
                "target_column": "employee_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "training",
                "source_column": "training_id",
                "target_table": "training_enrollments",
                "target_column": "training_id",
                "relationship_type": "one-to-many"
            },
            {
                "source_table": "employees",
                "source_column": "employee_id",
                "target_table": "training_enrollments",
                "target_column": "employee_id",
                "relationship_type": "one-to-many"
            }
        ]
    }
    
    # Add table definitions
    for name, df in datasets.items():
        table_schema = {
            "name": name,
            "description": f"HR {name.replace('_', ' ').title()} data",
            "columns": []
        }
        
        for col in df.columns:
            col_schema = {
                "name": col,
                "data_type": str(df[col].dtype),
                "is_nullable": df[col].isnull().any(),
                "is_primary_key": col.endswith('_id') and name in col,
                "is_unique": df[col].nunique() == len(df)
            }
            table_schema["columns"].append(col_schema)
        
        schema["tables"].append(table_schema)
    
    return schema

if __name__ == "__main__":
    datasets, schema = generate_complex_hr_dataset()
    print("\n🎉 Complex HR dataset generated successfully!")
    print("📁 Files created:")
    for name in datasets.keys():
        print(f"   - complex_hr_{name}.csv")
    print("   - complex_hr_schema.json") 