#!/usr/bin/env python3
# train_model.py - Train and save the IPR risk prediction model

"""
This script trains the IPR risk prediction model and saves it as a pickle file.
Run this before starting your Flask app to generate the required model file.

Usage:
    python train_model.py

The script will:
1. Process the IPR data from Excel file
2. Train a Random Forest model
3. Save the trained model to models/ipr_risk_model.pkl
"""

import os
import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from data_processor import IPRDataProcessor
from ml_model import IPRRiskPredictor

def main():
    """Main training pipeline"""
    print("🚀 Starting IPR Model Training Pipeline...")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        # Step 1: Process the data
        print("\n📊 Step 1: Processing IPR Data...")
        processor = IPRDataProcessor()
        
        # Check if data file exists
        data_file = Path("data/raw/24-0405_ohss_dhs_ipr_seizures_fy2019-2023.xlsx")
        if not data_file.exists():
            print(f"❌ ERROR: Data file not found at {data_file}")
            print("Please ensure the Excel file is in the correct location.")
            return False
            
        df = processor.load_and_process_data()
        print(f"✅ Data processed successfully: {df.shape[0]:,} records")
        
        # Step 2: Train the model
        print("\n🤖 Step 2: Training Machine Learning Model...")
        predictor = IPRRiskPredictor()
        metrics = predictor.train_model()
        
        print(f"✅ Model trained successfully!")
        print(f"   - Accuracy: {metrics['accuracy']:.3f}")
        print(f"   - AUC-ROC: {metrics['auc_roc']:.3f}")
        
        # Step 3: Save the model
        print("\n💾 Step 3: Saving Model...")
        predictor.save_model()
        
        # Verify the saved model
        model_file = Path("models/ipr_risk_model.pkl")
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"✅ Model saved successfully!")
            print(f"   - File: {model_file}")
            print(f"   - Size: {size_mb:.1f} MB")
        
        print("\n🎉 Training Pipeline Complete!")
        print("Your Flask app is now ready to run with the trained model.")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: Required file not found: {e}")
        print("Please ensure all data files are in the correct location.")
        return False
        
    except Exception as e:
        print(f"❌ ERROR during training: {e}")
        print("Check your data files and try again.")
        return False

def create_sample_data():
    """Create sample data if the real dataset is not available"""
    print("\n📋 Creating sample data for testing...")
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample data structure
    np.random.seed(42)
    n_samples = 1000
    
    countries = [
        "People's Republic of China", "United States of America", 
        "United Mexican States", "Socialist Republic of Vietnam",
        "Republic of India", "Republic of Turkey", "Japan",
        "Federal Republic of Germany", "Canada"
    ]
    
    products = [
        "Electronics", "Clothing", "Jewelry", "Watch", "Footwear",
        "Computer/Computer Parts", "Toys", "Luggage"
    ]
    
    conveyances = [
        "Express Consignment", "Commercial Air", "Commercial Vessel", 
        "Commercial Truck"
    ]
    
    # Generate sample data
    sample_data = {
        'ORIGIN': np.random.choice(countries, n_samples),
        'PRODUCT': np.random.choice(products, n_samples),
        'CONVEYANCE': np.random.choice(conveyances, n_samples),
        'SUM_OF_MSRP': np.random.lognormal(8, 2, n_samples).astype(int),
        'COUNT_OF_LINES': np.random.randint(1, 50, n_samples),
        'FISCAL_YEAR': np.random.choice([2019, 2020, 2021, 2022, 2023], n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample data
    os.makedirs('data/raw', exist_ok=True)
    sample_file = 'data/raw/sample_ipr_data.xlsx'
    df.to_excel(sample_file, index=False)
    
    print(f"✅ Sample data created: {sample_file}")
    print(f"   - Records: {len(df):,}")
    print("You can now run the training pipeline with sample data.")
    
    return sample_file

if __name__ == "__main__":
    # Check if data file exists
    data_file = Path("data/raw/24-0405_ohss_dhs_ipr_seizures_fy2019-2023.xlsx")
    
    if not data_file.exists():
        print("⚠️  Original data file not found.")
        choice = input("Create sample data for testing? (y/n): ").lower().strip()
        
        if choice == 'y':
            sample_file = create_sample_data()
            # Update the data processor to use sample data
            choice = input("Train model with sample data? (y/n): ").lower().strip()
            if choice == 'y':
                main()
        else:
            print("Please add the original data file and run again.")
    else:
        main()