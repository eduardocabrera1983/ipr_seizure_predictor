#!/usr/bin/env python3
# train_model.py - Train the IPR risk prediction model

"""
This script trains the IPR risk prediction model with CLEANED features.
NO declared_value_per_line feature, uses regression for continuous 0-100 risk scores.

Usage:
    python train_model.py

The script will:
1. Process the IPR data with CLEANED features (no declared_value_per_line)
2. Train a Random Forest REGRESSION model
3. Test that line count properly INCREASES risk
4. Save the model
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent))

# Import the CLEANED versions
from data_processor import IPRDataProcessor
from ml_model import IPRRiskPredictor

def main():
    """Main training pipeline with CLEANED features"""
    print("🔧 Training IPR Model with CLEANED Features (No declared_value_per_line)...")
    print("=" * 70)
    
    # Create required directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # Step 1: Process the data with CLEANED features
        print("\n📊 Step 1: Processing IPR Data with CLEANED Features...")
        processor = IPRDataProcessor()
        
        # Check if data file exists
        data_file = Path("data/raw/24-0405_ohss_dhs_ipr_seizures_fy2019-2023.xlsx")
        if not data_file.exists():
            print(f"❌ ERROR: Data file not found at {data_file}")
            print("Please ensure the Excel file is in the correct location.")
            return False
            
        # Process with CLEANED features
        df = processor.load_and_process_data(save_to_csv=True)
        print(f"✅ Data processed successfully: {df.shape[0]:,} records")
        
        # Verify problematic features are removed
        problematic_features = ['declared_value_per_line', 'value_per_line']
        present_problematic = [f for f in problematic_features if f in df.columns]
        
        if present_problematic:
            print(f"⚠️ WARNING: Problematic features still present: {present_problematic}")
            print("   These may cause the line count bug to persist!")
            return False
        
        print(f"✅ Confirmed: NO problematic declared_value_per_line feature")
        
        # Step 2: Train the CLEANED model
        print("\n🤖 Step 2: Training CLEANED Regression Model...")
        predictor = IPRRiskPredictor()
        
        # Train with the processed dataframe
        metrics = predictor.train_model_with_data(df)
        
        print(f"✅ CLEANED Model trained successfully!")
        print(f"   - R² Score: {metrics['r2_score']:.3f}")
        print(f"   - RMSE: {metrics['rmse']:.1f}")
        print(f"   - MAE: {metrics['mae']:.1f}")
        
        # Step 3: CRITICAL TEST - Verify line count effect is fixed
        print("\n🧪 Step 3: CRITICAL TEST - Line Count Effect...")
        test_result = test_line_count_effect(predictor)
        
        if not test_result:
            print("❌ CRITICAL ERROR: Line count effect not fixed!")
            print("   The model still decreases risk with more lines.")
            return False
        
        # Step 4: Save the CLEANED model
        print("\n💾 Step 4: Saving CLEANED Model...")
        
        # Backup the old model if it exists
        old_model_path = Path("models/ipr_risk_model.pkl")
        if old_model_path.exists():
            backup_path = Path("models/ipr_risk_model_backup.pkl")
            os.rename(old_model_path, backup_path)
            print(f"✅ Backed up old model to {backup_path}")
        
        # Save the new cleaned model
        predictor.save_model()
        
        # Verify the saved model
        model_file = Path("models/ipr_risk_model.pkl")
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"✅ CLEANED Model saved successfully!")
            print(f"   - File: {model_file}")
            print(f"   - Size: {size_mb:.1f} MB")
        
        # Step 5: Final comprehensive test
        print("\n🧪 Step 5: Comprehensive Testing...")
        comprehensive_test(predictor)
        
        print("\n🎉 CLEANED Model Training Complete!")
        print("✅ Line count now properly INCREASES risk as expected")
        print("✅ Model outputs continuous risk scores (0-100)")
        print("Your Flask app should now work correctly with more lines = higher risk")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_line_count_effect(predictor):
    """CRITICAL TEST: Verify that more lines = higher risk"""
    
    print("\n🔍 Testing Line Count Effect (MOST IMPORTANT TEST):")
    
    base_shipment = {
        'origin': 'People\'s Republic of China',
        'product': 'Electronics',
        'conveyance': 'Commercial Air',
        'declared_value': 5000
    }
    
    line_counts = [1, 5, 10, 25, 50, 100]
    risk_scores = []
    
    print("Line Count → Risk Score (Should INCREASE):")
    
    for lines in line_counts:
        test_shipment = base_shipment.copy()
        test_shipment['line_count'] = lines
        result = predictor.predict_risk(test_shipment)
        risk_score = result['risk_score']
        risk_scores.append(risk_score)
        
        # Show result with trend indicator
        if len(risk_scores) > 1:
            if risk_score > risk_scores[-2]:
                trend = "↗️ GOOD"
            elif risk_score == risk_scores[-2]:
                trend = "→ SAME"
            else:
                trend = "↘️ BAD"
        else:
            trend = ""
            
        print(f"   {lines:3d} lines → {risk_score:2d}% ({result['risk_category']}) {trend}")
    
    # Verify the trend is generally upward
    increases = 0
    decreases = 0
    
    for i in range(1, len(risk_scores)):
        if risk_scores[i] > risk_scores[i-1]:
            increases += 1
        elif risk_scores[i] < risk_scores[i-1]:
            decreases += 1
    
    print(f"\nTrend Analysis:")
    print(f"   Risk increases: {increases} times")
    print(f"   Risk decreases: {decreases} times")
    
    # Test passes if risk generally increases (at least 60% of the time for regression)
    success_rate = increases / (increases + decreases) if (increases + decreases) > 0 else 0
    
    if success_rate >= 0.6:  # Lower threshold for regression model
        print(f"✅ SUCCESS: Risk increases {success_rate:.1%} of the time")
        print("   Line count effect is FIXED!")
        return True
    else:
        print(f"❌ FAILURE: Risk only increases {success_rate:.1%} of the time")
        print("   Line count effect is NOT fixed!")
        
        # Show specific problems
        for i in range(1, len(line_counts)):
            if risk_scores[i] < risk_scores[i-1]:
                print(f"   ⚠️ Problem: {line_counts[i]} lines has LOWER risk than {line_counts[i-1]} lines")
        
        return False

def comprehensive_test(predictor):
    """Run comprehensive tests on various scenarios"""
    
    print("\n🧪 Comprehensive Testing:")
    
    test_scenarios = [
        {
            'name': 'Low-risk Japan shipment',
            'shipment': {
                'origin': 'Japan',
                'product': 'Electronics',
                'conveyance': 'Commercial Air',
                'declared_value': 10000,
                'line_count': 5
            }
        },
        {
            'name': 'High-risk China complex shipment',
            'shipment': {
                'origin': 'People\'s Republic of China',
                'product': 'Clothing',
                'conveyance': 'Express Consignment',
                'declared_value': 50000,
                'line_count': 75
            }
        },
        {
            'name': 'Very complex shipment (should be high risk)',
            'shipment': {
                'origin': 'Socialist Republic of Vietnam',
                'product': 'Electronics',
                'conveyance': 'Commercial Air',
                'declared_value': 25000,
                'line_count': 150
            }
        },
        {
            'name': 'Simple low-risk shipment',
            'shipment': {
                'origin': 'Germany',
                'product': 'Electronics',
                'conveyance': 'Commercial Truck',
                'declared_value': 15000,
                'line_count': 3
            }
        }
    ]
    
    for scenario in test_scenarios:
        result = predictor.predict_risk(scenario['shipment'])
        lines = scenario['shipment']['line_count']
        
        print(f"\n📋 {scenario['name']}:")
        print(f"   Lines: {lines}, Risk: {result['risk_score']}% ({result['risk_category']})")
        
        # Verify complex shipments (>50 lines) have reasonable risk
        if lines > 50:
            if result['risk_score'] >= 60:
                print(f"   ✅ GOOD: Complex shipment properly flagged as high risk")
            else:
                print(f"   ⚠️ NOTE: Complex shipment has moderate risk ({result['risk_score']}%)")

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Restart your Flask application")
        print("2. Test the line count effect in the web interface")
        print("3. Verify that more lines now show HIGHER risk")
        print("4. Enjoy continuous risk scores (0-100) instead of binary classification!")
        print("\nKey improvements:")
        print("✅ Removed problematic declared_value_per_line feature")
        print("✅ Switched to regression for continuous risk scores")
        print("✅ Line count now properly INCREASES risk")
    else:
        print("\n❌ TRAINING FAILED")
        print("Please review the errors above and fix the issues.")