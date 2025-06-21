# data_processor.py - IPR Data Processing with USTR Enhancement
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import warnings
import openpyxl 
warnings.filterwarnings('ignore')

class IPRDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_and_process_data(self):
        """Load and process IPR data with HTS enhancement"""
        print("Loading IPR data...")
        
        # Load the Excel file
        from pathlib import Path
        
        # Get the path where this script lives
        BASE_DIR = Path(__file__).resolve().parent

        # Build the full path to the Excel file
        file_path = BASE_DIR / "data" / "raw" / "24-0405_ohss_dhs_ipr_seizures_fy2019-2023.xlsx"

        print(f"Loading file from: {file_path}")
        df = pd.read_excel(file_path)
        print(f"✓ Loaded {df.shape[0]:,} records")
        
        # Add HTS-enhanced features
        df = self.add_hts_features(df)
        df = self.create_risk_features(df)
        df = self.create_target_variable(df)
        
        return df
    
    def add_hts_features(self, df):
        """Add HTS-based features"""
        
        # HTS mapping with duty rates and complexity
        hts_mapping = {
            'Clothing': {'duty_rate': 12.5, 'complexity': 'High', 'hts_chapter': '62'},
            'Footwear': {'duty_rate': 15.2, 'complexity': 'Medium', 'hts_chapter': '64'},
            'Purse/Wallets': {'duty_rate': 8.1, 'complexity': 'Medium', 'hts_chapter': '42'},
            'Watch': {'duty_rate': 3.8, 'complexity': 'High', 'hts_chapter': '91'},
            'Jewelry': {'duty_rate': 5.5, 'complexity': 'High', 'hts_chapter': '71'},
            'Electronics': {'duty_rate': 2.1, 'complexity': 'High', 'hts_chapter': '85'},
            'Luggage': {'duty_rate': 9.2, 'complexity': 'Medium', 'hts_chapter': '42'},
            'Toys': {'duty_rate': 6.8, 'complexity': 'Medium', 'hts_chapter': '95'},
            'Label/Emblems': {'duty_rate': 7.5, 'complexity': 'Low', 'hts_chapter': '83'},
            'Movies Music Media': {'duty_rate': 3.2, 'complexity': 'Medium', 'hts_chapter': '85'},
            'Computer/Computer Parts': {'duty_rate': 1.8, 'complexity': 'High', 'hts_chapter': '84'},
            'Household Appliances': {'duty_rate': 4.5, 'complexity': 'Medium', 'hts_chapter': '85'},
            'Drug': {'duty_rate': 0.0, 'complexity': 'High', 'hts_chapter': '30'},
            'Computer Software': {'duty_rate': 0.0, 'complexity': 'High', 'hts_chapter': '85'},
            'Computer Chips': {'duty_rate': 1.2, 'complexity': 'High', 'hts_chapter': '85'},
            'Office Equipment': {'duty_rate': 3.1, 'complexity': 'Medium', 'hts_chapter': '84'},
            'Golf': {'duty_rate': 5.7, 'complexity': 'Low', 'hts_chapter': '95'},
            'other': {'duty_rate': 7.5, 'complexity': 'Medium', 'hts_chapter': '99'}
        }
        
        # Add new columns
        df['duty_rate'] = df['PRODUCT'].map(lambda x: hts_mapping.get(x, hts_mapping['other'])['duty_rate'])
        df['product_complexity'] = df['PRODUCT'].map(lambda x: hts_mapping.get(x, hts_mapping['other'])['complexity'])
        df['hts_chapter'] = df['PRODUCT'].map(lambda x: hts_mapping.get(x, hts_mapping['other'])['hts_chapter'])
        
        print("✓ Added HTS features: duty_rate, product_complexity, hts_chapter")
        return df
    
    def create_risk_features(self, df):
        """Create risk-based features"""
        
        # Historical risk scores
        country_stats = df.groupby('ORIGIN')['SUM_OF_MSRP'].agg(['count', 'mean'])
        df['country_risk_score'] = df['ORIGIN'].map(country_stats['count'])
        df['country_avg_value'] = df['ORIGIN'].map(country_stats['mean'])
        
        product_stats = df.groupby('PRODUCT')['SUM_OF_MSRP'].agg(['count', 'mean'])
        df['product_risk_score'] = df['PRODUCT'].map(product_stats['count'])
        
        conveyance_stats = df.groupby('CONVEYANCE')['SUM_OF_MSRP'].agg(['count', 'mean'])
        df['conveyance_risk_score'] = df['CONVEYANCE'].map(conveyance_stats['count'])
        
        # Value-based features
        df['value_per_line'] = df['SUM_OF_MSRP'] / df['COUNT_OF_LINES']
        df['log_msrp'] = np.log1p(df['SUM_OF_MSRP'])
        df['log_lines'] = np.log1p(df['COUNT_OF_LINES'])
        
        # HTS-enhanced features
        df['tariff_evasion_incentive'] = df['duty_rate'] * df['value_per_line'] / 100
        df['high_duty_product'] = (df['duty_rate'] > 10).astype(int)
        df['complex_product'] = (df['product_complexity'] == 'High').astype(int)
        
        # Geographic indicators
        df['is_china'] = (df['ORIGIN'] == 'People\'s Republic of China').astype(int)
        df['is_hong_kong'] = (df['ORIGIN'] == 'Hong Kong Special Administrative Region').astype(int)
        df['is_express'] = (df['CONVEYANCE'] == 'Express Consignment').astype(int)
        
        print("✓ Created risk features")
        return df
    
    def create_target_variable(self, df):
        """Create target variable for prediction"""
        
        # High-value seizures (above 75th percentile)
        value_threshold = df['SUM_OF_MSRP'].quantile(0.75)
        df['high_value_seizure'] = (df['SUM_OF_MSRP'] > value_threshold).astype(int)
        
        print(f"✓ Target created: {df['high_value_seizure'].sum()} high-value seizures ({df['high_value_seizure'].mean():.1%})")
        return df

# Test the processor
if __name__ == "__main__":
    processor = IPRDataProcessor()
    df = processor.load_and_process_data()
    
    # Save processed data
    df.to_csv('data/processed/ipr_enhanced.csv', index=False)
    print("✓ Saved processed data to data/processed/ipr_enhanced.csv")
    
    # Show sample
    print("\nSample data:")
    print(df[['PRODUCT', 'ORIGIN', 'duty_rate', 'high_value_seizure']].head())