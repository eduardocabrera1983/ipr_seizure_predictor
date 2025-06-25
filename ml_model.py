# ml_model.py - Machine Learning Model for IPR Risk Prediction
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os


# Import your existing handlers
from country_handler import USTREnhancedCountryHandler


class IPRRiskPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.model_metrics = {}
        
        # Initialize your existing handlers
        self.country_handler = USTREnhancedCountryHandler()
        
        # HTS mapping from your data_processor.py
        from constants import HTS_MAPPING
        self.hts_mapping = HTS_MAPPING
        
        
        
    def train_model(self):
        """Train the IPR risk prediction model"""
        print("Training IPR Risk Prediction Model...")
        
        # Load processed data
        df = pd.read_csv('data/processed/ipr_enhanced.csv')
        print(f"✓ Loaded {df.shape[0]:,} records")
        
        # Prepare features
        feature_columns = [
            'COUNT_OF_LINES', 'value_per_line', 'log_msrp', 'log_lines',
            'country_risk_score', 'product_risk_score', 'conveyance_risk_score',
            'duty_rate', 'tariff_evasion_incentive', 'FISCAL_YEAR',
            'is_china', 'is_hong_kong', 'is_express',
            'high_duty_product', 'complex_product'
        ]
        
        # Handle categorical variables
        categorical_features = ['PRODUCT', 'CONVEYANCE', 'hts_chapter', 'product_complexity']
        
        for col in categorical_features:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            feature_columns.append(col + '_encoded')
        
        self.feature_columns = feature_columns
        
        # Prepare data
        X = df[feature_columns].fillna(0)
        y = df['high_value_seizure']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.model_metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"✓ Model trained - AUC: {self.model_metrics['auc_roc']:.3f}")
        print(f"✓ Accuracy: {self.model_metrics['accuracy']:.3f}")
        
        return self.model_metrics
    
       
    def predict_risk(self, shipment_data):
        """Predict risk using trained Random Forest model"""
        
        # Check if model is trained
        if self.model is None:
            # Fallback to rule-based prediction
            print("⚠️ Model not trained, using rule-based fallback")
            return self._rule_based_prediction(shipment_data)
        
        try:
            # Step 1: Create feature vector from input data
            features = self._prepare_features(shipment_data)
            
            # Step 2: Get prediction probabilities from trained model
            risk_probabilities = self.model.predict_proba(features)
            
            # Step 3: Extract probability of positive class (high-risk seizure)
            risk_prob = risk_probabilities[0][1]
            
            # Step 4: Categorize risk and generate response
            risk_category = self._categorize_risk(risk_prob)
            
            return {
                'risk_probability': risk_prob,
                'risk_score': int(risk_prob * 100),
                'risk_category': risk_category,
                'recommendations': self._generate_recommendations(shipment_data, risk_prob)
            }
        except Exception as e:
            print(f"❌ ML prediction failed: {e}")
            return self._rule_based_prediction(shipment_data)


    def _rule_based_prediction(self, shipment_data):
        """Fallback rule-based prediction when ML model unavailable"""
        risk_prob = 0.3  # Base risk
        
        # Use USTR data if available
        if hasattr(self, 'country_handler'):
            country_assessment = self.country_handler.calculate_composite_risk_score(
                shipment_data.get('origin', 'Unknown')
            )
            country_risk = country_assessment['final_risk_score'] / 100
            risk_prob = country_risk * 0.7 + risk_prob * 0.3
        
        # Basic rule adjustments
        if shipment_data.get('origin', '').lower().find('china') != -1:
            risk_prob += 0.2
        if shipment_data.get('conveyance', '').lower().find('express') != -1:
            risk_prob += 0.15
        
        risk_prob = min(0.95, max(0.05, risk_prob))
        
        return {
            'risk_probability': risk_prob,
            'risk_score': int(risk_prob * 100),
            'risk_category': self._categorize_risk(risk_prob),
            'recommendations': self._generate_recommendations(shipment_data, risk_prob)
        }
        
      
    
    def _prepare_features(self, shipment_data):
        """Convert raw shipment data into model-ready feature vector"""
        
        # Initialize feature dictionary
        feature_dict = {}
        
        # Basic numerical features
        feature_dict['COUNT_OF_LINES'] = shipment_data.get('line_count', 1)
        declared_value = max(shipment_data.get('declared_value', 1), 1)  # Avoid log(0)
        line_count = max(shipment_data.get('line_count', 1), 1)
        
        feature_dict['value_per_line'] = declared_value / line_count
        feature_dict['log_msrp'] = np.log1p(declared_value)  # log1p handles log(0)
        feature_dict['log_lines'] = np.log1p(line_count)
        
        # Get country risk using USTR handler
        country_name = shipment_data.get('origin', 'Unknown')
        country_assessment = self.country_handler.calculate_composite_risk_score(country_name)
        feature_dict['country_risk_score'] = min(country_assessment['historical_data']['seizures'], 500)
        
        # Get product info from HTS mapping
        product_name = shipment_data.get('product', 'other')
        product_info = self.hts_mapping.get(product_name, self.hts_mapping['other'])
        feature_dict['duty_rate'] = product_info['duty_rate']
        
        # Estimate product and conveyance risk scores
        feature_dict['product_risk_score'] = self._get_product_risk_score(product_name)
        feature_dict['conveyance_risk_score'] = self._get_conveyance_risk_score(
            shipment_data.get('conveyance', 'Unknown')
        )
        
        # Calculate tariff evasion incentive
        feature_dict['tariff_evasion_incentive'] = (
            feature_dict['duty_rate'] * feature_dict['value_per_line'] / 100
        )
        
        # Current fiscal year
        from datetime import datetime
        current_year = datetime.now().year
        fiscal_year = current_year if datetime.now().month >= 10 else current_year - 1
        feature_dict['FISCAL_YEAR'] = fiscal_year
        
        # Binary flags
        origin = shipment_data.get('origin', '').lower()
        feature_dict['is_china'] = 1 if 'china' in origin else 0
        feature_dict['is_hong_kong'] = 1 if 'hong kong' in origin else 0
        feature_dict['is_express'] = 1 if 'express' in shipment_data.get('conveyance', '').lower() else 0
        
        # Product flags
        feature_dict['high_duty_product'] = 1 if feature_dict['duty_rate'] > 10 else 0
        feature_dict['complex_product'] = 1 if product_info['complexity'] == 'High' else 0
        
        # Handle categorical features using saved encoders
        categorical_mappings = {
            'PRODUCT': product_name,
            'CONVEYANCE': shipment_data.get('conveyance', 'Unknown'),
            'hts_chapter': product_info['hts_chapter'],
            'product_complexity': product_info['complexity']
        }
        
        for col, raw_value in categorical_mappings.items():
            if col in self.label_encoders:
                try:
                    encoded_value = self.label_encoders[col].transform([str(raw_value)])[0]
                except ValueError:
                    # Handle unseen categories
                    encoded_value = 0
                feature_dict[col + '_encoded'] = encoded_value
            else:
                feature_dict[col + '_encoded'] = 0
        
        # Create feature array in correct order
        feature_array = np.array([[feature_dict.get(col, 0) for col in self.feature_columns]])
        
        return feature_array

    def _get_product_risk_score(self, product_name):
        """Get product risk score based on historical patterns"""
        from constants import PRODUCT_RISK_SCORES
        return PRODUCT_RISK_SCORES.get(product_name, 200)

    
    def _get_conveyance_risk_score(self, conveyance):
        """Get conveyance risk score based on historical patterns"""
        from constants import CONVEYANCE_RISK_SCORES
        return CONVEYANCE_RISK_SCORES.get(conveyance, 500)

    def _categorize_risk(self, probability):
        if probability >= 0.7: return 'CRITICAL'
        elif probability >= 0.5: return 'HIGH'
        elif probability >= 0.3: return 'MEDIUM'
        else: return 'LOW'
    
    def _generate_recommendations(self, shipment_data, risk_prob):
        recommendations = []
        if risk_prob >= 0.7:
            recommendations.append("🚨 CRITICAL RISK: Consider alternative sourcing")
        elif risk_prob >= 0.5:
            recommendations.append("⚠️ HIGH RISK: Enhanced documentation required")
        else:
            recommendations.append("✅ Standard compliance verification recommended")
        return recommendations
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'metrics': self.model_metrics
        }
        joblib.dump(model_data, 'models/ipr_risk_model.pkl')
        print("✓ Model saved to models/ipr_risk_model.pkl")

# Train and save model
if __name__ == "__main__":
    predictor = IPRRiskPredictor()
    metrics = predictor.train_model()
    predictor.save_model()
    
    # Test prediction
    test_shipment = {
        'origin': 'People\'s Republic of China',
        'product': 'Clothing',
        'conveyance': 'Express Consignment',
        'declared_value': 50000,
        'line_count': 25
    }
    
    result = predictor.predict_risk(test_shipment)
    print(f"\n✓ Test prediction: {result['risk_score']}% risk ({result['risk_category']})")
    
    