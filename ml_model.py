# ml_model.py - Machine Learning Model for IPR Risk Prediction
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

class IPRRiskPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.model_metrics = {}
        
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
        """Predict risk for new shipment"""
        
        # Create feature vector (simplified for demo)
        risk_prob = 0.5  # Placeholder - you'll replace with actual model prediction
        
        # Basic risk calculation based on shipment data
        if shipment_data.get('origin') == 'People\'s Republic of China':
            risk_prob += 0.2
        if shipment_data.get('conveyance') == 'Express Consignment':
            risk_prob += 0.15
        if shipment_data.get('line_count', 0) > 25:
            risk_prob += 0.1
            
        risk_prob = min(0.95, max(0.05, risk_prob))
        
        risk_category = self._categorize_risk(risk_prob)
        
        return {
            'risk_probability': risk_prob,
            'risk_score': int(risk_prob * 100),
            'risk_category': risk_category,
            'recommendations': self._generate_recommendations(shipment_data, risk_prob)
        }
    
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