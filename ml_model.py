# ml_model.py - ML Model with REGRESSION (0-100 risk score)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Regression metrics
import joblib
import os

# Import existing handlers
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
        
    def train_model_with_data(self, df):
        """Train the IPR risk prediction model with provided DataFrame"""
        print("Training IPR Risk Prediction Model with REGRESSION (0-100 risk score)...")
        
        print(f"✓ Using provided DataFrame with {df.shape[0]:,} records")
        
        feature_columns = [
            'declared_value_category', 'declared_line_category', 
            'complexity_multiplier', 'high_value_shipment', 'complex_shipment',
            'extremely_complex', 'very_complex',
            
            # Geographic risk factors
            'is_china', 'is_hong_kong', 'is_vietnam', 'is_mexico',
            
            # Conveyance risk factors  
            'is_express', 'is_air_freight', 'is_vessel',
            
            # Product risk factors
            'duty_rate', 'high_duty_product', 'complex_product',
            'is_clothing', 'is_electronics', 'is_jewelry',
            
            # Historical patterns (legitimate - not from current seizure)
            'country_seizure_history', 'product_seizure_history', 'conveyance_seizure_history',
            'repeat_country', 'repeat_product',
            
            # Temporal trends
            'fiscal_year_trend', 'FISCAL_YEAR',
            
            # Risk interaction patterns
            'china_clothing', 'china_electronics', 'express_clothing', 'high_duty_express',
            
            # Tariff evasion
            'tariff_evasion_incentive'
        ]
        
        # Handle categorical variables (encode them)
        categorical_features = ['PRODUCT', 'CONVEYANCE', 'ORIGIN', 'hts_chapter', 'product_complexity']
        
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                feature_columns.append(col + '_encoded')
        
        self.feature_columns = feature_columns
        
        # Verify all features exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # ✅ CREATE CONTINUOUS RISK SCORE TARGET (0-100)
        df = self._create_continuous_risk_target(df)
        
        # Prepare data
        X = df[feature_columns].fillna(0)
        y = df['risk_score_target']  # Use continuous target
        
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"✓ Target statistics: mean={y.mean():.1f}, std={y.std():.1f}, range=[{y.min():.1f}, {y.max():.1f}]")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Training set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        
        # ✅ REGRESSION MODEL
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        print("✓ Training Random Forest REGRESSOR...")
        self.model.fit(X_train, y_train)
        
        # Evaluate with regression metrics
        print("✓ Evaluating model performance...")
        y_pred = self.model.predict(X_test)
        
        # Ensure predictions are within 0-100 range
        y_pred_clipped = np.clip(y_pred, 0, 100)
        
        self.model_metrics = {
            'r2_score': r2_score(y_test, y_pred_clipped),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_clipped)),
            'mae': mean_absolute_error(y_test, y_pred_clipped),
            'mean_prediction': y_pred_clipped.mean(),
            'std_prediction': y_pred_clipped.std()
        }
        
        print(f"✅ Model trained successfully!")
        print(f"  - R² Score: {self.model_metrics['r2_score']:.3f}")
        print(f"  - RMSE: {self.model_metrics['rmse']:.1f}")
        print(f"  - MAE: {self.model_metrics['mae']:.1f}")
        print(f"  - Prediction range: [{y_pred_clipped.min():.1f}, {y_pred_clipped.max():.1f}]")
        
        # Show feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Top 10 Most Important Features (Regression Version):")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.3f}")
        
        return self.model_metrics
    
    def _create_continuous_risk_target(self, df):
        """Create a continuous risk score target (0-100) based on multiple factors"""
        print("Creating continuous risk score target (0-100)...")
        
        # Initialize base risk score
        risk_score = np.full(len(df), 30.0)  # Base risk of 30%
        
        # ✅ VALUE-BASED RISK (0-25 points)
        value_percentiles = df['SUM_OF_MSRP'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
        risk_score += np.where(df['SUM_OF_MSRP'] > value_percentiles[0.95], 25,
                      np.where(df['SUM_OF_MSRP'] > value_percentiles[0.9], 20,
                      np.where(df['SUM_OF_MSRP'] > value_percentiles[0.75], 15,
                      np.where(df['SUM_OF_MSRP'] > value_percentiles[0.5], 10,
                      np.where(df['SUM_OF_MSRP'] > value_percentiles[0.25], 5, 0)))))
        
        # ✅ LINE COUNT RISK (0-20 points) - MORE LINES = MORE RISK
        risk_score += np.where(df['COUNT_OF_LINES'] >= 100, 20,
                      np.where(df['COUNT_OF_LINES'] >= 50, 15,
                      np.where(df['COUNT_OF_LINES'] >= 25, 10,
                      np.where(df['COUNT_OF_LINES'] >= 10, 5,
                      np.where(df['COUNT_OF_LINES'] >= 5, 2, 0)))))
        
        # ✅ COUNTRY RISK (0-15 points)
        china_mask = df['ORIGIN'] == 'People\'s Republic of China'
        hk_mask = df['ORIGIN'] == 'Hong Kong Special Administrative Region'
        vietnam_mask = df['ORIGIN'] == 'Socialist Republic of Vietnam'
        mexico_mask = df['ORIGIN'] == 'United Mexican States'
        
        risk_score += np.where(china_mask, 15,
                      np.where(hk_mask, 12,
                      np.where(vietnam_mask, 8,
                      np.where(mexico_mask, 5, 0))))
        
        # ✅ PRODUCT RISK (0-10 points)
        clothing_mask = df['PRODUCT'] == 'Clothing'
        electronics_mask = df['PRODUCT'] == 'Electronics'
        jewelry_mask = df['PRODUCT'] == 'Jewelry'
        watch_mask = df['PRODUCT'] == 'Watch'
        
        risk_score += np.where(clothing_mask, 10,
                      np.where(electronics_mask, 8,
                      np.where(jewelry_mask, 8,
                      np.where(watch_mask, 6, 2))))
        
        # ✅ CONVEYANCE RISK (0-10 points)
        express_mask = df['CONVEYANCE'] == 'Express Consignment'
        air_mask = df['CONVEYANCE'] == 'Commercial Air'
        vessel_mask = df['CONVEYANCE'] == 'Commercial Vessel'
        
        risk_score += np.where(express_mask, 10,
                      np.where(air_mask, 6,
                      np.where(vessel_mask, 3, 0)))
        
        # ✅ COMBINATION BONUSES (0-10 points)
        # China + Express + Clothing = high risk combo
        high_risk_combo = china_mask & express_mask & clothing_mask
        risk_score += np.where(high_risk_combo, 10, 0)
        
        # High lines + High value combination
        complex_valuable = (df['COUNT_OF_LINES'] > 25) & (df['SUM_OF_MSRP'] > df['SUM_OF_MSRP'].quantile(0.75))
        risk_score += np.where(complex_valuable, 5, 0)
        
        # Ensure scores are within 0-100 range
        risk_score = np.clip(risk_score, 0, 100)
        
        df['risk_score_target'] = risk_score
        
        print(f"✓ Created continuous risk target:")
        print(f"   - Range: [{risk_score.min():.1f}, {risk_score.max():.1f}]")
        print(f"   - Mean: {risk_score.mean():.1f}")
        print(f"   - Std: {risk_score.std():.1f}")
        
        # Show distribution
        print(f"   - Low risk (0-40): {(risk_score < 40).sum()} records")
        print(f"   - Medium risk (40-60): {((risk_score >= 40) & (risk_score < 60)).sum()} records")
        print(f"   - High risk (60-80): {((risk_score >= 60) & (risk_score < 80)).sum()} records")
        print(f"   - Critical risk (80-100): {(risk_score >= 80).sum()} records")
        
        return df
        
    def train_model(self):
        """Train the IPR risk prediction model (legacy method for CSV)"""
        print("Training IPR Risk Prediction Model...")
        
        # Load processed data
        df = pd.read_csv('data/processed/ipr_enhanced.csv')
        print(f"✓ Loaded {df.shape[0]:,} records")
        
        # Use the new training method
        return self.train_model_with_data(df)
           
    def predict_risk(self, shipment_data):
        """Predict risk using trained Random Forest model REGRESSION (0-100 risk score)"""
        
        # Check if model is trained
        if self.model is None:
            print("⚠️ Model not trained, using rule-based fallback")
            return self._rule_based_prediction(shipment_data)
        
        try:
            # Step 1: Create feature vector from input data
            features = self._prepare_features_cleaned(shipment_data)
            
            # DEBUG: Print feature values for troubleshooting
            print(f"🔍 Debug - Key CLEANED features:")
            key_features = [
                'declared_value_category', 'declared_line_category', 
                'complexity_multiplier', 'extremely_complex', 'very_complex',
                'is_china', 'is_express', 'duty_rate', 'country_seizure_history', 'is_clothing'
            ]
            for feat in key_features:
                if feat in features.columns:
                    print(f"   - {feat}: {features[feat].iloc[0]}")
            
            # Step 2: Get prediction from regression model
            risk_score_raw = self.model.predict(features)[0]
            
            # Step 3: Ensure score is within 0-100 range
            risk_score = max(0, min(100, risk_score_raw))
            risk_probability = risk_score / 100
            
            print(f"🔍 Debug - Model prediction: {risk_score:.1f}% (raw: {risk_score_raw:.1f})")
            
            # Step 4: Categorize risk and generate response
            risk_category = self._categorize_risk(risk_probability)
            
            return {
                'risk_probability': risk_probability,
                'risk_score': int(risk_score),
                'risk_category': risk_category,
                'recommendations': self._generate_recommendations(shipment_data, risk_probability)
            }
        except Exception as e:
            print(f"❌ ML prediction failed: {e}")
            return self._rule_based_prediction(shipment_data)

    def _prepare_features_cleaned(self, shipment_data):
        """CLEANED: Convert raw shipment data into model-ready feature vector (NO declared_value_per_line)"""
        
        # Initialize feature dictionary
        feature_dict = {}
        
        # Get basic shipment info
        origin = shipment_data.get('origin', 'Unknown').lower()
        product = shipment_data.get('product', 'other')
        conveyance = shipment_data.get('conveyance', 'Unknown').lower()
        declared_value = max(shipment_data.get('declared_value', 1), 1)
        line_count = max(shipment_data.get('line_count', 1), 1)
        
        # ✅ BOUNDS CHECKING FOR REALISTIC VALUES
        if declared_value > 10_000_000:
            declared_value = 10_000_000
        if line_count > 1000:
            line_count = 1000
        
        # ✅ DECLARED VALUE AND LINE FEATURES
        feature_dict['declared_value_category'] = self._categorize_declared_value(declared_value)
        feature_dict['declared_line_category'] = self._categorize_declared_lines(line_count)
        
        # ✅ IMPROVED COMPLEXITY FEATURES - Add explicit complexity bonuses
        feature_dict['complexity_multiplier'] = np.where(
            line_count > 25,
            np.minimum(line_count / 25, 4.0),  # Up to 4x bonus
            1.0
        )
        
        # ✅ LINE COUNT RISK TIERS (more granular complexity indicators)
        feature_dict['extremely_complex'] = (line_count >= 100)
        feature_dict['very_complex'] = (line_count >= 50)
        
        # High value/complexity flags
        feature_dict['high_value_shipment'] = (declared_value > 50000)
        feature_dict['complex_shipment'] = (line_count > 25)
        
        print(f"🔍 CLEANED Features:")
        print(f"   Value: ${declared_value:,} → category {feature_dict['declared_value_category']}")
        print(f"   Lines: {line_count:,} → category {feature_dict['declared_line_category']}")
        print(f"   Complexity multiplier: {feature_dict['complexity_multiplier']:.2f}x")
        print(f"   Complex shipment: {feature_dict['complex_shipment']}")
        
        if line_count > 25:
            print(f"   ✅ HIGH COMPLEXITY DETECTED: {line_count} lines should INCREASE risk")
        
        # Get product info for tariff calculation
        product_info = self.hts_mapping.get(product, self.hts_mapping['other'])
        feature_dict['duty_rate'] = product_info['duty_rate']
        
        # ✅ FIXED: Tariff evasion calculation (no longer using problematic value_per_line)
        base_evasion_incentive = (
            feature_dict['duty_rate'] * np.log1p(declared_value) / 100
        )
        # Add complexity factor to tariff evasion
        feature_dict['tariff_evasion_incentive'] = base_evasion_incentive * feature_dict['complexity_multiplier']
        
        # ✅ GEOGRAPHIC RISK FLAGS
        feature_dict['is_china'] = 1 if 'china' in origin else 0
        feature_dict['is_hong_kong'] = 1 if 'hong kong' in origin else 0
        feature_dict['is_vietnam'] = 1 if 'vietnam' in origin else 0
        feature_dict['is_mexico'] = 1 if 'mexico' in origin else 0
        
        # ✅ CONVEYANCE RISK FLAGS
        feature_dict['is_express'] = 1 if 'express' in conveyance else 0
        feature_dict['is_air_freight'] = 1 if 'air' in conveyance else 0
        feature_dict['is_vessel'] = 1 if 'vessel' in conveyance else 0
        
        # ✅ PRODUCT RISK FLAGS
        feature_dict['high_duty_product'] = 1 if product_info['duty_rate'] > 10 else 0
        feature_dict['complex_product'] = 1 if product_info['complexity'] == 'High' else 0
        feature_dict['is_clothing'] = 1 if product == 'Clothing' else 0
        feature_dict['is_electronics'] = 1 if product == 'Electronics' else 0
        feature_dict['is_jewelry'] = 1 if product == 'Jewelry' else 0
        
        # ✅ HISTORICAL SEIZURE PATTERNS
        country_assessment = self.country_handler.calculate_composite_risk_score(shipment_data.get('origin', 'Unknown'))
        feature_dict['country_seizure_history'] = min(country_assessment['historical_data']['seizures'], 500)
        
        # Estimate product and conveyance history from constants
        feature_dict['product_seizure_history'] = self._get_product_risk_score(product)
        feature_dict['conveyance_seizure_history'] = self._get_conveyance_risk_score(shipment_data.get('conveyance', 'Unknown'))
        
        # Pattern flags (high seizure countries/products)
        feature_dict['repeat_country'] = 1 if feature_dict['country_seizure_history'] > 100 else 0
        feature_dict['repeat_product'] = 1 if feature_dict['product_seizure_history'] > 200 else 0
        
        # ✅ TEMPORAL FEATURES
        from datetime import datetime
        current_year = datetime.now().year
        fiscal_year = current_year if datetime.now().month >= 10 else current_year - 1
        feature_dict['FISCAL_YEAR'] = fiscal_year
        feature_dict['fiscal_year_trend'] = fiscal_year - 2019  # Years since baseline
        
        # ✅ INTERACTION FEATURES (risk combinations)
        feature_dict['china_clothing'] = feature_dict['is_china'] * feature_dict['is_clothing']
        feature_dict['china_electronics'] = feature_dict['is_china'] * feature_dict['is_electronics']
        feature_dict['express_clothing'] = feature_dict['is_express'] * feature_dict['is_clothing']
        feature_dict['high_duty_express'] = feature_dict['high_duty_product'] * feature_dict['is_express']
        
        # Handle categorical features using saved encoders
        categorical_mappings = {
            'PRODUCT': product,
            'CONVEYANCE': shipment_data.get('conveyance', 'Unknown'),
            'ORIGIN': shipment_data.get('origin', 'Unknown'),
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
        
        import pandas as pd
        feature_df = pd.DataFrame(feature_array, columns=self.feature_columns)
        
        return feature_df

    def _rule_based_prediction(self, shipment_data):
        """Rule-based prediction when ML model unavailable (REGRESSION VERSION)"""
        risk_score = 30  # Base risk of 30%
        
        # Get declared value and line count with bounds checking
        declared_value = max(shipment_data.get('declared_value', 1), 1)
        line_count = max(shipment_data.get('line_count', 1), 1)
        
        # Cap unrealistic values
        if declared_value > 10_000_000:
            declared_value = 10_000_000
        if line_count > 1000:
            line_count = 1000
        
        # Use USTR data if available
        if hasattr(self, 'country_handler'):
            country_assessment = self.country_handler.calculate_composite_risk_score(
                shipment_data.get('origin', 'Unknown')
            )
            country_risk = country_assessment['final_risk_score']
            risk_score = country_risk * 0.6 + risk_score * 0.4
        
        # Basic rule adjustments
        if shipment_data.get('origin', '').lower().find('china') != -1:
            risk_score += 15
        if shipment_data.get('conveyance', '').lower().find('express') != -1:
            risk_score += 10
        if shipment_data.get('product', '') == 'Clothing':
            risk_score += 8
            
        # ✅ IMPROVED VALUE ADJUSTMENTS
        if declared_value > 1_000_000:
            risk_score += 15
        elif declared_value > 100_000:
            risk_score += 10
        elif declared_value > 50_000:
            risk_score += 5
            
        # ✅ IMPROVED LINE COUNT ADJUSTMENTS (always increases risk)
        if line_count >= 100:
            line_risk = 20  # Extremely complex
        elif line_count >= 50:
            line_risk = 15  # Very complex
        elif line_count >= 25:
            line_risk = 10  # Complex
        elif line_count >= 10:
            line_risk = 5   # Medium complexity
        elif line_count >= 5:
            line_risk = 2   # Some complexity
        else:
            line_risk = 0   # No complexity bonus
        
        risk_score += line_risk
        
        # ✅ COMBINATION BONUS
        if line_count > 25 and declared_value > 50000:
            risk_score += 5  # Complex AND high value = extra suspicious
        
        print(f"🔍 Rule-based risk calculation:")
        print(f"   - Base risk: 30%")
        print(f"   - Value ${declared_value:,} adds: {(declared_value > 50000) * 5}%")
        print(f"   - Lines {line_count:,} adds: {line_risk}%")
        print(f"   - Final risk: {risk_score}%")
        
        risk_score = max(5, min(95, risk_score))
        risk_probability = risk_score / 100
        
        return {
            'risk_probability': risk_probability,
            'risk_score': int(risk_score),
            'risk_category': self._categorize_risk(risk_probability),
            'recommendations': self._generate_recommendations(shipment_data, risk_probability)
        }

    def _get_product_risk_score(self, product_name):
        """Get product risk score based on historical patterns"""
        from constants import PRODUCT_RISK_SCORES
        return PRODUCT_RISK_SCORES.get(product_name, 200)

    def _get_conveyance_risk_score(self, conveyance):
        """Get conveyance risk score based on historical patterns"""
        from constants import CONVEYANCE_RISK_SCORES
        return CONVEYANCE_RISK_SCORES.get(conveyance, 500)

    def _categorize_risk(self, probability):
        if probability >= 0.8: return 'CRITICAL'
        elif probability >= 0.6: return 'HIGH'
        elif probability >= 0.4: return 'MEDIUM'
        else: return 'LOW'
    
    def _generate_recommendations(self, shipment_data, risk_prob):
        recommendations = []
        
        # Get values for recommendation logic
        declared_value = max(shipment_data.get('declared_value', 1), 1)
        line_count = max(shipment_data.get('line_count', 1), 1)
        
        # Risk level recommendations
        if risk_prob >= 0.8:
            recommendations.append("🚨 CRITICAL RISK: Consider alternative sourcing")
        elif risk_prob >= 0.6:
            recommendations.append("⚠️ HIGH RISK: Enhanced documentation required")
        elif risk_prob >= 0.4:
            recommendations.append("⚠️ MEDIUM RISK: Additional verification recommended")
        else:
            recommendations.append("✅ LOW RISK: Standard compliance verification")
            
        # Value-specific recommendations
        if declared_value > 1_000_000:
            recommendations.append(f"💰 VERY HIGH VALUE: ${declared_value:,} requires special handling")
        elif declared_value > 100_000:
            recommendations.append(f"💰 HIGH VALUE: ${declared_value:,} - enhanced documentation required")
            
        # ✅ ENHANCED Line complexity recommendations
        if line_count > 100:
            recommendations.append(f"📦 EXTREMELY COMPLEX: {line_count:,} lines - CRITICAL inspection difficulty")
        elif line_count > 50:
            recommendations.append(f"📦 VERY COMPLEX: {line_count:,} lines - HIGH inspection difficulty")
        elif line_count > 25:
            recommendations.append(f"📦 COMPLEX SHIPMENT: {line_count} lines - detailed review required")
        elif line_count > 10:
            recommendations.append(f"📦 MODERATE COMPLEXITY: {line_count} lines - thorough verification needed")
            
        return recommendations

    def _categorize_declared_value(self, declared_value):
        """Categorize declared value into risk buckets (matches training data)"""
        if declared_value <= 1000:
            return 0  # Very low value
        elif declared_value <= 10000:
            return 1  # Low value
        elif declared_value <= 50000:
            return 2  # Medium value
        elif declared_value <= 100000:
            return 3  # High value
        else:
            return 4  # Very high value

    def _categorize_declared_lines(self, line_count):
        """Categorize line count into complexity buckets (matches training data)"""
        if line_count <= 5:
            return 0  # Very simple
        elif line_count <= 10:
            return 1  # Simple
        elif line_count <= 25:
            return 2  # Medium complexity
        elif line_count <= 50:
            return 3  # Complex
        else:
            return 4  # Very complex
    
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
    
    # Test prediction with different line counts
    test_shipments = [
        {
            'origin': 'People\'s Republic of China',
            'product': 'Electronics',
            'conveyance': 'Commercial Air',
            'declared_value': 5000,
            'line_count': 1,
            'name': 'China Electronics - 1 line'
        },
        {
            'origin': 'People\'s Republic of China',
            'product': 'Electronics', 
            'conveyance': 'Commercial Air',
            'declared_value': 5000,
            'line_count': 50,
            'name': 'China Electronics - 50 lines (should be HIGHER risk)'
        },
        {
            'origin': 'People\'s Republic of China',
            'product': 'Electronics',
            'conveyance': 'Commercial Air',
            'declared_value': 5000,
            'line_count': 100,
            'name': 'China Electronics - 100 lines (should be HIGHEST risk)'
        }
    ]
    
    print(f"\n🧪 Testing LINE COUNT EFFECT (REGRESSION):")
    for shipment in test_shipments:
        name = shipment.pop('name')
        result = predictor.predict_risk(shipment)
        print(f"   {name}: {result['risk_score']}% ({result['risk_category']})")
        
    print("\n✅ Expected behavior: More lines = Higher risk score")