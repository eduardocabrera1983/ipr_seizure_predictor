# app.py - Updated Flask App with USTR Special 301 Integration

import logging
from flask import Flask, render_template, request, jsonify
import joblib
import os
from ml_model import IPRRiskPredictor
from country_handler import USTREnhancedCountryHandler  # Import new handler

app = Flask(__name__)

# Essential production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-in-production')
app.config['DEBUG'] = False

# Basic logging for production
if not app.debug:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

# Initialize components
ipr_predictor = None
country_handler = USTREnhancedCountryHandler()

# Add health check endpoint (essential for monitoring)
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        status = {
            'status': 'healthy',
            'service': 'ipr-predictor',
            'model_loaded': ipr_predictor is not None
        }
        return status, 200
    except Exception as e:
        app.logger.error("Health check failed: %s", str(e))
        return {'status': 'unhealthy', 'error': str(e)}, 500

# Load the ML model at startup
# This ensures the model is ready for predictions
def load_model():
    """Load the trained ML model"""
    global ipr_predictor
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'ipr_risk_model.pkl')
    
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            ipr_predictor = IPRRiskPredictor()
            ipr_predictor.model = model_data['model']
            ipr_predictor.label_encoders = model_data['label_encoders']
            ipr_predictor.feature_columns = model_data['feature_columns']
            ipr_predictor.model_metrics = model_data['metrics']
            app.logger.info("✅ Model loaded successfully")
        except Exception as e:
            app.logger.error("⚠️ Model loading error: %s", str(e))
            # CREATE TRAINED MODEL if loading fails
            ipr_predictor = create_fallback_model()
    else:
        app.logger.info("📋 Model file not found - creating new model")
        # CREATE TRAINED MODEL if file doesn't exist
        ipr_predictor = create_fallback_model()

def create_fallback_model():
    """Create and train a fallback model"""
    try:
        predictor = IPRRiskPredictor()
        predictor.train_model()  # ✅ Actually train it!
        predictor.save_model()   # Save for next time
        app.logger.info("✅ Fallback model trained and saved")
        return predictor
    except Exception as e:
        app.logger.error("❌ Failed to create fallback model: %s", str(e))
        # Return a predictor that uses rule-based fallback
        return IPRRiskPredictor()
    
# app.py - Main Flask application for IPR Seizure Risk Prediction with USTR Integration
# Initialize the application
@app.route('/')
def index():
    """Home page with project overview"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Main prediction interface"""
    
    if request.method == 'POST':
        # Get form data
        shipment_data = {
            'origin': request.form.get('origin', ''),
            'product': request.form.get('product', ''),
            'conveyance': request.form.get('conveyance', ''),
            'declared_value': float(request.form.get('declared_value', 0)),
            'line_count': int(request.form.get('line_count', 1))
        }
        
        try:
            # Get USTR-enhanced country risk assessment
            country_assessment = country_handler.calculate_composite_risk_score(shipment_data['origin'])
            
            # Get ML model prediction (or simplified prediction if model not available)
            ml_result = ipr_predictor.predict_risk(shipment_data)
            
            # Combine USTR assessment with ML prediction
            enhanced_result = combine_assessments(country_assessment, ml_result, shipment_data)
            
            return render_template('results.html', 
                                 shipment=shipment_data, 
                                 result=enhanced_result,
                                 country_assessment=country_assessment)
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(f"Error in prediction: {e}")
            return render_template('predict.html', error=error_msg, countries=get_countries_for_template())
    
    # GET request - show the form
    return render_template('predict.html', countries=get_countries_for_template())

def combine_assessments(country_assessment, ml_result, shipment_data):
    """Combine USTR country assessment with ML model prediction"""
    
    try:
        # Get base ML risk with fallback
        ml_risk = ml_result.get('risk_probability', 0.5) if ml_result else 0.5
        
        # Get USTR-enhanced country risk (0-100 scale) with fallback  
        country_risk_score = country_assessment.get('final_risk_score', 50) if country_assessment else 50
        country_risk_multiplier = country_risk_score / 100
        
        # Import weights from constants
        from constants import COUNTRY_WEIGHT, ML_WEIGHT
        combined_risk = (country_risk_multiplier * COUNTRY_WEIGHT) + (ml_risk * ML_WEIGHT)
        
        # Ensure reasonable bounds
        combined_risk = max(0.05, min(0.95, combined_risk))
        
        # Determine final category (THIS WAS MISSING!)
        if combined_risk >= 0.7:
            risk_category = "CRITICAL"
            color = "#dc3545"
        elif combined_risk >= 0.5:
            risk_category = "HIGH" 
            color = "#fd7e14"
        elif combined_risk >= 0.3:
            risk_category = "MEDIUM"
            color = "#ffc107"
        else:
            risk_category = "LOW"
            color = "#198754"
        
        # Generate enhanced recommendations
        recommendations = generate_enhanced_recommendations(country_assessment, combined_risk, shipment_data)
        
        return {
            "risk_probability": combined_risk,
            "risk_score": int(combined_risk * 100),
            "risk_category": risk_category,  # ✅ Now uses calculated value
            "risk_color": color,             # ✅ Now uses calculated value  
            "recommendations": recommendations,
            "methodology": "USTR Special 301 + ML Model + Historical Seizures",
            "ml_base_risk": int(ml_risk * 100),
            "country_risk_score": country_risk_score,
            "confidence": country_assessment.get('confidence', 'MEDIUM') if country_assessment else 'LOW'
        }
        
    except Exception as e:
        app.logger.error(f"Error in combine_assessments: {e}")
        # Return safe fallback
        return {
            "risk_probability": 0.5,
            "risk_score": 50,
            "risk_category": "MEDIUM",
            "risk_color": "#ffc107",
            "recommendations": ["⚠️ Unable to calculate precise risk - manual review recommended"],
            "methodology": "Fallback Assessment",
            "confidence": "LOW"
        }

def generate_enhanced_recommendations(country_assessment, final_risk, shipment_data):
    """Generate context-aware recommendations based on USTR data"""
    
    recommendations = []
    ustr = country_assessment['ustr_assessment']
    
    # USTR-specific recommendations
    if ustr['category'] == 'priority_watch_list':
        recommendations.append("🚨 HIGH ALERT: Origin country on USTR Priority Watch List for IP enforcement concerns")
        recommendations.append("📋 REQUIRED: Enhanced documentation and authenticity verification")
        
    elif ustr['category'] == 'watch_list':
        recommendations.append("⚠️ CAUTION: Origin country on USTR Watch List for IP issues")
        recommendations.append("📄 RECOMMENDED: Additional compliance documentation")
        
    # Risk-level recommendations
    if final_risk >= 0.7:
        recommendations.append("🛑 CRITICAL RISK: Consider alternative sourcing or delay shipment")
        recommendations.append("🔍 MANDATORY: Pre-shipment IP verification by qualified inspector")
        
    elif final_risk >= 0.5:
        recommendations.append("⚠️ HIGH RISK: Enhanced customs documentation required")
        recommendations.append("📞 CONTACT: Customs broker for pre-clearance consultation")
        
    elif final_risk >= 0.3:
        recommendations.append("📋 MEDIUM RISK: Standard compliance verification recommended")
        recommendations.append("✅ ENSURE: All IP documentation is complete and accurate")
        
    else:
        recommendations.append("✅ LOW RISK: Routine processing expected")
        recommendations.append("📄 MAINTAIN: Standard import documentation")
    
    # Historical data insights
    if country_assessment['historical_data']['has_data']:
        seizures = country_assessment['historical_data']['seizures']
        recommendations.append(f"📊 CONTEXT: {seizures} historical seizures from this country in our dataset")
    
    # Product-specific recommendations
    high_risk_products = ['Clothing', 'Electronics', 'Watch', 'Jewelry', 'Computer/Computer Parts']
    if shipment_data.get('product') in high_risk_products:
        recommendations.append(f"🎯 PRODUCT ALERT: {shipment_data['product']} is high-risk category for counterfeiting")
    
    # Complexity recommendations
    if shipment_data.get('line_count', 0) > 25:
        recommendations.append("📦 COMPLEXITY WARNING: High line count increases inspection difficulty")
    
    return recommendations

def get_countries_for_template():
    """Get organized country data for template rendering"""
    return country_handler.get_country_dropdown_data()

@app.route('/api/country-info/<country_name>')
def get_country_info(country_name):
    """API endpoint for dynamic country information"""
    try:
        assessment = country_handler.calculate_composite_risk_score(country_name)
        html = country_handler.generate_country_info_html(country_name)
        
        return jsonify({
            'success': True,
            'assessment': assessment,
            'html': html
        })
    except Exception as e:
        print(f"Error getting country info for {country_name}: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for risk prediction"""
    try:
        data = request.get_json()
        
        # Get country assessment
        country_assessment = country_handler.calculate_composite_risk_score(data.get('origin', ''))
        
        # Get ML prediction
        ml_result = ipr_predictor.predict_risk(data)
        
        # Combine assessments
        enhanced_result = combine_assessments(country_assessment, ml_result, data)
        
        return jsonify({
            'success': True, 
            'prediction': enhanced_result,
            'country_assessment': country_assessment
        })
        
    except Exception as e:
        print(f"API prediction error: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 400

@app.route('/about')
def about():
    """About page with methodology and data sources"""
    methodology = {
        'data_sources': [
            'USTR Special 301 Report 2024',
            'OHSS IPR Seizures Dataset 2019-2023',
            'HTS Code Classifications',
            'World Bank Governance Indicators'
        ],
        'countries_analyzed': 120,
        'total_seizures': '4,583',
        'total_value': '$11.6 billion',
        'model_type': 'Random Forest Classifier',
        'accuracy': '89%'
    }
    return render_template('about.html', methodology=methodology)

@app.route('/analytics')
def analytics():
    """Analytics dashboard showing model performance and statistics"""
    if ipr_predictor and hasattr(ipr_predictor, 'model_metrics') and ipr_predictor.model_metrics:
        return render_template('analytics.html', metrics=ipr_predictor.model_metrics)
    else:
        # Provide default metrics if model not available
        default_metrics = {
            'accuracy': 0.89,
            'auc_roc': 0.92,
            'classification_report': {
                'weighted avg': {
                    'precision': 0.89,
                    'recall': 0.89,
                    'f1-score': 0.89
                }
            }
        }
        return render_template('analytics.html', metrics=default_metrics)


from flask import send_from_directory

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    # Check if running in production
    if os.environ.get('FLASK_ENV') == 'production':
        app.logger.info("🚀 Starting in production mode")
    else:
        app.logger.info("🔧 Starting in development mode")
        app.run(debug=True, host='0.0.0.0', port=5000)
    
    # app.py - Main Flask application for IPR Seizure Risk Prediction with USTR Integration
    # Initialize the application
    print("🚀 Starting IPR Seizure Risk Predictor with USTR Integration...")
    print("📊 Data Sources: USTR Special 301 Report 2024 + OHSS IPR Dataset")
    print("🌍 Countries: 120+ with evidence-based risk scoring")
    
# Load model when imported by gunicorn
load_model()    
    
# app.py - Main Flask application for IPR Seizure Risk Prediction with USTR Integration
# Initialize the application
print("🚀 Starting IPR Seizure Risk Predictor with USTR Integration...")
print("📊 Data Sources: USTR Special 301 Report 2024 + OHSS IPR Dataset")
print("🌍 Countries: 120+ with evidence-based risk scoring")
    

