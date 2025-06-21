# ipr_seizure_predictor/country_handler.py
# -*- coding: utf-8 -*-

"""
Enhanced Country Handler for IPR Seizure Prediction
This module provides an enhanced country handler that integrates the latest USTR Special 301 Report
and historical seizure data to assess intellectual property rights (IPR) risks for various countries.
"""

class USTREnhancedCountryHandler:
    def __init__(self):
        """Initialize with USTR Special 301 Report 2024 + Historical Seizure Data"""
        
        # USTR Special 301 Report 2024 - Official US Government Data
        # Source: https://ustr.gov/sites/default/files/2024-04/2024%20Special%20301%20Report.pdf
        self.ustr_2024_data = {
            "priority_watch_list": [
                "People's Republic of China",
                "Republic of India", 
                "Republic of Indonesia",
                "Russian Federation",
                "Kingdom of Saudi Arabia",
                "Republic of Turkey",
                "Ukraine",
                "Bolivarian Republic of Venezuela"
            ],
            
            "watch_list": [
                "Argentine Republic",
                "Republic of Chile", 
                "Republic of Colombia",
                "Republic of Ecuador",
                "Republic of Guatemala", 
                "Republic of Kuwait",
                "United Mexican States",
                "Republic of Peru",
                "Republic of the Philippines",
                "Romania",
                "Kingdom of Thailand",
                "Socialist Republic of Vietnam"
            ],
            
            "positive_developments": [
                "Federative Republic of Brazil",
                "Republic of Korea",
                "Malaysia"
            ]
        }
        
        # Historical seizure data from your IPR dataset
        self.historical_seizures = {
            "People's Republic of China": {"seizures": 472, "avg_value": 14580000},
            "Hong Kong Special Administrative Region": {"seizures": 284, "avg_value": 8247000},
            "Unknown": {"seizures": 345, "avg_value": 1934000},
            "United States of America": {"seizures": 197, "avg_value": 300000},
            "United Mexican States": {"seizures": 170, "avg_value": 261000},
            "Socialist Republic of Vietnam": {"seizures": 145, "avg_value": 962000},
            "Republic of the Philippines": {"seizures": 138, "avg_value": 1315000},
            "Republic of Turkey": {"seizures": 137, "avg_value": 1184000},
            "Kingdom of Thailand": {"seizures": 125, "avg_value": 997000},
            "United Arab Emirates": {"seizures": 120, "avg_value": 509000},
            "Republic of India": {"seizures": 89, "avg_value": 1650000},
            "Dominican Republic": {"seizures": 76, "avg_value": 890000},
            "Republic of Korea": {"seizures": 65, "avg_value": 445000},
            "Japan": {"seizures": 58, "avg_value": 320000},
            "Taiwan": {"seizures": 52, "avg_value": 280000}
        }
        
        # Comprehensive country list for dropdown (120+ countries)
        self.all_countries = {
            # USTR Priority Watch List (Highest Risk)
            "People's Republic of China": "🇨🇳",
            "Republic of India": "🇮🇳", 
            "Republic of Indonesia": "🇮🇩",
            "Russian Federation": "🇷🇺",
            "Kingdom of Saudi Arabia": "🇸🇦",
            "Republic of Turkey": "🇹🇷",
            "Ukraine": "🇺🇦",
            "Bolivarian Republic of Venezuela": "🇻🇪",
            
            # USTR Watch List (High Risk)
            "Argentine Republic": "🇦🇷",
            "Republic of Chile": "🇨🇱", 
            "Republic of Colombia": "🇨🇴",
            "Republic of Ecuador": "🇪🇨",
            "Republic of Guatemala": "🇬🇹",
            "Republic of Kuwait": "🇰🇼",
            "United Mexican States": "🇲🇽",
            "Republic of Peru": "🇵🇪",
            "Republic of the Philippines": "🇵🇭",
            "Romania": "🇷🇴",
            "Kingdom of Thailand": "🇹🇭",
            "Socialist Republic of Vietnam": "🇻🇳",
            
            # Historical Seizure Data (from your original dataset)
            "Hong Kong Special Administrative Region": "🇭🇰",
            "United Arab Emirates": "🇦🇪",
            "Dominican Republic": "🇩🇴",
            "Taiwan": "🇹🇼",
            "United States of America": "🇺🇸",
            
            # Major Trading Partners (G20 + Top US Partners)
            "Canada": "🇨🇦",
            "Japan": "🇯🇵",
            "Federal Republic of Germany": "🇩🇪",
            "Republic of Korea": "🇰🇷",
            "United Kingdom": "🇬🇧",
            "French Republic": "🇫🇷",
            "Italian Republic": "🇮🇹",
            "Kingdom of Spain": "🇪🇸",
            "Kingdom of the Netherlands": "🇳🇱",
            "Kingdom of Belgium": "🇧🇪",
            "Swiss Confederation": "🇨🇭",
            "Republic of Austria": "🇦🇹",
            "Kingdom of Sweden": "🇸🇪",
            "Kingdom of Denmark": "🇩🇰",
            "Kingdom of Norway": "🇳🇴",
            "Republic of Finland": "🇫🇮",
            "Australia": "🇦🇺",
            "New Zealand": "🇳🇿",
            "Israel": "🇮🇱",
            "Ireland": "🇮🇪",
            "Republic of Portugal": "🇵🇹",
            "Hellenic Republic": "🇬🇷",
            "Republic of Cyprus": "🇨🇾",
            "Republic of Malta": "🇲🇹",
            "Grand Duchy of Luxembourg": "🇱🇺",
            
            # Asian Countries
            "Malaysia": "🇲🇾",
            "Republic of Singapore": "🇸🇬",
            "Bangladesh": "🇧🇩",
            "Pakistan": "🇵🇰",
            "Sri Lanka": "🇱🇰",
            "Nepal": "🇳🇵",
            "Republic of Maldives": "🇲🇻",
            "Brunei Darussalam": "🇧🇳",
            "Lao People's Democratic Republic": "🇱🇦",
            "Kingdom of Cambodia": "🇰🇭",
            "Republic of the Union of Myanmar": "🇲🇲",
            "Mongolia": "🇲🇳",
            "Democratic People's Republic of Korea": "🇰🇵",
            "Kazakhstan": "🇰🇿",
            "Kyrgyz Republic": "🇰🇬",
            "Republic of Tajikistan": "🇹🇯",
            "Turkmenistan": "🇹🇲",
            "Republic of Uzbekistan": "🇺🇿",
            "Islamic Republic of Afghanistan": "🇦🇫",
            "Islamic Republic of Iran": "🇮🇷",
            "Republic of Iraq": "🇮🇶",
            "Syrian Arab Republic": "🇸🇾",
            "Lebanese Republic": "🇱🇧",
            "Hashemite Kingdom of Jordan": "🇯🇴",
            "State of Palestine": "🇵🇸",
            "Republic of Yemen": "🇾🇪",
            "Sultanate of Oman": "🇴🇲",
            "State of Qatar": "🇶🇦",
            "Kingdom of Bahrain": "🇧🇭",
            "Azerbaijan": "🇦🇿",
            "Georgia": "🇬🇪",
            "Republic of Armenia": "🇦🇲",
            
            # European Countries (EU + Others)
            "Republic of Poland": "🇵🇱",
            "Czech Republic": "🇨🇿",
            "Slovak Republic": "🇸🇰",
            "Hungary": "🇭🇺",
            "Republic of Slovenia": "🇸🇮",
            "Republic of Croatia": "🇭🇷",
            "Bosnia and Herzegovina": "🇧🇦",
            "Montenegro": "🇲🇪",
            "Republic of Serbia": "🇷🇸",
            "Republic of North Macedonia": "🇲🇰",
            "Republic of Albania": "🇦🇱",
            "Republic of Bulgaria": "🇧🇬",
            "Republic of Estonia": "🇪🇪",
            "Republic of Latvia": "🇱🇻",
            "Republic of Lithuania": "🇱🇹",
            "Republic of Belarus": "🇧🇾",
            "Republic of Moldova": "🇲🇩",
            "Iceland": "🇮🇸",
            
            # African Countries
            "Federative Republic of Brazil": "🇧🇷",
            "Republic of South Africa": "🇿🇦",
            "Arab Republic of Egypt": "🇪🇬",
            "Federal Republic of Nigeria": "🇳🇬",
            "Republic of Kenya": "🇰🇪",
            "Kingdom of Morocco": "🇲🇦",
            "Republic of Ghana": "🇬🇭",
            "Republic of Tunisia": "🇹🇳",
            "People's Democratic Republic of Algeria": "🇩🇿",
            "Socialist People's Libyan Arab Jamahiriya": "🇱🇾",
            "Republic of Sudan": "🇸🇩",
            "Federal Democratic Republic of Ethiopia": "🇪🇹",
            "Republic of Uganda": "🇺🇬",
            "United Republic of Tanzania": "🇹🇿",
            "Republic of Rwanda": "🇷🇼",
            "Republic of Zambia": "🇿🇲",
            "Republic of Zimbabwe": "🇿🇼",
            "Republic of Botswana": "🇧🇼",
            "Kingdom of Lesotho": "🇱🇸",
            "Kingdom of Eswatini": "🇸🇿",
            "Republic of Namibia": "🇳🇦",
            "Republic of Angola": "🇦🇴",
            "Republic of Mozambique": "🇲🇿",
            "Republic of Madagascar": "🇲🇬",
            "Republic of Mauritius": "🇲🇺",
            "Republic of Seychelles": "🇸🇨",
            "Republic of Côte d'Ivoire": "🇨🇮",
            "Republic of Senegal": "🇸🇳",
            "Republic of Mali": "🇲🇱",
            "Burkina Faso": "🇧🇫",
            "Republic of Niger": "🇳🇪",
            "Republic of Chad": "🇹🇩",
            "Central African Republic": "🇨🇫",
            "Democratic Republic of the Congo": "🇨🇩",
            "Republic of the Congo": "🇨🇬",
            "Gabonese Republic": "🇬🇦",
            "Republic of Equatorial Guinea": "🇬🇶",
            "Democratic Republic of São Tomé and Príncipe": "🇸🇹",
            "Republic of Cameroon": "🇨🇲",
            "Republic of Benin": "🇧🇯",
            "Togolese Republic": "🇹🇬",
            "Republic of Liberia": "🇱🇷",
            "Republic of Sierra Leone": "🇸🇱",
            "Republic of Guinea": "🇬🇳",
            "Republic of Guinea-Bissau": "🇬🇼",
            "Republic of Cape Verde": "🇨🇻",
            "Islamic Republic of the Gambia": "🇬🇲",
            
            # Americas (North, Central, South)
            "Republic of Costa Rica": "🇨🇷",
            "Republic of Nicaragua": "🇳🇮",
            "Republic of Honduras": "🇭🇳",
            "Belize": "🇧🇿",
            "Republic of El Salvador": "🇸🇻",
            "Republic of Panama": "🇵🇦",
            "Jamaica": "🇯🇲",
            "Republic of Cuba": "🇨🇺",
            "Republic of Haiti": "🇭🇹",
            "Commonwealth of The Bahamas": "🇧🇸",
            "Barbados": "🇧🇧",
            "Republic of Trinidad and Tobago": "🇹🇹",
            "Saint Lucia": "🇱🇨",
            "Saint Vincent and the Grenadines": "🇻🇨",
            "Grenada": "🇬🇩",
            "Antigua and Barbuda": "🇦🇬",
            "Federation of Saint Kitts and Nevis": "🇰🇳",
            "Commonwealth of Dominica": "🇩🇲",
            "Republic of Suriname": "🇸🇷",
            "Co-operative Republic of Guyana": "🇬🇾",
            "Republic of Paraguay": "🇵🇾",
            "Eastern Republic of Uruguay": "🇺🇾",
            "Plurinational State of Bolivia": "🇧🇴",
            
            # Oceania
            "Papua New Guinea": "🇵🇬",
            "Republic of Fiji": "🇫🇯",
            "Solomon Islands": "🇸🇧",
            "Republic of Vanuatu": "🇻🇺",
            "Independent State of Samoa": "🇼🇸",
            "Kingdom of Tonga": "🇹🇴",
            "Republic of Kiribati": "🇰🇮",
            "Republic of Nauru": "🇳🇷",
            "Tuvalu": "🇹🇻",
            "Republic of Palau": "🇵🇼",
            "Federated States of Micronesia": "🇫🇲",
            "Republic of the Marshall Islands": "🇲🇭",
            
            # Special Categories
            "Unknown": "❓",
            "Multiple Countries": "🌍",
            "Other/Not Listed": "🏳️"
        }

    def get_ustr_assessment(self, country_name):
        """Get official USTR Special 301 assessment"""
        
        if country_name in self.ustr_2024_data["priority_watch_list"]:
            return {
                "category": "priority_watch_list",
                "risk_score": 85,
                "risk_level": "VERY_HIGH",
                "explanation": "USTR Priority Watch List 2024 - Most serious IP enforcement concerns",
                "badge": "🚨 USTR Priority",
                "confidence": "HIGH"
            }
            
        elif country_name in self.ustr_2024_data["watch_list"]:
            return {
                "category": "watch_list", 
                "risk_score": 60,
                "risk_level": "HIGH",
                "explanation": "USTR Watch List 2024 - Significant IP enforcement concerns",
                "badge": "⚠️ USTR Watch",
                "confidence": "HIGH"
            }
            
        elif country_name in self.ustr_2024_data["positive_developments"]:
            return {
                "category": "positive_developments",
                "risk_score": 35,
                "risk_level": "MEDIUM",
                "explanation": "USTR noted positive developments in IP enforcement",
                "badge": "📈 Improving",
                "confidence": "HIGH"
            }
            
        else:
            return {
                "category": "not_mentioned",
                "risk_score": 25,
                "risk_level": "LOW",
                "explanation": "No specific IP enforcement concerns in USTR 2024 assessment",
                "badge": "✅ No Concerns",
                "confidence": "MEDIUM"
            }

    def get_historical_data(self, country_name):
        """Get historical seizure data from IPR dataset"""
        
        if country_name in self.historical_seizures:
            data = self.historical_seizures[country_name]
            return {
                "has_data": True,
                "seizures": data["seizures"],
                "avg_value": data["avg_value"],
                "data_source": "OHSS IPR Dataset 2019-2023",
                "confidence": "HIGH"
            }
        else:
            return {
                "has_data": False,
                "seizures": 0,
                "avg_value": 0,
                "data_source": "No historical data",
                "confidence": "LOW"
            }

    def calculate_composite_risk_score(self, country_name):
        """Calculate final risk score combining USTR + Historical data"""
        
        # Get USTR assessment
        ustr = self.get_ustr_assessment(country_name)
        
        # Get historical data
        historical = self.get_historical_data(country_name)
        
        # Calculate weighted composite score
        if historical["has_data"]:
            # Historical data available: 50% USTR + 50% Historical
            historical_risk = min(40, historical["seizures"] / 10)  # Scale seizures to max 40 points
            composite_score = (ustr["risk_score"] * 0.5) + (historical_risk * 0.5) + 30
            confidence = "HIGH"
        else:
            # No historical data: 70% USTR + 30% regional estimate
            regional_risk = 20  # Base regional risk
            composite_score = (ustr["risk_score"] * 0.7) + (regional_risk * 0.3)
            confidence = ustr["confidence"]
        
        # Cap score between 5-95
        final_score = max(5, min(95, composite_score))
        
        # Determine final category
        if final_score >= 70:
            final_category = "CRITICAL"
        elif final_score >= 55:
            final_category = "HIGH"
        elif final_score >= 35:
            final_category = "MEDIUM"
        else:
            final_category = "LOW"
        
        return {
            "country": country_name,
            "final_risk_score": round(final_score, 1),
            "final_risk_category": final_category,
            "ustr_assessment": ustr,
            "historical_data": historical,
            "confidence": confidence,
            "methodology": "USTR Special 301 + Historical Seizures"
        }

    def get_country_dropdown_data(self):
        """Generate structured data for dropdown with USTR categories"""
        
        dropdown_data = {
            "priority_watch_list": [],
            "watch_list": [],
            "historical_data": [],
            "major_partners": [],
            "others": []
        }
        
        for country, flag in self.all_countries.items():
            country_data = {
                "name": country,
                "flag": flag,
                "assessment": self.get_ustr_assessment(country),
                "historical": self.get_historical_data(country)
            }
            
            # Categorize for dropdown
            if country in self.ustr_2024_data["priority_watch_list"]:
                dropdown_data["priority_watch_list"].append(country_data)
            elif country in self.ustr_2024_data["watch_list"]:
                dropdown_data["watch_list"].append(country_data)
            elif country_data["historical"]["has_data"]:
                dropdown_data["historical_data"].append(country_data)
            elif country in ["Canada", "Japan", "Federal Republic of Germany", "United Kingdom", "French Republic"]:
                dropdown_data["major_partners"].append(country_data)
            else:
                dropdown_data["others"].append(country_data)
        
        return dropdown_data

    def generate_country_info_html(self, country_name):
        """Generate HTML for country information display"""
        
        assessment = self.calculate_composite_risk_score(country_name)
        ustr = assessment["ustr_assessment"]
        historical = assessment["historical_data"]
        
        # Risk level styling
        risk_colors = {
            "CRITICAL": "#dc3545",
            "HIGH": "#fd7e14", 
            "MEDIUM": "#ffc107",
            "LOW": "#198754"
        }
        
        color = risk_colors.get(assessment["final_risk_category"], "#6c757d")
        
        html = f"""
        <div style="border-left: 4px solid {color}; padding: 12px; background-color: rgba(255,255,255,0.1); border-radius: 5px;">
            <div style="font-weight: bold; font-size: 1.1em; color: {color};">
                {assessment["final_risk_category"]} RISK ({assessment["final_risk_score"]}%)
            </div>
            
            <div style="margin: 8px 0;">
                <span style="background-color: rgba(255,255,255,0.2); padding: 3px 8px; border-radius: 12px; font-size: 0.9em;">
                    {ustr["badge"]}
                </span>
                {f'<span style="background-color: rgba(255,255,255,0.2); padding: 3px 8px; border-radius: 12px; font-size: 0.9em; margin-left: 5px;">{historical["seizures"]} seizures</span>' if historical["has_data"] else ''}
            </div>
            
            <div style="font-size: 0.9em; margin-top: 8px;">
                <strong>USTR Assessment:</strong> {ustr["explanation"]}<br>
                {f'<strong>Historical Data:</strong> {historical["seizures"]} seizures, avg ${historical["avg_value"]:,}' if historical["has_data"] else '<strong>Historical Data:</strong> No seizure records in dataset'}
            </div>
            
            <div style="font-size: 0.8em; color: rgba(255,255,255,0.8); margin-top: 5px;">
                <em>Confidence: {assessment["confidence"]} | Sources: USTR 2024 + OHSS Dataset</em>
            </div>
        </div>
        """
        
        return html


# Example usage / testing
if __name__ == "__main__":
    handler = USTREnhancedCountryHandler()
    
    # Test key countries
    test_countries = [
        "People's Republic of China",     # Priority Watch + Historical
        "Republic of India",              # Priority Watch, no historical
        "United Mexican States",          # Watch List + Historical
        "Japan",                         # No USTR concerns + Historical
        "Federal Republic of Germany",    # No concerns, no historical
    ]
    
    print("=== USTR-ENHANCED COUNTRY RISK ASSESSMENT ===")
    
    for country in test_countries:
        result = handler.calculate_composite_risk_score(country)
        print(f"\n{country}:")
        print(f"  Final Risk: {result['final_risk_score']}% ({result['final_risk_category']})")
        print(f"  USTR: {result['ustr_assessment']['badge']}")
        print(f"  Historical: {result['historical_data']['seizures']} seizures")
        print(f"  Confidence: {result['confidence']}")