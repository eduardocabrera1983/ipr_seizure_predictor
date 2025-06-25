# constants.py - Shared constants for IPR prediction
# HTS mapping with duty rates and complexity
HTS_MAPPING = {
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

# Risk combination weights
COUNTRY_WEIGHT = 0.4
ML_WEIGHT = 0.6

# Product risk scores
PRODUCT_RISK_SCORES = {
    'Clothing': 1200,
    'Footwear': 900,
    'Electronics': 800,
    'Purse/Wallets': 700,
    'Jewelry': 600,
    'Watch': 500,
    'Toys': 400,
    'Computer/Computer Parts': 300,
    'other': 200
}

# Conveyance risk scores  
CONVEYANCE_RISK_SCORES = {
    'Express Consignment': 2000,
    'Air Freight': 1500,
    'Ocean Freight': 800,
    'Truck': 300,
    'Rail': 200,
    'Unknown': 500
}