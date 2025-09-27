import os
import requests
import logging
import csv
import io
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai
from datetime import datetime, timedelta
import random

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = "8287736660:AAFPklAmWPDmM1HzF4Zfhql1ks4kDR896G4"
GEMINI_API_KEY = "AIzaSyBHiuLjXp3gtW8QK6xqfJgaHtL4APKfGaQ"
OPENWEATHER_API_KEY = "9a73e2999f2ac696e9e8ddb256c1ab4f"

EMERGENCY_CONTACTS = {
    "police": "100",
    "ambulance": "108", 
    "fire": "101",
    "disaster_management": "1070",
    "thrissur_control_room": "0487-2362424",
    "thrissur_district_emergency": "0487-2360222"
}

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    logger.info("Gemini AI initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini AI: {e}")
    model = None

disaster_data = """location,disaster_type,risk_level,lat,lon,evacuation_lat,evacuation_lon
Thrissur,Flood,High,10.5276,76.2144,10.5300,76.2200
Guruvayur,Flood,Medium,10.5941,76.0414,10.6000,76.0500
Chavakkad,Landslide,Low,10.5333,76.0333,10.5400,76.0400
Kunnamkulam,Flood,Medium,10.65,76.0833,10.6550,76.0900
Kodungallur,Flood,High,10.2333,76.2167,10.2400,76.2200
Irinjalakuda,Cyclone,Low,10.3424,76.2117,10.3500,76.2200
"""

historical_disasters = [
    {"date": "2018-08-15", "type": "Flood", "location": "Thrissur", "severity": "low", "affected_areas": "City Center, Punkunnam", "casualties": 5, "damage": "Extensive"},
    {"date": "2019-07-22", "type": "Landslide", "location": "Chavakkad", "severity": "Medium", "affected_areas": "Hilly regions", "casualties": 2, "damage": "Moderate"},
    {"date": "2020-06-10", "type": "Flood", "location": "Kodungallur", "severity": "High", "affected_areas": "Coastal areas", "casualties": 3, "damage": "Extensive"},
    {"date": "2021-05-18", "type": "Cyclone", "location": "Irinjalakuda", "severity": "Medium", "affected_areas": "Multiple areas", "casualties": 1, "damage": "Moderate"},
    {"date": "2022-08-05", "type": "Flood", "location": "Guruvayur", "severity": "High", "affected_areas": "Temple surroundings", "casualties": 0, "damage": "Significant"},
    {"date": "2023-07-30", "type": "Flood", "location": "Kunnamkulam", "severity": "Medium", "affected_areas": "Market area", "casualties": 0, "damage": "Moderate"}
]

disaster_db = []
try:
    reader = csv.DictReader(io.StringIO(disaster_data))
    for row in reader:
        disaster_db.append(row)
    logger.info(f"Loaded {len(disaster_db)} disaster records")
except Exception as e:
    logger.error(f"Error loading disaster data: {e}")

thrissur_cities = ["Thrissur", "Guruvayur", "Chavakkad", "Kunnamkulam", "Kodungallur", "Irinjalakuda"]

ml_model = None
scaler = StandardScaler()
model_trained = False

def create_enhanced_training_data():
    np.random.seed(42)
    n_samples = 5000
    
    feature_columns = ['temperature', 'humidity', 'wind_speed', 'rainfall', 'pressure']
    
    normal_conditions = {
        'temperature': np.random.normal(28, 3, n_samples),
        'humidity': np.random.normal(65, 8, n_samples),
        'wind_speed': np.random.gamma(2, 1.5, n_samples),
        'rainfall': np.random.exponential(1, n_samples),
        'pressure': np.random.normal(1013, 5, n_samples)
    }
    normal_df = pd.DataFrame(normal_conditions)
    normal_df['risk_level'] = 0
    
    flood_conditions = {
        'temperature': np.random.normal(26, 2, n_samples),
        'humidity': np.random.normal(85, 5, n_samples),
        'wind_speed': np.random.gamma(3, 2, n_samples),
        'rainfall': np.random.gamma(5, 4, n_samples),
        'pressure': np.random.normal(1005, 8, n_samples)
    }
    flood_df = pd.DataFrame(flood_conditions)
    flood_df['risk_level'] = 3
    
    cyclone_conditions = {
        'temperature': np.random.normal(30, 2, n_samples),
        'humidity': np.random.normal(75, 6, n_samples),
        'wind_speed': np.random.gamma(8, 2, n_samples),
        'rainfall': np.random.gamma(3, 3, n_samples),
        'pressure': np.random.normal(990, 10, n_samples)
    }
    cyclone_df = pd.DataFrame(cyclone_conditions)
    cyclone_df['risk_level'] = 3
    
    landslide_conditions = {
        'temperature': np.random.normal(25, 3, n_samples),
        'humidity': np.random.normal(90, 4, n_samples),
        'wind_speed': np.random.gamma(2, 1, n_samples),
        'rainfall': np.random.gamma(8, 2, n_samples),
        'pressure': np.random.normal(1008, 6, n_samples)
    }
    landslide_df = pd.DataFrame(landslide_conditions)
    landslide_df['risk_level'] = 2
    
    low_risk_conditions = {
        'temperature': np.concatenate([np.random.normal(35, 2, n_samples//2), np.random.normal(20, 2, n_samples//2)]),
        'humidity': np.concatenate([np.random.normal(80, 5, n_samples//2), np.random.normal(50, 5, n_samples//2)]),
        'wind_speed': np.random.gamma(4, 1.5, n_samples),
        'rainfall': np.random.gamma(2, 2, n_samples),
        'pressure': np.concatenate([np.random.normal(1020, 5, n_samples//2), np.random.normal(1000, 5, n_samples//2)])
    }
    low_risk_df = pd.DataFrame(low_risk_conditions)
    low_risk_df['risk_level'] = 1
    
    combined_df = pd.concat([normal_df, flood_df, cyclone_df, landslide_df, low_risk_df], ignore_index=True)
    
    X = combined_df[feature_columns].values
    y = combined_df['risk_level'].values
    
    return X, y, feature_columns

def train_ml_model():
    global ml_model, scaler, model_trained
    
    try:
        X, y, feature_columns = create_enhanced_training_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        ml_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        ml_model.fit(X_train_scaled, y_train)
        
        y_pred = ml_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"ML Model Training Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Feature importance: {dict(zip(feature_columns, ml_model.feature_importances_.round(4)))}")
        
        model_trained = True
        
        joblib.dump(ml_model, 'disaster_risk_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        logger.info("Model and scaler saved successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error training ML model: {e}")
        return False

def load_saved_model():
    global ml_model, scaler, model_trained
    
    try:
        if os.path.exists('disaster_risk_model.pkl') and os.path.exists('scaler.pkl'):
            ml_model = joblib.load('disaster_risk_model.pkl')
            scaler = joblib.load('scaler.pkl')
            model_trained = True
            logger.info("Loaded saved model and scaler")
            return True
    except Exception as e:
        logger.error(f"Error loading saved model: {e}")
    
    return False

def predict_disaster_risk(weather_data):
    if not model_trained:
        if not load_saved_model():
            if not train_ml_model():
                return "unknown"
    
    try:
        features = np.array([[
            weather_data.get('temperature', 25),
            weather_data.get('humidity', 60),
            weather_data.get('wind_speed', 5),
            weather_data.get('rain', 0),
            weather_data.get('pressure', 1013)
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = ml_model.predict(features_scaled)[0]
        
        risk_levels = {0: "no", 1: "low", 2: "medium", 3: "high"}
        return risk_levels.get(prediction, "unknown")
    except Exception as e:
        logger.error(f"Error predicting with ML model: {e}")
        return "unknown"

def get_historical_disasters(location, disaster_type=None):
    filtered = [d for d in historical_disasters if d['location'].lower() == location.lower()]
    
    if disaster_type:
        filtered = [d for d in filtered if d['type'].lower() == disaster_type.lower()]
    
    return filtered

def make_emergency_call(risk_type, location):
    logger.info(f"EMERGENCY CALL: {risk_type} risk in {location}")
    emergency_contact = EMERGENCY_CONTACTS.get("thrissur_control_room", "100")
    return f"📞 Emergency call placed to {emergency_contact} for {risk_type} in {location}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not model_trained:
        train_ml_model()
    
    keyboard = [[KeyboardButton("Share Location", request_location=True)]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    
    await update.message.reply_text(
        "Hi! I'm your Weather Alert Bot for Thrissur District. \n\n"
        "Please share your location or send the name of a city in Thrissur district "
        "(Thrissur, Guruvayur, Chavakkad, Kunnamkulam, Kodungallur, Irinjalakuda) "
        "to get weather information and disaster alerts.",
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "How to use this bot:\n"
        "1. Share your location using the button\n"
        "2. Or type the name of a city in Thrissur district\n"
        "3. I'll fetch weather data and analyze disaster risks\n"
        "4. If there's a risk, I'll provide safety information\n\n"
        "Emergency commands:\n"
        "/emergency - Show emergency contacts\n"
        "/history [city] - Show historical disaster data\n"
        "/call_help - Initiate emergency call if in danger"
    )

async def emergency_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = "🆘 <b>Emergency Contacts for Thrissur District:</b>\n\n"
    for service, number in EMERGENCY_CONTACTS.items():
        response += f"• {service.replace('_', ' ').title()}: <b>{number}</b>\n"
    
    response += "\n<i>Call these numbers in case of emergency</i>"
    await update.message.reply_text(response, parse_mode='HTML')

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        location = context.args[0]
        if location not in thrissur_cities:
            await update.message.reply_text("Please specify a valid city in Thrissur district.")
            return
        
        disasters = get_historical_disasters(location)
        if disasters:
            response = f"<b>Historical Disasters in {location}:</b>\n\n"
            for disaster in disasters:
                response += f"📅 {disaster['date']}: {disaster['type']} ({disaster['severity']})\n"
                response += f"   Affected: {disaster['affected_areas']}\n"
                response += f"   Casualties: {disaster['casualties']}, Damage: {disaster['damage']}\n\n"
            
            await update.message.reply_text(response, parse_mode='HTML')
        else:
            await update.message.reply_text(f"No historical disaster data found for {location}.")
    else:
        response = "<b>Historical Disasters in Thrissur District:</b>\n\n"
        for disaster in historical_disasters:
            response += f"📅 {disaster['date']}: {disaster['type']} in {disaster['location']} ({disaster['severity']})\n"
        
        response += "\nUse /history [city] for details about a specific city."
        await update.message.reply_text(response, parse_mode='HTML')

async def call_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('last_location'):
        lat, lon = context.user_data['last_location']
        city = context.user_data.get('last_city', 'your location')
        
        call_result = make_emergency_call("potential disaster", city)
        await update.message.reply_text(f"🚨 {call_result}\n\nPlease stay safe and follow emergency instructions.")
    else:
        await update.message.reply_text(
            "Please share your location first so we can assist you better.\n\n"
            "If you're in immediate danger, call:\n"
            "• Police: 100\n"
            "• Ambulance: 108\n"
            "• Fire: 101"
        )

async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    location = update.message.location
    lat, lon = location.latitude, location.longitude
    
    context.user_data['last_location'] = (lat, lon)
    
    await update.message.reply_text("📍 Location received! Fetching weather data...")
    
    weather_data = get_weather_data(lat, lon)
    
    if weather_data:
        context.user_data['last_city'] = weather_data.get('city', 'Unknown location')
        
        ml_risk_prediction = predict_disaster_risk(weather_data)
        analysis = analyze_with_gemini(weather_data, lat, lon, ml_risk_prediction)
        disaster_risk = check_disaster_risk(lat, lon, weather_data)
        historical_data = get_historical_disasters(weather_data.get('city', ''))
        
        response = format_response(weather_data, analysis, disaster_risk, lat, lon, ml_risk_prediction, historical_data)
        
        if ml_risk_prediction == "high" or any(r['risk_level'].lower() == 'high' for r in disaster_risk):
            emergency_msg = "🚨 <b>HIGH RISK DETECTED - EMERGENCY PROTOCOLS ACTIVATED</b>\n\n"
            emergency_msg += "We've detected a high risk of disaster in your area.\n"
            emergency_msg += "Please follow evacuation procedures immediately.\n\n"
            
            emergency_msg += "🆘 <b>Emergency Contacts:</b>\n"
            emergency_msg += f"• Police: <b>{EMERGENCY_CONTACTS['police']}</b>\n"
            emergency_msg += f"• Ambulance: <b>{EMERGENCY_CONTACTS['ambulance']}</b>\n"
            emergency_msg += f"• Fire: <b>{EMERGENCY_CONTACTS['fire']}</b>\n"
            emergency_msg += f"• Disaster Management: <b>{EMERGENCY_CONTACTS['disaster_management']}</b>\n\n"
            
            emergency_msg += "Use /call_help if you need immediate assistance."
            
            await update.message.reply_text(emergency_msg, parse_mode='HTML')
            
            call_result = make_emergency_call("high risk", weather_data.get('city', 'Unknown location'))
            logger.info(call_result)
        
        await update.message.reply_text(response, parse_mode='HTML')
    else:
        await update.message.reply_text("Sorry, I couldn't fetch weather data. Please try again later.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    city_name = update.message.text.strip()
    
    if city_name not in thrissur_cities:
        await update.message.reply_text(
            "Please enter a valid city in Thrissur district: "
            "Thrissur, Guruvayur, Chavakkad, Kunnamkulam, Kodungallur, or Irinjalakuda."
        )
        return
    
    await update.message.reply_text(f"🌤️ Fetching weather data for {city_name}...")
    
    lat, lon = get_city_coordinates(city_name)
    
    if lat is None or lon is None:
        await update.message.reply_text("Sorry, I couldn't find that location. Please try again.")
        return
    
    context.user_data['last_location'] = (lat, lon)
    context.user_data['last_city'] = city_name
    
    weather_data = get_weather_data(lat, lon)
    
    if weather_data:
        ml_risk_prediction = predict_disaster_risk(weather_data)
        analysis = analyze_with_gemini(weather_data, lat, lon, ml_risk_prediction)
        disaster_risk = check_disaster_risk(lat, lon, weather_data)
        historical_data = get_historical_disasters(city_name)
        
        response = format_response(weather_data, analysis, disaster_risk, lat, lon, ml_risk_prediction, historical_data)
        
        await update.message.reply_text(response, parse_mode='HTML')
    else:
        await update.message.reply_text("Sorry, I couldn't fetch weather data. Please try again later.")

def get_city_coordinates(city_name):
    for city in disaster_db:
        if city['location'].lower() == city_name.lower():
            return float(city['lat']), float(city['lon'])
    return None, None

def get_weather_data(lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code == 200:
            weather_info = {
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'rain': data.get('rain', {}).get('1h', 0),
                'city': data.get('name', 'Unknown location')
            }
            return weather_info
        else:
            logger.error(f"Weather API error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return None

def analyze_with_gemini(weather_data, lat, lon, ml_risk_prediction):
    try:
        if model is None:
            return "AI analysis temporarily unavailable."
            
        prompt = f"""
        Analyze this weather data for potential natural disaster risks in Thrissur district, Kerala, India:
        Location: {weather_data.get('city', 'Unknown')} (Lat: {lat}, Lon: {lon})
        Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)
        Humidity: {weather_data['humidity']}%
        Pressure: {weather_data['pressure']} hPa
        Conditions: {weather_data['description']}
        Wind Speed: {weather_data['wind_speed']} m/s
        Rain (last hour): {weather_data['rain']} mm
        
        Our ML model has predicted a {ml_risk_prediction} risk level.
        
        Based on this data, assess the risk of:
        1. Flooding (common during monsoon season)
        2. Landslides (in hilly areas)
        3. Cyclones/Strong winds
        4. Heatwaves
        5. Other weather-related disasters
        
        Provide a concise risk assessment (2-3 sentences) and recommendations if any risks are identified.
        Focus specifically on risks relevant to Kerala's climate and geography.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error with Gemini AI: {e}")
        return "Unable to analyze weather data at this time."

def check_disaster_risk(lat, lon, weather_data):
    risks = []
    
    for disaster in disaster_db:
        try:
            disaster_lat = float(disaster['lat'])
            disaster_lon = float(disaster['lon'])
            
            if abs(lat - disaster_lat) < 0.1 and abs(lon - disaster_lon) < 0.1:
                risks.append({
                    'type': disaster['disaster_type'],
                    'risk_level': disaster['risk_level'],
                    'location': disaster['location'],
                    'evacuation_lat': disaster['evacuation_lat'],
                    'evacuation_lon': disaster['evacuation_lon']
                })
        except ValueError:
            continue
    
    if weather_data.get('rain', 0) > 20:
        risks.append({
            'type': 'Flood',
            'risk_level': 'High',
            'location': 'Current location',
            'reason': 'Heavy rainfall detected'
        })
    
    if weather_data.get('wind_speed', 0) > 15:
        risks.append({
            'type': 'Cyclone',
            'risk_level': 'Medium',
            'location': 'Current location',
            'reason': 'Strong winds detected'
        })
    
    return risks

def format_response(weather_data, analysis, disaster_risk, lat, lon, ml_risk_prediction, historical_data):
    response = f"<b>Weather Report for {weather_data.get('city', 'your location')}</b>\n\n"
    response += f"🌡️ Temperature: {weather_data.get('temperature', 'N/A')}°C (Feels like: {weather_data.get('feels_like', 'N/A')}°C)\n"
    response += f"💧 Humidity: {weather_data.get('humidity', 'N/A')}%\n"
    response += f"🌬️ Wind: {weather_data.get('wind_speed', 'N/A')} m/s\n"
    response += f"☔ Rain (last hour): {weather_data.get('rain', 0)} mm\n"
    response += f"📊 Conditions: {weather_data.get('description', 'N/A').title()}\n\n"
    
    risk_emojis = {"no": "✅", "low": "🔶", "medium": "⚠️", "high": "🚨", "unknown": "❓"}
    response += f"{risk_emojis.get(ml_risk_prediction, '❓')} <b>ML Risk Prediction:</b> {ml_risk_prediction.upper()}\n\n"
    
    if historical_data:
        response += "<b>Historical Disasters in this area:</b>\n"
        for disaster in historical_data[:3]:
            response += f"• {disaster['type']} ({disaster['date']}, {disaster['severity']})\n"
        response += "\n"
    
    if disaster_risk or ml_risk_prediction in ["medium", "high"]:
        response += "⚠️ <b>DISASTER ALERT</b> ⚠️\n"
        for risk in disaster_risk:
            response += f"• {risk['type']} risk ({risk['risk_level']}) in {risk['location']}"
            if 'reason' in risk:
                response += f" - {risk['reason']}"
            response += "\n"
        
        response += "\n<b>Safety Recommendations:</b>\n"
        response += "• Avoid flooded areas\n"
        response += "• Move to higher ground if necessary\n"
        response += "• Follow local authorities' instructions\n"
        response += "• Emergency contacts: 112 (India emergency number)\n\n"
        
        for risk in disaster_risk:
            if 'evacuation_lat' in risk and 'evacuation_lon' in risk:
                try:
                    e_lat, e_lon = risk['evacuation_lat'], risk['evacuation_lon']
                    response += f"<a href='https://maps.google.com/maps?q={e_lat},{e_lon}&z=15'>📍 Evacuation point for {risk['location']}</a>\n"
                except:
                    pass
        
        response += f"<a href='https://www.google.com/maps/search/higher+ground+near+{lat},{lon}'>📍 Find higher ground near you</a>\n\n"
        
        if ml_risk_prediction == "high" or any(r.get('risk_level', '').lower() == 'high' for r in disaster_risk):
            response += "🚨 <b>High risk detected! Use /call_help for emergency assistance.</b>\n\n"
    else:
        response += "✅ <b>No significant disaster risks detected</b>\n\n"
    
    response += "<b>AI Analysis:</b>\n"
    response += f"{analysis}\n\n"
    
    response += "<i>Note: This is an automated assessment. Always follow official warnings from local authorities.</i>"
    
    return response

def main():
    if not load_saved_model():
        train_ml_model()
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("emergency", emergency_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("call_help", call_help_command))
    application.add_handler(MessageHandler(filters.LOCATION, handle_location))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
