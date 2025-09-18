from flask import Flask,request,render_template,send_from_directory,session,redirect,url_for
import os
import math
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import pandas
import sklearn
import pickle
import datetime

from typing import List, Dict, Any
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY','dev-secret-key-change')  # needed for session

# --------------------
# Localization resources (English, Hindi minimal set). Extendable.
# --------------------
LOCALE = {
    'en': {
        'easy_label': 'Easy Explanation:',
        'suggestions_label': 'Suggestions',
        'good_factors': 'Good factors',
        'needs_attention': 'Needs attention',
        'model_confidence': 'Model confidence',
        'planting_window': 'Planting Window',
        'dosage_label': 'Approximate nutrient guidance',
        'disclaimer': 'Generated automatically. Always validate with local experts.'
    },
    'hi': {
        'easy_label': 'सरल जानकारी:',
        'suggestions_label': 'सुझाव',
        'good_factors': 'अच्छे कारक',
        'needs_attention': 'ध्यान देने वाले कारक',
        'model_confidence': 'मॉडल विश्वास',
        'planting_window': 'रोपण समय',
        'dosage_label': 'अनुमानित पोषक दिशा‑निर्देश',
        'disclaimer': 'स्वचालित रूप से तैयार। स्थानीय कृषि विशेषज्ञ से पुष्टि करें.'
    }
}

# --------------------
# Load dataset & compute per-crop statistics for explanations
# --------------------
try:
    _df = pandas.read_csv('Crop_recommendation.csv')
    # Keep feature columns mapping to user input order
    FEATURE_COLUMNS = ['N','P','K','temperature','humidity','ph','rainfall']
    # group stats
    _grp = _df.groupby('label')[FEATURE_COLUMNS].agg(['mean','std'])
    # Convert to nested dict: stats[crop][feature] = {'mean':x,'std':y}
    crop_stats = {}
    for crop_label in _grp.index:
        crop_stats[crop_label] = {}
        for f in FEATURE_COLUMNS:
            mean_val = _grp.loc[crop_label,(f,'mean')]
            std_val = _grp.loc[crop_label,(f,'std')]
            crop_stats[crop_label][f] = {'mean': float(mean_val), 'std': float(std_val) if not math.isnan(std_val) else 0.0}
except Exception as e:
    crop_stats = {}
    FEATURE_COLUMNS = ['N','P','K','temperature','humidity','ph','rainfall']
    print("[WARN] Failed to load stats for explanations:", e)

# --------------------
# Planting window: simple climate preference profiles (temp °C, rainfall mm/day)
# These ranges are illustrative; refine with agronomic data as needed.
# Each crop maps to preferred mean temperature range and approximate daily rainfall range.
# We'll evaluate upcoming 14-day forecast mean temperature & accumulated rainfall distribution.
# --------------------
CLIMATE_PROFILES = {
    'Rice':       {'temp': (20, 32), 'rain_mm_day': (4, 15)},
    'Maize':      {'temp': (18, 30), 'rain_mm_day': (2, 10)},
    'Jute':       {'temp': (22, 34), 'rain_mm_day': (5, 18)},
    'Cotton':     {'temp': (21, 33), 'rain_mm_day': (2, 8)},
    'Coconut':    {'temp': (22, 34), 'rain_mm_day': (3, 12)},
    'Papaya':     {'temp': (21, 33), 'rain_mm_day': (3, 10)},
    'Orange':     {'temp': (15, 28), 'rain_mm_day': (1, 6)},
    'Apple':      {'temp': (12, 24), 'rain_mm_day': (1, 6)},
    'Muskmelon':  {'temp': (18, 30), 'rain_mm_day': (1, 5)},
    'Watermelon': {'temp': (20, 32), 'rain_mm_day': (1, 5)},
    'Grapes':     {'temp': (15, 30), 'rain_mm_day': (1, 6)},
    'Mango':      {'temp': (22, 35), 'rain_mm_day': (1, 6)},
    'Banana':     {'temp': (22, 34), 'rain_mm_day': (3, 12)},
    'Pomegranate':{'temp': (18, 32), 'rain_mm_day': (1, 6)},
    'Lentil':     {'temp': (10, 25), 'rain_mm_day': (1, 5)},
    'Blackgram':  {'temp': (20, 32), 'rain_mm_day': (2, 10)},
    'Mungbean':   {'temp': (20, 34), 'rain_mm_day': (2, 10)},
    'Mothbeans':  {'temp': (25, 36), 'rain_mm_day': (0, 4)},
    'Pigeonpeas': {'temp': (20, 33), 'rain_mm_day': (2, 10)},
    'Kidneybeans':{'temp': (18, 28), 'rain_mm_day': (2, 8)},
    'Chickpea':   {'temp': (15, 27), 'rain_mm_day': (1, 5)},
    'Coffee':     {'temp': (18, 28), 'rain_mm_day': (3, 10)},
}

def evaluate_planting_window(crop: str, daily_forecast: List[Dict[str, Any]]):
    """Given crop name and list of daily forecast entries with keys:
    {'date': date, 't_mean': float, 'rain_mm': float}
    Return suitability summary.
    """
    profile = CLIMATE_PROFILES.get(crop)
    if not profile:
        return { 'status': 'unknown', 'message': 'No climate profile available', 'score': None }
    t_low, t_high = profile['temp']
    r_low, r_high = profile['rain_mm_day']
    ok_days = 0
    near_days = 0
    annotated = []
    for d in daily_forecast:
        t = d['t_mean']
        r = d['rain_mm']
        temp_ok = t_low <= t <= t_high
        rain_ok = r_low <= r <= r_high
        if temp_ok and rain_ok:
            ok_days += 1
            quality = 'ideal'
        else:
            # near if within 2C or 2mm outside
            temp_near = (t_low - 2) <= t <= (t_high + 2)
            rain_near = (r_low - 2) <= r <= (r_high + 2)
            if temp_near and rain_near:
                near_days += 1
                quality = 'near'
            else:
                quality = 'poor'
        annotated.append({**d, 'quality': quality})
    # Basic scoring: ideal day =2 pts, near=1 pt
    score = ok_days * 2 + near_days
    max_score = len(daily_forecast) * 2
    pct = (score / max_score) * 100 if max_score else 0
    if pct >= 70:
        status = 'optimal'
        message = 'Conditions look favorable for planting in the next two weeks.'
    elif pct >= 45:
        status = 'moderate'
        message = 'Mixed conditions; acceptable but monitor weather shifts.'
    else:
        status = 'poor'
        message = 'Suboptimal conditions—delay planting if possible.'
    return {
        'status': status,
        'message': message,
        'ideal_days': ok_days,
        'near_days': near_days,
        'total_days': len(daily_forecast),
        'score_percent': round(pct, 1),
        'profile': profile,
        'daily': annotated
    }

def build_explanation(predicted_crop: str, user_values: dict, probabilities=None, crop_dict=None, lang='en'):
    """Generate an explanation using training data stats.
    user_values: dict with raw feature names matching FEATURE_COLUMNS (dataset names)
    probabilities: (crop_name -> prob) mapping if available
    returns explanation dict or None
    """
    crop_key = predicted_crop.lower() if predicted_crop else None
    if not crop_key or crop_key not in crop_stats:
        return None
    stats = crop_stats[crop_key]
    feature_analysis = []
    supportive = []
    weak = []
    for ds_name in FEATURE_COLUMNS:
        val = user_values.get(ds_name)
        m = stats[ds_name]['mean']
        s = stats[ds_name]['std'] or 0.0
        if s == 0:
            z = 0.0
        else:
            z = (val - m)/s
        az = abs(z)
        if az <= 0.75:
            assess = 'within typical range'
            cls = 'good'
            supportive.append(ds_name)
        elif az <= 1.5:
            assess = 'slightly different'
            cls = 'ok'
        else:
            assess = 'outside typical range'
            cls = 'warn'
            weak.append(ds_name)
        feature_analysis.append({
            'name': ds_name,
            'value': round(val,2) if isinstance(val,(int,float)) else val,
            'mean': round(m,2),
            'std': round(s,2),
            'z': round(z,2),
            'assessment': assess,
            'assessment_class': cls
        })
    # Compose technical summary
    if lang not in LOCALE: lang = 'en'
    if supportive and not weak:
        summary_en = f"Most of your parameters ({', '.join(supportive)}) closely match typical conditions for {predicted_crop}."
        summary_hi = f"आपके अधिकांश मान ({', '.join(supportive)}) {predicted_crop} के सामान्य मानों से मेल खाते हैं।"
        summary = summary_en if lang=='en' else summary_hi
    elif supportive and weak:
        summary_en = f"{predicted_crop} fits because {', '.join(supportive)} align with its profile; consider adjusting {', '.join(weak)}."
        summary_hi = f"{predicted_crop} उपयुक्त है क्योंकि {', '.join(supportive)} अच्छे हैं; {', '.join(weak)} को सुधारें।"
        summary = summary_en if lang=='en' else summary_hi
    elif weak and not supportive:
        summary_en = f"Several inputs ({', '.join(weak)}) differ from the usual profile, so treat this recommendation with caution."
        summary_hi = f"कुछ मान ({', '.join(weak)}) सामान्य सीमा से अलग हैं, सावधानी रखें।"
        summary = summary_en if lang=='en' else summary_hi
    else:
        summary = "Insufficient statistical alignment data." if lang=='en' else 'पर्याप्त सांख्यिकीय डेटा नहीं।'

    # Simple farmer-friendly summary (avoid stats jargon)
    name_map = {
        'N':'nitrogen','P':'phosphorus','K':'potassium','temperature':'temperature','humidity':'humidity','ph':'soil pH','rainfall':'rainfall'
    }
    friendly_good = [name_map.get(x,x) for x in supportive]
    friendly_bad = [name_map.get(x,x) for x in weak]
    if lang=='hi':
        if friendly_good and not friendly_bad:
            farmer_summary = f"यह फसल उपयुक्त है। {', '.join(friendly_good)} अच्छे हैं। देखभाल जारी रखें।"
        elif friendly_good and friendly_bad:
            farmer_summary = f"फसल ठीक है। अच्छे: {', '.join(friendly_good)} | सुधारें: {', '.join(friendly_bad)}"
        elif friendly_bad and not friendly_good:
            farmer_summary = f"सावधान रहें। {', '.join(friendly_bad)} इस फसल के लिए सही स्तर पर नहीं हैं।"
        else:
            farmer_summary = "फसल चल सकती है, कृपया मान पुनः जाँचें।"
    else:
        if friendly_good and not friendly_bad:
            farmer_summary = f"This crop suits your field. Your {', '.join(friendly_good)} levels look good. Keep the same care." \
                              f""
        elif friendly_good and friendly_bad:
            farmer_summary = f"Crop is suitable. Good: {', '.join(friendly_good)}. Needs attention: {', '.join(friendly_bad)}."
        elif friendly_bad and not friendly_good:
            farmer_summary = f"Be careful. {', '.join(friendly_bad)} are not in the best range for this crop. Consider improvement or another crop."\
            
        else:
            farmer_summary = f"This crop may work, but please double‑check your soil values."

    # Basic actionable suggestions for weak features
    suggestions = []
    suggestion_templates = {
        'N': 'Add well‑decomposed compost or a balanced nitrogen fertilizer before planting.',
        'P': 'Incorporate rock phosphate or single super phosphate; avoid over‑watering early.',
        'K': 'Add potash (muriate of potash) or wood ash in moderate amounts.',
        'ph': 'If soil is too acidic (low pH), apply agricultural lime; if too alkaline (high pH), add organic matter.',
        'temperature': 'Consider waiting for a few warmer/cooler days if possible.',
        'humidity': 'Ensure good airflow; avoid waterlogging and stagnant moisture.',
        'rainfall': 'Plan irrigation or drainage depending on expected rain.'
    }
    for w in weak:
        key = w.lower()
        # map dataset name to template key (ph name difference already matches)
        base = suggestion_templates.get(key, f'Improve {name_map.get(w,w)} conditions.')
        if lang=='hi':
            # Very simple Hindi equivalents (can refine later)
            hi_map = {
                'Add well‑decomposed compost or a balanced nitrogen fertilizer before planting.': 'अच्छी सड़ी खाद या संतुलित नाइट्रोजन उर्वरक डालें।',
                'Incorporate rock phosphate or single super phosphate; avoid over‑watering early.': 'रॉक फॉस्फेट / एसएसपी मिलाएँ; अधिक सिंचाई न करें।',
                'Add potash (muriate of potash) or wood ash in moderate amounts.': 'पोटाश (MOP) या राख सीमित मात्रा में डालें।',
                'If soil is too acidic (low pH), apply agricultural lime; if too alkaline (high pH), add organic matter.': 'अम्लीय होने पर चुना डालें; अधिक क्षारीय होने पर जैविक पदार्थ मिलाएँ।',
                'Consider waiting for a few warmer/cooler days if possible.': 'तापमान ठीक होने तक कुछ दिन प्रतीक्षा करें।',
                'Ensure good airflow; avoid waterlogging and stagnant moisture.': 'अच्छा वायु प्रवाह रखें; पानी भराव से बचें।',
                'Plan irrigation or drainage depending on expected rain.': 'अनुमानित वर्षा के अनुसार सिंचाई या निकासी की योजना बनाएं।'
            }
            base = hi_map.get(base, base)
        suggestions.append(base)

    # Probability details
    prob_val = None
    top_alts = None
    if probabilities:
        prob_val = probabilities.get(predicted_crop)
        # get top 3 other crops
        alt = [ (c,p) for c,p in probabilities.items() if c != predicted_crop ]
        alt.sort(key=lambda x: x[1], reverse=True)
        top_alts = [{ 'crop': c, 'prob': p } for c,p in alt[:3]] if alt else None

    return {
        'summary': summary,
        'probability': prob_val,
        'top_alternatives': top_alts,
        'feature_analysis': feature_analysis,
        'farmer_summary': farmer_summary,
        'suggestions': suggestions,
        'supportive_features': supportive,
        'weak_features': weak,
        'lang': lang,
        'labels': LOCALE.get(lang, LOCALE['en'])
    }

def generate_recommendation_pdf(path:str, crop:str, result_text:str, feature_values:dict, explanation:dict, probabilities_map:dict):
    """Generate a detailed PDF report."""
    try:
        doc = SimpleDocTemplate(path, pagesize=letter, leftMargin=50, rightMargin=50, topMargin=60, bottomMargin=50)
        styles = getSampleStyleSheet()
        story = []
        title_style = styles['Heading1']
        title_style.textColor = colors.HexColor('#0a8f6a')
        story.append(Paragraph('Crop Recommendation Report', title_style))
        story.append(Spacer(1,12))
        story.append(Paragraph(f'<b>Recommended Crop:</b> {crop}', styles['Heading3']))
        story.append(Paragraph(result_text, styles['Normal']))
        story.append(Spacer(1,10))
        if explanation:
            story.append(Paragraph(f"<b>Technical Summary:</b> {explanation.get('summary','')}", styles['Normal']))
            story.append(Paragraph(f"<b>Farmer Friendly:</b> {explanation.get('farmer_summary','')}", styles['Normal']))
            if explanation.get('suggestions'):
                story.append(Spacer(1,6))
                story.append(Paragraph('<b>Suggestions / Actions:</b>', styles['Normal']))
                for s in explanation['suggestions']:
                    story.append(Paragraph(f"- {s}", styles['Normal']))
        story.append(Spacer(1,12))
        # Inputs table
        fv_rows = [['Parameter','Value']]
        for k,v in feature_values.items():
            fv_rows.append([k, v])
        t_inputs = Table(fv_rows, hAlign='LEFT')
        t_inputs.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#0a8f6a')),
            ('TEXTCOLOR',(0,0),(-1,0), colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,0),10),
            ('BOTTOMPADDING',(0,0),(-1,0),6),
            ('BACKGROUND',(0,1),(-1,-1), colors.whitesmoke),
            ('GRID',(0,0),(-1,-1), 0.3, colors.grey)
        ]))
        story.append(Paragraph('<b>Input Values</b>', styles['Heading3']))
        story.append(t_inputs)
        story.append(Spacer(1,14))
        # Probability table
        if probabilities_map:
            prob_rows = [['Crop','Probability %']]
            for c,p in sorted(probabilities_map.items(), key=lambda x: x[1], reverse=True)[:10]:
                prob_rows.append([c, f"{p*100:.2f}"])
            t_prob = Table(prob_rows, hAlign='LEFT')
            t_prob.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#134f70')),
                ('TEXTCOLOR',(0,0),(-1,0), colors.white),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('FONTSIZE',(0,0),(-1,0),10),
                ('BOTTOMPADDING',(0,0),(-1,0),6),
                ('BACKGROUND',(0,1),(-1,-1), colors.HexColor('#e9f4f2')),
                ('GRID',(0,0),(-1,-1), 0.3, colors.grey)
            ]))
            story.append(Paragraph('<b>Model Probability (Top 10)</b>', styles['Heading3']))
            story.append(t_prob)
            story.append(Spacer(1,14))
        # Feature analysis table
        if explanation and explanation.get('feature_analysis'):
            fa_rows = [['Feature','Your','Mean','Std','Z','Assessment']]
            for f in explanation['feature_analysis']:
                fa_rows.append([
                    f['name'], f['value'], f['mean'], f['std'], f['z'], f['assessment']
                ])
            t_fa = Table(fa_rows, colWidths=[70,40,40,40,30,140], hAlign='LEFT')
            t_fa.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#37474F')),
                ('TEXTCOLOR',(0,0),(-1,0), colors.white),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('FONTSIZE',(0,0),(-1,0),9),
                ('BACKGROUND',(0,1),(-1,-1), colors.HexColor('#f5f9f8')),
                ('GRID',(0,0),(-1,-1), 0.25, colors.grey)
            ]))
            story.append(Paragraph('<b>Feature Alignment</b>', styles['Heading3']))
            story.append(t_fa)
            story.append(Spacer(1,14))
        story.append(Paragraph('<i>Generated automatically. Always validate with local agronomy advice.</i>', styles['Normal']))
        doc.build(story)
    except Exception as e:
        print('[WARN] PDF generation failed:', e)

@app.route('/')
def index():
    pdata = session.pop('prediction_data', None)
    # persistent language across requests
    lang = session.get('lang','en')
    if pdata:
        return render_template("index.html", lang=lang, **pdata)
    return render_template("index.html", lang=lang)

@app.route("/predict",methods=['POST'])
def predict():
    # Extract form inputs
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    # Language preference (persist)
    lang = request.form.get('lang', session.get('lang','en'))
    session['lang'] = lang

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    explanation = None
    probabilities_map = None
    # Attempt probability extraction
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(final_features)[0]
            probabilities_map = {crop_dict[idx]: float(p) for idx, p in enumerate(proba, start=1) if idx in crop_dict}
        except Exception:
            probabilities_map = None

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated right there"
        user_ds_values = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temp,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        explanation = build_explanation(crop, user_ds_values, probabilities=probabilities_map, crop_dict=crop_dict, lang=lang)

        # Ensure downloads directory exists
        downloads_dir = os.path.join('static', 'downloads')
        os.makedirs(downloads_dir, exist_ok=True)

        pdf_path = os.path.join(downloads_dir, 'recommendation.pdf')
        text_path = os.path.join(downloads_dir, 'recommendation.txt')
        feature_values_plain = {
            'Nitrogen (N)': N,
            'Phosphorus (P)': P,
            'Potassium (K)': K,
            'Temperature (°C)': temp,
            'Humidity (%)': humidity,
            'Soil pH': ph,
            'Rainfall (mm)': rainfall
        }
        generate_recommendation_pdf(
            pdf_path,
            crop,
            result,
            feature_values_plain,
            explanation,
            probabilities_map
        )
        # Text export (human-readable)
        try:
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write('CROP RECOMMENDATION REPORT\n')
                f.write('='*32+'\n')
                f.write(f'Recommended Crop: {crop}\n')
                f.write(result+'\n\n')
                if explanation:
                    f.write('Technical Summary: '+explanation.get('summary','')+'\n')
                    f.write('Farmer Friendly: '+explanation.get('farmer_summary','')+'\n')
                f.write('\nInput Values:\n')
                for k,v in feature_values_plain.items():
                    f.write(f'  - {k}: {v}\n')
                if probabilities_map:
                    f.write('\nModel Probabilities (Top 10):\n')
                    for c_name,p in sorted(probabilities_map.items(), key=lambda x: x[1], reverse=True)[:10]:
                        f.write(f'  - {c_name}: {p*100:.2f}%\n')
                if explanation and explanation.get('feature_analysis'):
                    f.write('\nFeature Alignment:\n')
                    for fa in explanation['feature_analysis']:
                        f.write(f"  - {fa['name']}: value={fa['value']} mean={fa['mean']} z={fa['z']} assess={fa['assessment']}\n")
                if explanation and explanation.get('suggestions'):
                    f.write('\nSuggestions:\n')
                    for s in explanation['suggestions']:
                        f.write(f'  - {s}\n')
                f.write('\nDisclaimer: Generated automatically. Validate with local agronomy experts.\n')
        except Exception as e:
            print('[WARN] Text export failed:', e)
        download_links = {
            'pdf': '/static/downloads/recommendation.pdf',
            'txt': '/static/downloads/recommendation.txt'
        }
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        download_links = None

    # Store in session then redirect (PRG pattern)
    session['prediction_data'] = {
        'result': result,
        'download_links': download_links,
        'explanation': explanation
    }
    return redirect(url_for('index'))




# python main
from flask import jsonify
import requests

"""
Enhanced soil/weather endpoint
--------------------------------
Replaces the earlier OpenWeatherMap soil API call (which returned 400 due to
an incorrect / deprecated endpoint or missing paid plan access) with a
zero‑API‑key solution using Open‑Meteo. Open‑Meteo supplies hourly forecast
and (optionally) current values for:
  - temperature (2m)
  - relative humidity (2m)
  - precipitation
  - soil moisture (0‑7 cm)

Because public, free APIs typically do NOT provide direct soil nutrient (N/P/K)
or pH data, we return clearly labeled placeholder estimates. These allow the
frontend autofill to function; the user may edit before prediction.

To integrate real nutrient / pH values later, replace the placeholder section
with calls to a sensor service or an agronomic API and map into the same keys.
"""

@app.route('/get_soil_data')
def get_soil_data():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    if lat is None or lon is None:
        return jsonify({
            "error": "Missing or invalid lat/lon parameters.",
            "hint": "Provide numeric query params ?lat=..&lon=.."
        }), 400

    # Open-Meteo endpoint (no API key required)
    meteo_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation,soil_moisture_0_to_7cm"
        "&forecast_days=1&timezone=auto"
    )

    try:
        resp = requests.get(meteo_url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        # Network or HTTP error: return graceful fallback
        return jsonify({
            "error": "Failed to fetch weather/soil moisture data",
            "details": str(e),
            "placeholders": True
        }), 502

    hourly = data.get('hourly', {})
    # Extract last available readings safely
    def last_value(key):
        seq = hourly.get(key)
        if isinstance(seq, list) and seq:
            return seq[-1]
        return None

    temperature = last_value('temperature_2m')  # Celsius
    humidity = last_value('relative_humidity_2m')  # %
    rainfall = last_value('precipitation')  # mm for last hour
    moisture = last_value('soil_moisture_0_to_7cm')  # m^3/m^3 (volumetric)

    # Convert / normalize values where helpful
    if moisture is not None:
        # Express volumetric moisture roughly as percentage (approximation)
        moisture_pct = round(moisture * 100, 2)
    else:
        moisture_pct = None

    # Placeholder nutrient & pH values (heuristic defaults)
    # These are intentionally moderate mid-range values. Users can edit.
    placeholder_n = 50  # mg/kg (example mid value)
    placeholder_p = 40  # mg/kg
    placeholder_k = 50  # mg/kg
    placeholder_ph = 6.5  # slightly acidic-neutral ideal for many crops

    result = {
        "temperature": round(temperature, 2) if temperature is not None else None,
        "humidity": round(humidity, 2) if humidity is not None else None,
        "rainfall": round(rainfall, 2) if rainfall is not None else None,
        "moisture": moisture_pct,  # percentage approximation
        # Placeholders (user should adjust)
        "nitrogen": placeholder_n,
        "phosphorus": placeholder_p,
        "potassium": placeholder_k,
        "ph": placeholder_ph,
        "source": "open-meteo (N/P/K & pH are placeholders)",
    }

    return jsonify(result)

@app.route('/api/planting_window')
def planting_window():
    """Return a planting window assessment for a given crop and location.
    Query params: crop (string), lat, lon
    Uses Open-Meteo 16-day daily forecast (temperature_2m_max/min, precipitation_sum).
    """
    crop = request.args.get('crop', type=str)
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if not crop or lat is None or lon is None:
        return jsonify({'error': 'Missing crop, lat, or lon'}), 400
    # Daily forecast endpoint
    meteo_url = (
        'https://api.open-meteo.com/v1/forecast'
        f'?latitude={lat}&longitude={lon}'
        '&daily=temperature_2m_max,temperature_2m_min,precipitation_sum'
        '&forecast_days=16&timezone=auto'
    )
    try:
        r = requests.get(meteo_url, timeout=10)
        r.raise_for_status()
        d = r.json()
    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Forecast fetch failed', 'details': str(e)}), 502
    daily = d.get('daily', {})
    dates = daily.get('time', [])
    tmax = daily.get('temperature_2m_max', [])
    tmin = daily.get('temperature_2m_min', [])
    rain = daily.get('precipitation_sum', [])
    forecast_rows = []
    for idx in range(min(len(dates), 14)):  # Use first 14 days
        try:
            t_mean = (tmax[idx] + tmin[idx]) / 2.0
            forecast_rows.append({
                'date': dates[idx],
                't_mean': t_mean,
                'rain_mm': rain[idx] if idx < len(rain) else 0.0
            })
        except Exception:
            continue
    assessment = evaluate_planting_window(crop, forecast_rows)
    return jsonify({
        'crop': crop,
        'lat': lat,
        'lon': lon,
        'assessment': assessment
    })
if __name__ == "__main__":
    app.run(debug=True)