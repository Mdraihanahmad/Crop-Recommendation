from flask import Flask,request,render_template,send_from_directory
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():

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

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
        # Save recommendation as PDF and text
        pdf_path = os.path.join('static', 'downloads', 'recommendation.pdf')
        text_path = os.path.join('static', 'downloads', 'recommendation.txt')
        # PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.drawString(100, 700, result)
        c.save()
        # Text
        with open(text_path, 'w') as f:
            f.write(result)
        download_links = {
            'pdf': '/static/downloads/recommendation.pdf',
            'txt': '/static/downloads/recommendation.txt'
        }
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        download_links = None
    return render_template('index.html', result=result, download_links=download_links)




# python main
if __name__ == "__main__":
    app.run(debug=True)