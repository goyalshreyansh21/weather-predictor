import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model.h5')

labels = ['Blizzard', 'Cloudy', 'Fog', 'Heavy rain', 'Light rain', 'Light sleet', 'Mist', 'Moderate or heavy sleet', 'Moderate or heavy snow showers', 'Moderate rain', 'Moderate snow', 'Overcast', 'Partly cloudy', 'Sunny', 'Torrential rain shower']

images = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png', '10.png', '11.png', '12.png', '13.png', '14.png', '15.png']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature = [int(x) for x in request.form.values()]
    feature = np.array(feature).reshape(1, -1)
    prediction = np.argmax(model.predict(feature))  
    predicted_weather = labels[prediction]
    image_url = f"../static/img/{images[prediction]}"
    print(image_url)
    return render_template('home.html',prediction_text='Predicted Weather: {}'.format(predicted_weather), image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)