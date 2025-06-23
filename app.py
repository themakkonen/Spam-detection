from flask import Flask, render_template, request, session, send_file, jsonify
import joblib
from sentence_transformers import SentenceTransformer
import pandas as pd
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key'

simcse_model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')
classifier = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    history = session.get('history', [])

    if request.method == 'POST':
        message = request.form['message']
        embedding = simcse_model.encode([message])
        prediction = classifier.predict(embedding)[0]
        probability = classifier.predict_proba(embedding)[0][1] * 100
        result = "Spam Message ❌" if prediction == 1 else "Not Spam ✅"
        confidence = f"Confidence: {probability:.2f}%"
        history.append({'msg': message, 'res': result, 'conf': f"{probability:.2f}%"})
        session['history'] = history

    return render_template('index.html', result=result, confidence=confidence, history=history)

@app.route('/predict-live', methods=['POST'])
def predict_live():
    message = request.json.get('message', '')
    if not message.strip():
        return jsonify({'prediction': '', 'confidence': ''})
    embedding = simcse_model.encode([message])
    prediction = classifier.predict(embedding)[0]
    probability = classifier.predict_proba(embedding)[0][1] * 100
    result = "Spam Message ❌" if prediction == 1 else "Not Spam ✅"
    confidence = f"{probability:.2f}%"
    return jsonify({'prediction': result, 'confidence': confidence})

@app.route('/download')
def download():
    history = session.get('history', [])
    df = pd.DataFrame(history)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='spam_history.csv')

@app.route('/clear-history', methods=['POST'])
def clear_history():
    session.pop('history', None)
    session.modified = True
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True)
