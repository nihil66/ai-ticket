from flask import Flask, request, jsonify
import joblib

# Load trained model and vectorizer
clf = joblib.load("ticket_classifier_pipeline.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route('/predictTicket', methods=['POST'])
def predict_ticket():
    try:
        data = request.get_json()
        ticket_text = data.get("text", "")

        if not ticket_text:
            return jsonify({"error": "No text provided"}), 400

        # Transform input and predict
        input_vec = vectorizer.transform([ticket_text])
        prediction = clf.predict(input_vec)[0]

        return jsonify({"predicted_category": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
