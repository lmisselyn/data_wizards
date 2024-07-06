from flask import Flask, request, jsonify
from src.model_usage import NB_classify_sentence

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    result = NB_classify_sentence(text)
    return jsonify({'text': text, 'classification': result})

if __name__ == '__main__':
    app.run(host="0.0.0.0")