from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Allow React frontend to communicate with Flask
CORS(app)


@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Brain Tumor Detection API is running"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)