from flask import Flask, jsonify
from flask_cors import CORS

from routes.upload import upload_bp


app = Flask(__name__)

CORS(app)


# Register upload route
app.register_blueprint(
    upload_bp
)


@app.route("/")
def home():

    return jsonify({
        "status": "success",
        "message":
            "Brain Tumor Detection API is running"
    })


@app.route("/health")
def health():

    return jsonify({
        "status": "healthy"
    })


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )