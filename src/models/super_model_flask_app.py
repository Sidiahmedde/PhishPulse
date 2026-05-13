from pathlib import Path

from flask import Flask, jsonify, render_template, request

try:
    from .super_module import super_module
except ImportError:
    from super_module import super_module

_ROOT = Path(__file__).resolve().parents[2]

app = Flask(
    __name__,
    template_folder=str(_ROOT / "templates"),
    static_folder=str(_ROOT / "static"),
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/send", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    return jsonify(super_module(data))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
