from flask import Flask, request, jsonify
from flask.logging import create_logger
from werkzeug.utils import secure_filename
import logging

import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from ml_model.transfer_style import model_selection, run_nst, save_nst_image

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

def scale(payload):
    """Scales Payload"""

    LOG.info(f"Scaling Payload: {payload}")
    scaler = StandardScaler().fit(payload)
    scaled_adhoc_predict = scaler.transform(payload)
    return scaled_adhoc_predict

@app.route("/")
def home():
    html = f"<h3>Style transfer Home</h3>"
    return html.format(format)

# TO DO:  Log out the prediction value
@app.route("/style_transfer", methods=['POST'])
def style_transfer():
    """Performs an style transfer

    inputs are

    """
    base_image = request.files['image_file']
    style_image = request.files['style_file']

    

    filename = secure_filename(file.filename)
    LOG.info(f"img filename: {json_payload}")
    inference_payload = pd.DataFrame(json_payload)
    LOG.info(f"inference payload DataFrame: {inference_payload}")
    scaled_payload = scale(inference_payload)
    prediction = list(clf.predict(scaled_payload))
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    clf = joblib.load("boston_housing_prediction.joblib")
    app.run(host='0.0.0.0', port=80, debug=True)
