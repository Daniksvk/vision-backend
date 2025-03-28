from flask import Flask, request, jsonify
from google.cloud import vision
import os
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # pre frontend z iného pôvodu

# Cesta k JSON kľúču
import json

# z environment premennej
key_json = os.environ.get("GOOGLE_SERVICE_KEY")

if not key_json:
    raise Exception("Missing GOOGLE_SERVICE_KEY environment variable")

key_dict = json.loads(key_json)

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_info(key_dict)

client = vision.ImageAnnotatorClient(credentials=credentials)


client = vision.ImageAnnotatorClient()

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    content = file.read()

    image = vision.Image(content=content)
    objects = client.object_localization(image=image).localized_object_annotations

    results = []
    for obj in objects:
        box = [{
            'x': vertex.x,
            'y': vertex.y
        } for vertex in obj.bounding_poly.normalized_vertices]

        results.append({
            'name': obj.name,
            'score': obj.score,
            'bounding_box': box
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
