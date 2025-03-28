from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from google.oauth2 import service_account
import os
import json

app = Flask(__name__)
CORS(app)

# Načítanie Google API kľúča z environment premennej
key_json = os.environ.get("GOOGLE_SERVICE_KEY")

if not key_json:
    raise Exception("Missing GOOGLE_SERVICE_KEY environment variable")

key_dict = json.loads(key_json)
credentials = service_account.Credentials.from_service_account_info(key_dict)
client = vision.ImageAnnotatorClient(credentials=credentials)

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
