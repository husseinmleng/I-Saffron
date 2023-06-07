from flask import Flask, request, jsonify, render_template
from utils import get_result,img_to_base64,get_img_array
import io
import traceback
from PIL import Image
from flask_cors import CORS
import os
# Create Flask app  
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello World"

@app.route('/index')
def index():
    return app.send_static_file('index.html')

# Define route to accept image file
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded image file
        _file = request.files['image']
        # Load image and preprocess
        img = _file.read()
        if img is None:
            response = {'error': 'Unable to decode image'}
            return jsonify(response), 400
        
        # Make prediction and generate GradCAM heatmap
        result,sup_img = get_result(img)

        # Return response as JSON
        label = 'Saffron' if result[0][0] > 0.5 else 'Non-Saffron'
        saffron_prob = round(result[0][0]*100, 2)
        non_saffron_prob = round(100-saffron_prob,2)
        # convert image to pil image
        img = Image.open(io.BytesIO(img))
      
        response = {
                    'predicted_class': label,
                    'saffron_probability': saffron_prob,
                    'non_saffron_probability': non_saffron_prob,
                    'gradcam_image':  img_to_base64(sup_img),
                    'input_image': img_to_base64(get_img_array(img)),
                   }
        # render the json response in the index.html
        return jsonify(response)

    except Exception as e:
        print(f"Error predicting image: {e}")
        traceback.print_exc()
        response = {'error': 'Unable to process image'}
        return jsonify(response), 500

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5050)), debug=False)
