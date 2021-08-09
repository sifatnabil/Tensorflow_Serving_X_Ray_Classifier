import json
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
img = load_img('view1_frontal.jpg', target_size=(256, 256))
img = img_to_array(img)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))


input_data_json = json.dumps({
    "signature_name":"serving_default",
    "instances":img.tolist()
})

import requests
SERVER_URL = "http://localhost:8501/v1/models/saved_model:predict"
# Note: Again replace your saved model name with saved_model. Rest all remains some
response = requests.post(SERVER_URL,data=input_data_json)
response.raise_for_status()
response = response.json()
y_prob = np.array(response["predictions"])

print(y_prob)