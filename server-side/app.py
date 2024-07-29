from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from config import SERVER, DATABASE, USERNAME, PASSWORD, DRIVER
from utils import load_pretrained_model, preprocess_input_image
from user_face_management import get_user_encode_from_db, upload_user_data, read_image
import torch
import pyodbc
import os

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Connect to database
server = SERVER 
database = DATABASE
username = USERNAME
password = PASSWORD
driver = DRIVER

connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
connection = pyodbc.connect(connection_string)
cursor = connection.cursor()

# Load facial verification model
model = load_pretrained_model("FacialVerification.pth")

@app.route("/api/verify", methods=["POST"])
def verify_user():
    """
    Verify a user's identity based on the provided image and user encoding
    """
    data = request.get_json()
    image = data["image"]
    user_encode = torch.tensor(data['user_encode'])
    image = read_image(image=image)
    image_preprocessed = preprocess_input_image(image)
    extract = model.extract_feature(image_preprocessed)
    prob = model.verify(image_preprocessed,user_encode)
    print(prob)
    return jsonify({"probability" : prob.item()})

@app.route("/api/get_user_encode", methods=["POST"])
def get_user_encode():
    """
    Retrieve a user's face encoding from the database
    """
    data = request.get_json()
    user_encode = get_user_encode_from_db(cursor,data["user"])
    if user_encode:
        return jsonify(user_encode)
    else:
        return jsonify({"error": "User encode not found"}), 404

@app.route("/api/get_user_embedding", methods=["POST"])
def extract_feature_user():
    """
    Extract a user's face embedding from an uploaded image
    """
    data = request.get_json()
    image = data["image"]
    image = read_image(image=image)
    image_preprocessed = preprocess_input_image(image)
    image_embedding = model.extract_feature(image_preprocessed)
    encoded_list = image_embedding.tolist()
    return jsonify(encoded_list)

@app.route("/api/upload_data", methods=["POST"])
def upload_data():
    """Upload user data to the database"""
    data = request.get_json()
    res = upload_user_data(cursor=cursor,connection=connection,user=data["user"], encoded_user_img=data["encoded_user_img"])
    return jsonify(res)

@app.route("/")
def index():
    """
    Serve the index route
    """
    return "hello world"

if __name__ == "__main__":
    app.run(debug=True, port=4000)