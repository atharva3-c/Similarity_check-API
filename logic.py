from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
Model = tf.keras.models.Model
VGG16 = tf.keras.applications.VGG16
preprocess_input = tf.keras.applications.vgg16.preprocess_input
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
import tempfile
import os
from dotenv import load_dotenv

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Load environment variables from .env file
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

app = FastAPI()

# Load VGG16 model with ImageNet weights and extract features from 'block5_pool' layer
base_model = VGG16(weights='imagenet', include_top=False)  # Exclude fully connected layers
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

class VideoBytes(BaseModel):
    video_data: bytes

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def extract_frames_from_bytes(video_bytes, frame_interval=1):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        count += 1

    cap.release()
    os.remove(temp_video_path)
    return np.array(frames)

def extract_feature_vector(video_bytes, frame_interval=1):
    frames = extract_frames_from_bytes(video_bytes, frame_interval)
    features = []

    for frame in frames:
        frame = np.expand_dims(frame, axis=0)
        frame = preprocess_input(frame)
        feature = model.predict(frame)
        features.append(feature.flatten())

    return np.array(features)

def get_all_feature_vectors():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT video_id, feature_vector FROM video_features")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return [(video_id, np.array(feature_vector).reshape(11, 4096)) for video_id, feature_vector in results]

def insert_new_feature_vector(feature_matrix):
    flattened_vector = feature_matrix.flatten().tolist()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO video_features (feature_vector)
        VALUES (%s)
        RETURNING video_id
        """,
        (flattened_vector,)
    )
    video_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    return video_id

def compare_with_fixed_vector(Fv_fixed, threshold=0.5):
    all_feature_vectors = get_all_feature_vectors()
    similarities = []

    for video_id, feature_vector in all_feature_vectors:
        frame_similarities = [cosine_similarity([f_fixed], [f])[0][0] for f_fixed, f in zip(Fv_fixed, feature_vector)]
        average_similarity = np.mean(frame_similarities)
        similarities.append((video_id, average_similarity))

    max_video_id, max_similarity = max(similarities, key=lambda x: x[1])

    if max_similarity < threshold:
        new_video_id = insert_new_feature_vector(Fv_fixed)
        return new_video_id, max_similarity 
    else:
        return max_video_id, max_similarity

@app.post("/compare-video-bytes/")
async def compare_video(video: VideoBytes):
    try:
        features_fixed = extract_feature_vector(video.video_data, frame_interval=30)
        video_id, max_similarity = compare_with_fixed_vector(features_fixed)
        return {"video_id": video_id, "max_similarity": max_similarity}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
