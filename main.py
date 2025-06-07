import os
import json
import face_recognition
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

PHOTO_DIR = "data"
OUTPUT_FILE = "indexed_data.json"

# Setup MinIO client
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

# Ensure bucket exists
try:
    s3.head_bucket(Bucket=MINIO_BUCKET)
except:
    s3.create_bucket(Bucket=MINIO_BUCKET)

face_index = []

# Loop through photos
for filename in os.listdir(PHOTO_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    photo_path = os.path.join(PHOTO_DIR, filename)

    # Upload to MinIO
    with open(photo_path, "rb") as f:
        s3.upload_fileobj(f, MINIO_BUCKET, filename)
        print(f"Uploaded {filename} to MinIO")

    # Detect and encode faces
    image = face_recognition.load_image_file(photo_path)
    locations = face_recognition.face_locations(image)    

    for i, (top, right, bottom, left) in enumerate(locations):
        encoding = face_recognition.face_encodings(image, known_face_locations=[(top, right, bottom, left)])

        if not encoding:
            print(f"⚠️ Could not encode face {i} in {filename}, skipping.")
            continue

        face_data = {
            "photo": filename,
            "face_id": f"{filename}_face{i}",
            "embedding": encoding[0].tolist(),
            "bounding_box": [top, right, bottom, left]
        }

        face_index.append(face_data)

# Save index
with open(OUTPUT_FILE, "w") as f:
    json.dump(face_index, f, indent=2)

print(f"\n✅ Indexed {len(face_index)} faces from {len(os.listdir(PHOTO_DIR))} images.")
