import face_recognition
import json
import sys
import os

# --- Config ---
INDEX_FILE = "indexed_data.json"
MATCH_THRESHOLD = 0.6  # Smaller is stricter (0.6 is typical upper limit)


def load_selfie_embedding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        raise ValueError("No face found in the selfie image.")
    if len(encodings) > 1:
        raise ValueError("Multiple faces found. Please provide a clear selfie with one face.")

    return encodings[0]


def load_face_index():
    with open(INDEX_FILE, "r") as f:
        return json.load(f)


def compare_embeddings(selfie_embedding, index):
    matched_photos = set()

    for face_data in index:
        indexed_embedding = face_data["embedding"]
        distance = face_recognition.face_distance([indexed_embedding], selfie_embedding)[0]
        print(f"Distance to {face_data['face_id']}: {distance:.4f}")

        if distance <= MATCH_THRESHOLD:
            matched_photos.add(face_data["photo"])

    return list(matched_photos)


def main(selfie_path):
    if not os.path.isfile(selfie_path):
        print(f"File not found: {selfie_path}")
        return

    print("🔍 Generating embedding for selfie...")
    selfie_embedding = load_selfie_embedding(selfie_path)

    print("📂 Loading face index...")
    face_index = load_face_index()

    print("🧠 Comparing embeddings...")
    matches = compare_embeddings(selfie_embedding, face_index)

    if matches:
        print("✅ Found matches in the following photos:")
        for photo in matches:
            print(f"  - {photo}")
    else:
        print("❌ No matching photos found.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python search_selfie.py <path_to_selfie_image>")
    else:
        main(sys.argv[1])
