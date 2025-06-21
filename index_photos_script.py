#!/usr/bin/env python3
"""
Standalone Photo Indexing Script

This script processes photos in the data directory, uploads them to MinIO,
extracts face encodings, and saves the index to indexed_data.json.

Usage:
    python index_photos_script.py
"""

import os
import json
import face_recognition
import boto3
from dotenv import load_dotenv
import sys
from pathlib import Path

def load_environment():
    """Load environment variables and validate required settings"""
    load_dotenv()
    
    required_vars = [
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY", 
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file or environment settings.")
        sys.exit(1)
    
    return {
        "endpoint": os.getenv("MINIO_ENDPOINT"),
        "access_key": os.getenv("MINIO_ACCESS_KEY"),
        "secret_key": os.getenv("MINIO_SECRET_KEY"),
        "bucket": os.getenv("MINIO_BUCKET")
    }

def setup_minio_client(config):
    """Setup and test MinIO client connection"""
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=config["endpoint"],
            aws_access_key_id=config["access_key"],
            aws_secret_access_key=config["secret_key"]
        )
        
        # Test connection and ensure bucket exists
        try:
            s3.head_bucket(Bucket=config["bucket"])
            print(f"✅ Connected to MinIO bucket: {config['bucket']}")
        except:
            print(f"🔧 Creating MinIO bucket: {config['bucket']}")
            s3.create_bucket(Bucket=config["bucket"])
            print(f"✅ Created MinIO bucket: {config['bucket']}")
        
        return s3
    except Exception as e:
        print(f"❌ Error connecting to MinIO: {str(e)}")
        sys.exit(1)

def get_image_files(photo_dir):
    """Get list of image files from photo directory"""
    if not os.path.exists(photo_dir):
        print(f"❌ Error: Photo directory '{photo_dir}' does not exist")
        sys.exit(1)
    
    supported_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    image_files = []
    
    for filename in os.listdir(photo_dir):
        if filename.lower().endswith(supported_extensions):
            image_files.append(filename)
    
    if not image_files:
        print(f"⚠️  No image files found in '{photo_dir}'")
        return []
    
    print(f"📸 Found {len(image_files)} image files to process")
    return image_files

def upload_to_minio(s3, photo_path, filename, bucket):
    """Upload photo to MinIO"""
    try:
        with open(photo_path, "rb") as f:
            s3.upload_fileobj(f, bucket, filename)
        print(f"📤 Uploaded {filename} to MinIO")
        return True
    except Exception as e:
        print(f"❌ Error uploading {filename}: {str(e)}")
        return False

def extract_face_data(photo_path, filename):
    """Extract face encodings and locations from photo"""
    try:
        # Load image
        image = face_recognition.load_image_file(photo_path)
        
        # Detect face locations
        locations = face_recognition.face_locations(image)
        
        if not locations:
            print(f"ℹ️  No faces detected in {filename}")
            return []
        
        print(f"👤 Found {len(locations)} face(s) in {filename}")
        
        face_data_list = []
        
        for i, (top, right, bottom, left) in enumerate(locations):
            # Extract face encoding
            encoding = face_recognition.face_encodings(
                image, 
                known_face_locations=[(top, right, bottom, left)]
            )
            
            if not encoding:
                print(f"⚠️  Could not encode face {i} in {filename}, skipping.")
                continue
            
            face_data = {
                "photo": filename,
                "face_id": f"{filename}_face{i}",
                "embedding": encoding[0].tolist(),
                "bounding_box": [top, right, bottom, left]
            }
            
            face_data_list.append(face_data)
            print(f"✅ Encoded face {i} in {filename}")
        
        return face_data_list
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")
        return []

def save_face_index(face_index, output_file):
    """Save face index to JSON file"""
    try:
        # Create backup if file exists
        if os.path.exists(output_file):
            backup_file = f"{output_file}.backup"
            os.rename(output_file, backup_file)
            print(f"📋 Created backup: {backup_file}")
        
        # Save new index
        with open(output_file, "w") as f:
            json.dump(face_index, f, indent=2)
        
        print(f"💾 Saved face index to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving face index: {str(e)}")
        return False

def main():
    """Main function to orchestrate photo indexing"""
    print("🚀 Starting Photo Indexing Script")
    print("=" * 50)
    
    # Configuration
    PHOTO_DIR = "data"
    OUTPUT_FILE = "indexed_data.json"
    
    # Load environment and setup MinIO
    config = load_environment()
    s3 = setup_minio_client(config)
    
    # Get image files
    image_files = get_image_files(PHOTO_DIR)
    if not image_files:
        return
    
    # Process each image
    face_index = []
    processed_count = 0
    uploaded_count = 0
    
    print(f"\n📁 Processing images from '{PHOTO_DIR}'...")
    print("-" * 30)
    
    for filename in image_files:
        print(f"\n🖼️  Processing: {filename}")
        photo_path = os.path.join(PHOTO_DIR, filename)
        
        # Upload to MinIO
        if upload_to_minio(s3, photo_path, filename, config["bucket"]):
            uploaded_count += 1
        
        # Extract face data
        face_data_list = extract_face_data(photo_path, filename)
        face_index.extend(face_data_list)
        
        processed_count += 1
    
    # Save results
    print("\n" + "=" * 50)
    print("📊 INDEXING SUMMARY")
    print("=" * 50)
    
    if save_face_index(face_index, OUTPUT_FILE):
        print(f"✅ Successfully indexed {len(face_index)} faces")
        print(f"📸 Processed {processed_count}/{len(image_files)} images")
        print(f"📤 Uploaded {uploaded_count}/{len(image_files)} images to MinIO")
        print(f"💾 Results saved to: {OUTPUT_FILE}")
    else:
        print("❌ Failed to save face index")
        sys.exit(1)
    
    print("\n🎉 Photo indexing completed successfully!")

if __name__ == "__main__":
    main() 