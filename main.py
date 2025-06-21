import os
import json
import base64
import io
import numpy as np
import face_recognition
import boto3
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Photo Indexer API", description="API for photo indexing and face recognition")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response
class FindMatchesRequest(BaseModel):
    image: str  # base64-encoded image

class MatchResult(BaseModel):
    photoUrl: str
    faceId: str
    boundingBox: List[int]
    confidence: float

class FindMatchesResponse(BaseModel):
    matches: List[MatchResult]
    summary: Dict[str, Any]

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

PHOTO_DIR = "data"
OUTPUT_FILE = "indexed_data.json"
SIMILARITY_THRESHOLD = 0.5

# Setup MinIO client
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

def validate_minio_connection():
    """Validate MinIO connection and credentials"""
    try:
        # Test basic connection
        s3.list_buckets()
        
        # Test bucket access
        try:
            s3.head_bucket(Bucket=MINIO_BUCKET)
            print(f"✅ MinIO connection validated - Bucket '{MINIO_BUCKET}' accessible")
        except:
            print(f"🔧 Creating bucket '{MINIO_BUCKET}'")
            s3.create_bucket(Bucket=MINIO_BUCKET)
            print(f"✅ Created bucket '{MINIO_BUCKET}'")
            
        return True
    except Exception as e:
        print(f"❌ MinIO connection failed: {str(e)}")
        print("🔧 Check your MinIO credentials and endpoint configuration")
        return False

# Validate MinIO connection at startup
validate_minio_connection()

def load_face_index():
    """Load the face index from JSON file"""
    try:
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image string to numpy array"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image then to numpy array
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        return np.array(pil_image)
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Invalid image format: {str(e)}")

def extract_face_encoding(image: np.ndarray) -> tuple:
    """Extract face encoding from image"""
    try:
        # Detect face locations
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="Could not encode face in the uploaded image")
        
        # Return the first face encoding and location
        return face_encodings[0], face_locations[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face encoding failed: {str(e)}")

def calculate_face_distance(encoding1: np.ndarray, encoding2: list) -> float:
    """Calculate Euclidean distance between face encodings"""
    return np.linalg.norm(np.array(encoding1) - np.array(encoding2))

def calculate_confidence_score(distance: float, threshold: float = SIMILARITY_THRESHOLD) -> float:
    """
    Calculate confidence score from face distance with intuitive scaling
    
    Examples with threshold=0.5:
    - Distance 0.0 (perfect match) -> 100% confidence
    - Distance 0.1 (excellent match) -> ~96% confidence  
    - Distance 0.2 (very good match) -> ~85% confidence
    - Distance 0.3 (good match) -> ~69% confidence
    - Distance 0.4 (decent match) -> ~49% confidence
    - Distance 0.5 (threshold) -> ~15% confidence (still a valid match!)
    """
    if distance > threshold:
        return 0.0
    
    # Map threshold distance to 15% confidence instead of 0%
    # This acknowledges that threshold matches are still valid
    min_confidence = 0.15
    
    # Use power function for intuitive scaling
    # Scale from 100% (distance=0) to 15% (distance=threshold)
    normalized_distance = distance / threshold
    confidence_range = 1.0 - min_confidence
    confidence = 1.0 - (confidence_range * (normalized_distance ** 1.5))
    
    return max(min_confidence, min(1.0, confidence))

def generate_signed_url(filename: str, expiration: int = 315360000) -> str:
    """Generate signed URL for MinIO object with improved error handling"""
    try:
        # First, verify the object exists
        try:
            s3.head_object(Bucket=MINIO_BUCKET, Key=filename)
        except Exception as head_error:
            print(f"⚠️ Object {filename} not found in bucket: {str(head_error)}")
            # Return fallback URL even if object doesn't exist
            return f"{MINIO_ENDPOINT}/{MINIO_BUCKET}/{filename}"
        
        # Generate signed URL with proper error handling
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': MINIO_BUCKET, 'Key': filename},
            ExpiresIn=expiration,
            HttpMethod='GET'
        )
        
        # Validate the generated URL
        if not url or not url.startswith('http'):
            raise Exception("Invalid signed URL generated")
            
        return url
        
    except Exception as e:
        print(f"❌ Error generating signed URL for {filename}: {str(e)}")
        print(f"🔧 Using fallback URL instead")
        
        # Enhanced fallback with proper URL encoding
        import urllib.parse
        encoded_filename = urllib.parse.quote(filename, safe='')
        fallback_url = f"{MINIO_ENDPOINT}/{MINIO_BUCKET}/{encoded_filename}"
        return fallback_url

@app.get("/")
async def hello_world():
    """Hello World route handler"""
    return {"message": "Hello World from Photo Indexer API!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "photo-indexer"}

@app.get("/debug/minio")
async def debug_minio():
    """Debug MinIO connection and list objects"""
    try:
        # Test connection
        buckets = s3.list_buckets()
        
        # List objects in bucket
        objects = []
        try:
            response = s3.list_objects_v2(Bucket=MINIO_BUCKET, MaxKeys=10)
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Test signed URL generation for each object
                    try:
                        signed_url = generate_signed_url(obj['Key'])  # Use default long expiry
                        objects.append({
                            "key": obj['Key'],
                            "size": obj['Size'],
                            "last_modified": obj['LastModified'].isoformat(),
                            "signed_url": signed_url,
                            "signed_url_status": "success"
                        })
                    except Exception as url_error:
                        objects.append({
                            "key": obj['Key'],
                            "size": obj['Size'],
                            "last_modified": obj['LastModified'].isoformat(),
                            "signed_url": None,
                            "signed_url_status": f"error: {str(url_error)}"
                        })
        except Exception as list_error:
            return {
                "status": "error",
                "message": f"Cannot list bucket contents: {str(list_error)}",
                "bucket": MINIO_BUCKET
            }
        
        return {
            "status": "success",
            "minio_endpoint": MINIO_ENDPOINT,
            "bucket": MINIO_BUCKET,
            "bucket_count": len(buckets.get('Buckets', [])),
            "objects_in_bucket": len(objects),
            "sample_objects": objects[:5],  # Show first 5 objects
            "connection": "healthy"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"MinIO connection failed: {str(e)}",
            "minio_endpoint": MINIO_ENDPOINT,
            "bucket": MINIO_BUCKET
        }

@app.post("/api/find-matches", response_model=FindMatchesResponse)
async def find_matches(request: FindMatchesRequest):
    """Find matching faces in indexed photos"""
    try:
        # Load face index
        face_index = load_face_index()
        
        if not face_index:
            raise HTTPException(status_code=500, detail="No face index found. Please index photos first.")
        
        # Decode and process input image
        image = decode_base64_image(request.image)
        input_encoding, input_location = extract_face_encoding(image)
        
        # Find matches
        matches = []
        total_faces_considered = len(face_index)
        
        for face_data in face_index:
            # Calculate similarity
            distance = calculate_face_distance(input_encoding, face_data["embedding"])
            
            if distance < SIMILARITY_THRESHOLD:
                # Generate signed URL
                photo_url = generate_signed_url(face_data["photo"])
                
                # Calculate confidence using the improved scoring function
                confidence = calculate_confidence_score(distance)
                
                match = MatchResult(
                    photoUrl=photo_url,
                    faceId=face_data["face_id"],
                    boundingBox=face_data["bounding_box"],
                    confidence=round(confidence, 2)
                )
                matches.append(match)
        
        # Sort matches by confidence (highest first)
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        # Prepare response
        response = FindMatchesResponse(
            matches=matches,
            summary={
                "totalMatchedPhotos": len(matches),
                "totalFacesConsidered": total_faces_considered,
                "matchingThreshold": SIMILARITY_THRESHOLD
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def index_photos():
    """Function to index photos - moved from main execution"""
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
    return face_index

@app.post("/index-photos")
async def trigger_photo_indexing():
    """Endpoint to trigger photo indexing"""
    try:
        face_index = index_photos()
        return {
            "message": "Photo indexing completed successfully",
            "faces_indexed": len(face_index),
            "status": "success"
        }
    except Exception as e:
        return {
            "message": f"Photo indexing failed: {str(e)}",
            "status": "error"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
