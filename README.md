# Photo Indexer

A Python-based facial recognition system that indexes photos, uploads them to MinIO object storage, and enables searching for similar faces using facial embeddings. The system provides both a REST API and command-line interface for flexibility.

## Features

- **Face Detection & Recognition**: Automatically detects and encodes faces in photos using the `face_recognition` library
- **REST API**: FastAPI-based web service for photo indexing and face matching
- **Command-line Tools**: Standalone scripts for batch processing and searching
- **Cloud Storage**: Uploads photos to MinIO (S3-compatible object storage)
- **Face Search**: Search for photos containing similar faces by providing a reference image
- **Efficient Indexing**: Stores facial embeddings and metadata in JSON format for fast searching
- **Automatic Backup**: Creates backup copies of index data before updates

## Project Structure

```
photo-indexer/
├── main.py                    # FastAPI web service
├── index_photos_script.py     # Standalone photo indexing script
├── search.py                  # Command-line face search tool
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # MinIO setup
├── data/                      # Directory containing photos to index
├── indexed_data.json          # Generated face index file
├── indexed_data.json.backup   # Backup of previous index
└── README.md                  # This file
```

## Prerequisites

- Python 3.7+
- Docker and Docker Compose (for MinIO)
- CMake (required for dlib installation)

### Installing CMake

**macOS:**
```bash
brew install cmake
```

**Ubuntu/Debian:**
```bash
sudo apt-get install cmake
```

**Windows:**
Download and install from [CMake official website](https://cmake.org/download/)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd photo-indexer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MinIO:**
   ```bash
   docker-compose up -d
   ```

5. **Create environment variables:**
   Create a `.env` file in the project root:
   ```env
   MINIO_ENDPOINT=http://localhost:9005
   MINIO_ACCESS_KEY=admin
   MINIO_SECRET_KEY=secret123
   MINIO_BUCKET=photos
   ```

## Usage

### Option 1: REST API Service

Start the FastAPI web service:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### API Endpoints

- `GET /` - Health check endpoint
- `GET /health` - Detailed health status
- `GET /debug/minio` - MinIO connection diagnostics
- `POST /index-photos` - Trigger photo indexing via API
- `POST /api/find-matches` - Find matching faces (accepts base64 image)

#### Example API Usage

```bash
# Trigger photo indexing
curl -X POST "http://localhost:8000/index-photos"

# Find matches (with base64 encoded image)
curl -X POST "http://localhost:8000/api/find-matches" \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}'
```

### Option 2: Command-Line Tools

#### 1. Index Photos (Standalone Script)

Place your photos in the `data/` directory and run:

```bash
python index_photos_script.py
```

This will:
- Upload all photos to MinIO
- Detect faces in each photo
- Generate facial embeddings
- Save the index to `indexed_data.json`
- Create a backup of the previous index

#### 2. Search for Similar Faces

Use the command-line search tool with a reference image:

```bash
python search.py path/to/your/selfie.jpg
```

The script will:
- Analyze the reference image
- Compare it against all indexed faces
- Return photos containing similar faces

## Configuration

### Match Threshold

You can adjust the face matching sensitivity:

**In `search.py` (command-line):**
```python
MATCH_THRESHOLD = 0.6  # Lower values = stricter matching
```

**In `main.py` (API service):**
```python
SIMILARITY_THRESHOLD = 0.5  # API uses slightly stricter default
```

- **0.4-0.5**: Very strict (recommended for high accuracy)
- **0.6**: Balanced (command-line default)
- **0.7-0.8**: More lenient (may include false positives)

### MinIO Configuration

The MinIO instance can be accessed at:
- **API Endpoint**: http://localhost:9005
- **Web UI**: http://localhost:9006
- **Credentials**: admin / secret123

## Dependencies

- `fastapi`: Modern web framework for building APIs
- `uvicorn`: ASGI server for FastAPI
- `face-recognition`: Face detection and encoding
- `boto3`: AWS S3/MinIO client
- `python-dotenv`: Environment variable management
- `opencv-python`: Image processing
- `dlib`: Face detection backend
- `numpy`: Numerical computations
- `pillow`: Image processing library

## How It Works

1. **Face Detection**: Uses HOG (Histogram of Oriented Gradients) or CNN models to detect faces
2. **Face Encoding**: Generates 128-dimensional face embeddings using deep learning
3. **Storage**: Photos are stored in MinIO, embeddings in JSON
4. **Matching**: Uses Euclidean distance to compare face embeddings
5. **Confidence Scoring**: API provides intuitive confidence scores (0-100%)

## Output Format

The `indexed_data.json` file contains:

```json
[
  {
    "photo": "example.jpg",
    "face_id": "example.jpg_face0",
    "embedding": [0.1, -0.2, 0.3, ...],
    "bounding_box": [top, right, bottom, left]
  }
]
```

## API Response Format

The API returns matches with confidence scores:

```json
{
  "matches": [
    {
      "photoUrl": "https://minio-signed-url",
      "faceId": "photo.jpg_face0",
      "boundingBox": [top, right, bottom, left],
      "confidence": 0.85
    }
  ],
  "summary": {
    "totalMatches": 1,
    "highConfidenceMatches": 1,
    "processingTimeMs": 245
  }
}
```

## Development

### Running in Development Mode

```bash
# Start MinIO
docker-compose up -d

# Start API server with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
open http://localhost:8000/docs
```

### Testing

Test the API endpoints using the built-in FastAPI documentation at `http://localhost:8000/docs` or use curl/Postman.

## Troubleshooting

### Common Issues

1. **No faces detected**: Ensure photos have clear, visible faces
2. **dlib installation fails**: Install CMake and ensure you have sufficient RAM
3. **MinIO connection error**: Check if Docker container is running and ports are available
4. **Memory issues**: Large photos may require more RAM; consider resizing images
5. **API base64 errors**: Ensure images are properly base64 encoded with correct format

### Performance Tips

- Use smaller image sizes for faster processing
- Consider using GPU acceleration for CNN face detection
- Batch process large photo collections using the standalone script
- Monitor MinIO storage usage and clean up old files if needed

### Debugging

Use the debug endpoints to troubleshoot issues:
- `GET /debug/minio` - Check MinIO connectivity
- `GET /health` - Overall system health

## License

This project is open source. Please check the license file for more details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

If you encounter any issues or have questions, please open an issue on the repository. 