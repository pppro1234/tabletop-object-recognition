# API

## Endpoints

- `GET /healthz`
  - Returns service liveness.
- `POST /predict`
  - Request: `{ "image_path": "..." }`
  - Response shape: list of detections with `track_id`, `label`, `score`, `centroid`, `angle_deg`.

The FastAPI app is currently a scaffold. The next implementation step is to bind it to the C++ pipeline executable or shared library.
