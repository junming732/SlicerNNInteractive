import requests
import numpy as np
import io

SERVER_URL = "http://localhost:1527"

# Create test volume
image = np.random.randn(100, 100, 100).astype(np.float32)

# Upload
print("1. Uploading image...")
buffer = io.BytesIO()
np.save(buffer, image)
buffer.seek(0)

response = requests.post(
    f"{SERVER_URL}/upload_image",
    files={"file": ("volume.npy", buffer, "application/octet-stream")},
    timeout=60
)
print(f"   Upload response: {response.json()}")

# Test SAM-Med3D endpoint
print("\n2. Testing SAM-Med3D segmentation...")
response = requests.post(
    f"{SERVER_URL}/add_fastsam3d_interaction",
    json={
        "voxel_coord": [50, 50, 50],
        "positive_click": True
    },
    timeout=30
)

print(f"   Status: {response.status_code}")
print(f"   Response size: {len(response.content)} bytes")
print(f"   Content preview: {response.content[:20]}")

if response.status_code == 200:
    print("\n✓ SUCCESS! SAM-Med3D is working!")
else:
    print(f"\n✗ FAILED: {response.text}")
