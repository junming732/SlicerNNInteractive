# Technical Difficulties in SAM-Med3D Integration

## Overview
This document summarizes the core technical challenges encountered while integrating SAM-Med3D into the SlicerNNInteractive server infrastructure for interactive CT segmentation.

---

## 1. Coordinate System Misalignment

### The Problem
Medical imaging involves multiple coordinate systems that must be properly mapped:

- **Slicer's RAS Coordinate System**: Right-Anterior-Superior (physical space)
- **NumPy Array Indexing**: [Depth, Height, Width] or [Z, Y, X]  
- **User Click Coordinates**: [X, Y, Z] from GUI
- **SAM-Med3D Expected Format**: Model's internal coordinate convention

### Technical Impact
Clicks in one coordinate space were being interpreted in another, causing:
- Segmentation appearing 80-100 voxels away from click location
- Scattered segmentation across the volume
- Systematic offsets that persisted despite simple transformations

### Root Cause
The coordinate transformation pipeline had multiple conversion points where order and scaling were not consistently maintained:
```
Slicer GUI [X,Y,Z] → Server [?,?,?] → Scaled [?,?,?] → Model Input [?,?,?]
                   → Model Output [?,?,?] → Rescaled [?,?,?] → Client
```

Each `?` represented uncertainty about whether coordinates were in [X,Y,Z] or [D,H,W] order and whether they were properly scaled.

### Key Insight
Coordinate systems require **complete end-to-end mapping** rather than piecemeal transformations. Debug logging at every transformation step was essential to identify where the mapping broke down.

---

## 2. Architectural Differences: 2D Slices vs 3D Cubes

### nnInteractive Architecture
- **Processing Mode**: 2D or 2.5D (stack of slices)
- **Input**: Works on native resolution axial slices (e.g., 512×512)
- **Coordinate Handling**: Direct mapping to slice indices
- **Spacing Sensitivity**: Only cares about in-plane resolution (X-Y)
- **Why It Works**: Processes where data is naturally uniform

### SAM-Med3D Architecture
- **Processing Mode**: Full 3D volumetric
- **Input**: Requires 128×128×128 cubic regions
- **Coordinate Handling**: Expects isotropic or near-isotropic volumes
- **Spacing Sensitivity**: Affected by all three dimensions
- **Challenge**: Medical images are often anisotropic

### The Anisotropic Problem
Typical CT scan spacing:
```
X: 0.703mm  (512 pixels)
Y: 0.703mm  (512 pixels)  
Z: 5.0mm    (75 slices)
```

When SAM-Med3D crops 128×128×128 voxels:
- Physical size: ~90mm × 90mm × 640mm
- Result: Tall cylinder instead of cube
- Impact: Model trained on cubes can't properly segment cylinders

### Solution
Resample to isotropic spacing (1.5mm×1.5mm×1.5mm) before inference:
- Matches SAM-Med3D training data distribution
- Creates proper cubic regions in physical space
- Requires resampling mask back to original resolution

---

## 3. Checkpoint and Code Repository Mismatch

### The Discovery
FastSAM3D checkpoint from HuggingFace did not match the FastSAM3D repository code:

```python
# Checkpoint metadata
args.model_type = 'vit_b_ori'
args.checkpoint = './work_dir/SAM/sam_med3d_oringin.pth'  # Teacher reference!

# What this revealed:
# - Checkpoint was trained using SAM-Med3D code
# - FastSAM3D is a distilled student model
# - Checkpoint expects different architecture than FastSAM3D repo provides
```

### Technical Manifestation
```
RuntimeError: The size of tensor a (768) must match the size of tensor b (8) 
at non-singleton dimension 5
```

This occurred in the mask decoder, indicating fundamental architectural incompatibility.

### Root Cause Analysis
1. **FastSAM3D** (student): 6-layer encoder, distilled model
2. **SAM-Med3D** (teacher): 12-layer encoder, full model
3. **Checkpoint**: Trained with teacher architecture
4. **Repository**: Provided student architecture code

The checkpoint and code came from different model versions in the knowledge distillation pipeline.

### Resolution
Switch to SAM-Med3D repository and checkpoint:
- Clone `uni-medical/SAM-Med3D` instead of `arcadelab/FastSAM3D`
- Use `sam_med3d_turbo.pth` checkpoint
- Match model architecture to checkpoint training configuration

---

## 4. Interpolation Mode for 3D Data

### The Bug
SAM-Med3D repository code contained:

```python
# segment_anything/modeling/sam3D.py, line 152
masks = F.interpolate(
    masks,
    (self.image_encoder.img_size, ...),
    mode="bilinear",  # WRONG for 3D!
    align_corners=False,
)
```

### Technical Issue
- **bilinear**: 2D interpolation, expects 4D tensors [B, C, H, W]
- **Model output**: 5D tensors [B, C, D, H, W] for 3D volumes
- **PyTorch 2.6+**: Strict dimension checking, raises error

```
NotImplementedError: Got 5D input, but bilinear mode needs 4D input
```

### Why This Matters
Interpolation is critical for:
1. Resizing model output (128³) to match input size
2. Resampling back to original volume dimensions
3. Multi-scale feature processing

Using wrong mode causes:
- Runtime errors in newer PyTorch versions
- Potential incorrect segmentation in older versions that allowed it

### Fix
```python
mode="trilinear"  # Correct for 3D volumes
```

This is a fundamental requirement for 3D medical imaging models.

---

## 5. Grayscale vs RGB Normalization

### Standard Computer Vision
SAM and similar models typically expect RGB images:
```python
pixel_mean = [123.675, 116.28, 103.53]  # ImageNet RGB
pixel_std = [58.395, 57.12, 57.375]     # ImageNet RGB
```

### Medical Imaging Reality
CT scans are grayscale (single channel):
- Hounsfield Units (HU) range: -1000 to +3000
- Actual tissue range: -150 to +250 HU
- Single intensity value per voxel

### The Conflict
```python
# SAM-Med3D's default (from natural images)
self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]))
self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]))

# Shape: [3] for RGB
# Model expects: [B, 3, D, H, W]
# Medical image: [B, 1, D, H, W]
```

### Solution
Override with grayscale normalization:
```python
self.pixel_mean = torch.tensor([0.0], device=device).view(1, 1, 1, 1)
self.pixel_std = torch.tensor([1.0], device=device).view(1, 1, 1, 1)
```

Adjust preprocessing to match:
- Remove RGB channel replication
- Keep single-channel format
- Normalize based on medical image statistics (e.g., 0-1 range or HU ranges)

---

## 6. Repository Installation Without setup.py

### Challenge
Neither FastSAM3D nor SAM-Med3D provided standard Python packages:
- No `setup.py` or `pyproject.toml`
- Cannot use `pip install -e .`
- Import errors when trying to use the code

### Solution Approach
Use Python path manipulation:

```bash
# Create .pth file in virtual environment
echo "/path/to/SAM-Med3D" > venv/lib/python3.11/site-packages/sammed3d.pth

# Verify
python -c "from segment_anything.build_sam3D import sam_model_registry3D"
```

### Why This Matters
- Development workflow: Multiple environments need access
- Deployment: Production servers need reliable imports
- Collaboration: Teammates must set up same environment

Modern approach would be to fork and add proper packaging:
```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]

[project]
name = "sam-med3d"
dependencies = ["torch>=2.0", "numpy", "scipy"]
```

---

## 7. Cylinder vs Cube Segmentation Shapes

### Observed Symptom
Segmentations appeared as elongated cylinders in 3D view rather than compact blobs.

### Root Cause
Bounding box calculation was taking full depth dimension:

```python
# Wrong behavior
Crop box: D[0:128], H[132:260], W[117:245]
#         ^^^^^^^^  Full depth! Creates cylinder
```

This occurred because:
1. Point was near edge in depth dimension (e.g., z=48 in 128-sized volume)
2. Crop logic expanded to fill dimension when point was near boundary
3. Height and width properly cropped, but depth took entire volume

### Impact on Model
SAM-Med3D trained on cubic crops:
- Expects ~128³ cubic regions
- 3D convolutions assume similar extent in all dimensions
- Elongated cylinders violate training distribution

### Solution
Enforce cubic crops centered on clicks:

```python
def find_crop_box(points, padded_shape, target_size=128):
    for dim in range(3):
        center = int(np.mean([p[dim] for p in points]))
        half_size = target_size // 2
        
        min_coord = center - half_size
        max_coord = center + half_size
        
        # Shift if out of bounds, maintaining size
        if min_coord < 0:
            min_coord = 0
            max_coord = target_size
        elif max_coord > padded_shape[dim]:
            max_coord = padded_shape[dim]
            min_coord = max_coord - target_size
```

This ensures proper cubic regions even near volume boundaries.

---

## 8. Multi-Step API Workflow Design

### Architecture Decision
Rather than single-endpoint monolithic design, implemented two-step workflow:

```python
# Step 1: Upload image once
POST /upload_image
→ Server caches preprocessed volume
→ Returns session ready

# Step 2: Multiple interactions
POST /add_fastsam3d_interaction
→ Uses cached volume
→ Fast response time
```

### Benefits
1. **Efficiency**: Avoid re-uploading large volumes (75×512×512)
2. **Consistency**: All interactions work on same preprocessed volume
3. **Scalability**: Server can handle multiple interaction requests quickly

### Implementation Consideration
Cache management:
- Global state: `FASTSAM3D_PREDICTOR.set_image(volume)`
- Thread safety: Consider if multiple users access simultaneously
- Memory: Store preprocessed volumes in GPU memory

This pattern matches nnInteractive's existing design, enabling smooth integration.

---

## 9. Debugging Strategy: Systematic vs Iterative

### Initial Approach (Ineffective)
Trial-and-error parameter adjustments:
- Try coordinate swap [x,y,z] → [z,y,x]
- Try again with different order
- Add random transpose operations
- Test, observe failure, try something else

**Problem**: Treating symptoms rather than understanding root cause.

### Effective Approach
Comprehensive diagnostic logging at every transformation:

```python
print(f"1. Received coords: {coords}")
print(f"2. Image shape: {image.shape}")
print(f"3. After scaling: {scaled_coords}")
print(f"4. Tensor input: {tensor.shape}")
print(f"5. Model output: {output.shape}")
print(f"6. After postprocess: {result.shape}")
print(f"7. Mask center: {center_of_mass(result)}")
print(f"8. Distance from click: {distance}")
```

**Key Insight**: Can't fix what you can't observe. Complete visibility into data flow reveals where transformations break down.

---

## 10. Development Environment Complexity

### Multi-Location Challenge
Work distributed across:
- Local Mac: Development and Slicer GUI
- UPPMAX Server: Model inference (GPU access)
- Multiple paths: `/mnt/scratch` vs `/domus/h1/junming/private`

### SSH Tunneling
Required for local Slicer to communicate with remote server:

```bash
# Local machine
ssh -L 1527:localhost:1527 junming@p203.uppmax.uu.se

# Now Slicer can connect to localhost:1527
# which tunnels to p203's port 1527
```

### Virtual Environment Confusion
Multiple venvs created confusion:
- Scratch space venv pointing to wrong libraries
- Home directory venv with correct setup
- Path issues when imports failed

**Lesson**: Establish single source of truth for environment early in project.

---

## Summary of Key Learnings

1. **Coordinate Systems**: Require complete end-to-end mapping documentation
2. **Architecture Matters**: 2D slice-based vs 3D volumetric models have fundamentally different requirements
3. **Checkpoint Compatibility**: Model code and weights must match architecture versions
4. **Interpolation**: Use correct mode (trilinear for 3D, bilinear for 2D)
5. **Medical Imaging**: Grayscale, anisotropic spacing, requires domain-specific preprocessing
6. **Repository Quality**: Not all research code has production-quality packaging
7. **Shape Invariants**: Enforce geometric constraints (cubes not cylinders)
8. **API Design**: Multi-step workflows improve efficiency for interactive applications
9. **Debugging**: Systematic instrumentation beats trial-and-error
10. **Infrastructure**: Development environment setup is foundational to success

---

## Technical Recommendations

### For Future Integration Work

1. **Start with coordinate mapping**
   - Document all coordinate systems involved
   - Create explicit conversion functions
   - Test with known points before full integration

2. **Verify checkpoint-code compatibility**
   - Check model architecture matches checkpoint
   - Review training configuration
   - Test with minimal example before full system

3. **Understand domain requirements**
   - Medical imaging has specific preprocessing needs
   - Don't assume computer vision defaults apply
   - Consult papers for proper data handling

4. **Comprehensive logging**
   - Log shapes, types, and sample values at every step
   - Include coordinate transformations
   - Make debugging data-driven

5. **Environment discipline**
   - Single virtual environment
   - Clear dependency management  
   - Document installation steps

---

*This document captures the substantive technical challenges beyond simple coding errors, focusing on architectural mismatches, domain-specific requirements, and systematic debugging approaches.*
