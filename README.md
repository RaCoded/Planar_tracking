# Planar Tracking

This project implements **real-time planar tracking** in a video using **SIFT keypoints** and **homography estimation**. It enables the projection of a virtual 3D coordinate system onto a physical planar surface, tracked throughout a video sequence.

## Features

- Automatic detection and matching of SIFT keypoints
- Robust homography estimation with RANSAC
- Real-time tracking of a planar object across frames
- 3D coordinate system projection onto the detected surface
- Output video generation with visual overlays (axes + rectangle)
- Adjustable calibration for accurate world-to-image transformation

## Folder Structure

Planar_tracking/
│
├── dossier_calibration/
│ └── camera_calibration.npz # Camera intrinsics
│
├── dossier_travail/
│ ├── Clown2.mp4 # Input video
│ ├── reference.jpg # Reference image
│ └── output_video.avi # Output with visual overlays
│
├── Traitement_basique.py # Custom basic processing module
└── main.py # Main tracking script


## How It Works

1. The user selects 4 reference points on the planar object in the first frame.
2. A homography is computed between the known 3D coordinates and the 2D image points.
3. SIFT keypoints within the selected rectangle are extracted.
4. For each new frame:
   - SIFT keypoints are matched to those from the reference frame
   - A homography is estimated and updated recursively
   - The 3D axes are reprojected using the estimated camera pose

## ⚙️ Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy




##  Run the Project
python main.py



## Notes
The projection matrix is refined using the camera calibration matrix.

Traitement_basique.py contains helper functions such as the point capture interface.

This method is sensitive to lighting changes, occlusions, or severe motion blur.



## 🎥 Demo

![Planar Tracking Demo](output.gif)