# Image Processing Coursework

This repository contains four assignments completed during Fall 2023 as part of the *Introduction to Computer Vision (Digital Image Processing)* course at Bu-Ali Sina University.

The assignments cover fundamental image processing and computer vision concepts, implemented using Python and common image processing libraries. Each assignment includes practical implementations, experiments with different parameters and methods, and visual comparisons of the results.

## Assignments

### Assignment 1 – Fundamentals of Image Processing

Exploration of fundamental image processing techniques, focusing on image transformations, contrast enhancement, histogram analysis, and intensity manipulation.

- **Image Interpolation:** Implemented nearest-neighbor, bilinear, and bicubic interpolation for image zooming.
- **Contrast Enhancement:** Applied logarithmic and gamma correction techniques to improve image contrast.
- **Histogram Analysis:** Examined image histograms and applied histogram equalization.
- **Geometric Transformations:** Implemented image translation and rotation.
- **Intensity Transformations:** Applied exponential, logarithmic, and piecewise intensity transformations.

**Notebook:** `Assignment1.ipynb`

### Assignment 2 – Image Enhancement and Edge Detection

Exploration of spatial image processing techniques, focusing on noise reduction, sharpening, edge detection, and image enhancement.

- **Noise Reduction:** Applied mean filtering with different kernel sizes and investigated the effects of smoothing and repeated sharpening.
- **Median and Averaging Filtering:** Compared two noise reduction techniques using different kernel sizes, examining their effectiveness and their impact on image details.
- **Edge Detection:** Implemented Sobel and Laplacian operators to detect image edges and compared their results.
- **Roberts Cross and Prewitt Operators:** Applied two gradient-based edge detection methods and explored high-boost filtering with different parameter values.
- **Image Enhancement:** Combined sharpening and logarithmic transformation techniques to improve the visibility of structures in a grayscale image.

**Notebook:** `Assignment2.ipynb`

### Assignment 3 – Frequency Domain Image Processing

Exploration of frequency-domain image processing techniques, focusing on Fourier transforms, frequency filtering, ringing effects, and periodic noise removal.

- **Fourier Transform:** Computed 2-D FFTs, displayed magnitude spectra, and reconstructed images using only the phase component.
- **Butterworth Low-Pass Filtering:** Designed Butterworth filters with different cutoff frequencies and orders to investigate their effects on image smoothing and ringing.
- **Spatial-Domain Filter Analysis:** Visualized the spatial-domain representations of the filters and examined their cross-sectional views.
- **Periodic Noise Removal:** Identified periodic noise in the frequency domain and applied frequency-domain filtering to restore images.
- **Image Restoration Evaluation:** Extracted noise components and evaluated restoration quality using Peak Signal-to-Noise Ratio (PSNR).

**Notebook:** `Assignment3.ipynb`

### Assignment 4 – Image Restoration and Morphological Processing

Exploration of image restoration, histogram equalization, morphological operations, and object detection techniques.

- **Image Restoration:** Repaired a corrupted RGB image by removing periodic noise from individual color channels using frequency-domain filtering and median filtering.
- **Histogram Equalization:** Enhanced a low-exposure image by applying histogram equalization to its Value channel and comparing the intensity histograms.
- **Morphological Noise Removal:** Applied morphological opening and closing operations to remove noise from an image.
- **Blob Separation:** Used thresholding and morphological operations to separate larger and smaller blobs.
- **Object Detection and Measurement:** Extracted rice grains using thresholding and morphological opening, then calculated their number and average area using contour analysis.

**Notebook:** `Assignment4.ipynb`

## Technologies Used

- **Python**
- **OpenCV** – Image processing and geometric transformations.
- **NumPy** – Numerical operations and image manipulation.
- **Matplotlib** – Image visualization, plotting, and histogram analysis.
- **Pillow (PIL)** – Image loading and manipulation.

## Repository Structure

```text
Image-Processing-Coursework/
│
├── Assignment1/
│   ├── Assignment1.ipynb
│   ├── Assignment1.pdf
│   └── imgs/
│
├── Assignment2/
│   ├── Assignment2.ipynb
│   ├── Assignment2.pdf
│   └── imgs/
│
├── Assignment3/
│   ├── Assignment3.ipynb
│   ├── Assignment3.pdf
│   └── imgs/
│
├── Assignment4/
│   ├── Assignment4.ipynb
│   ├── Assignment4.pdf
│   └── imgs/
│
└── README.md
```

## Purpose

The purpose of this coursework is to develop practical experience with image processing and computer vision techniques, while understanding the principles behind different algorithms and their effects on image quality and representation.

Through the four assignments, the coursework explores spatial- and frequency-domain processing, image enhancement and restoration, noise reduction, edge detection, morphological operations, and object detection and measurement. The practical implementations and experiments provide an opportunity to compare different methods, investigate their parameters, and evaluate their results.

All implementations, experiments, and visual comparisons are documented in Jupyter notebooks.