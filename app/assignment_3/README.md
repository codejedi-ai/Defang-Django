# Stereo Vision Implementation - Assignment 3

This repository contains a comprehensive implementation of stereo vision algorithms, including window-based stereo matching and scan-line stereo using the Viterbi algorithm.

## Overview

The implementation consists of two main components:
1. **Window-based Stereo Matching** - Traditional stereo matching using window aggregation
2. **Scan-line Stereo with Viterbi Algorithm** - Advanced stereo matching with regularization

## Files Structure

```
assignment_3/
├── MyStereo - windows+scanlines-Copy1.ipynb  # Main notebook with stereo algorithms
├── viterbi_7b.py                             # Viterbi algorithm module
├── requirements.txt                           # Python dependencies
├── README.md                                 # This file
└── images/                                   # Test images directory
    └── stereo_pairs/
        ├── rds_left.gif                      # Random dot stereo pair (left)
        ├── rds_right.gif                     # Random dot stereo pair (right)
        └── tsukuba/                          # Tsukuba dataset
            ├── scene1.row3.col3.ppm          # Left image
            ├── scene1.row3.col4.ppm          # Right image (small baseline)
            ├── scene1.row3.col5.ppm          # Right image (large baseline)
            └── truedisp.row3.col3.pgm         # Ground truth disparity
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Or using conda:
```bash
conda install -c conda-forge numpy matplotlib scikit-image jupyter notebook -y
```

## Module Documentation: viterbi_7b.py

### Core Functions

#### `integral_image(img)`
**Purpose**: Computes the integral image (summed area table) for efficient window operations.

**Parameters**:
- `img` (numpy.ndarray): Input 2D image array

**Returns**:
- `numpy.ndarray`: Integral image of the same size as input

**Algorithm**: Uses cumulative sum along both axes to create a summed area table, enabling O(1) window sum queries.

#### `windSum(img, window_width)`
**Purpose**: Computes window sum using integral image for efficient computation.

**Parameters**:
- `img` (numpy.ndarray): Input 2D image array
- `window_width` (int): Size of the window for summation

**Returns**:
- `numpy.ndarray`: Image where each pixel contains the sum of values in a window around it

**Algorithm**: 
1. Computes integral image
2. For each pixel, calculates window sum using the formula: `sum = I[r2,c2] - I[r1-1,c2] - I[r2,c1-1] + I[r1-1,c1-1]`
3. Sets margin pixels to infinity (INFTY)

#### `SD_array(imageL, imageR, d_minimum, d_maximum)`
**Purpose**: Computes squared differences between left and right images for all disparity values.

**Parameters**:
- `imageL` (numpy.ndarray): Left image (reference)
- `imageR` (numpy.ndarray): Right image
- `d_minimum` (int): Minimum disparity value
- `d_maximum` (int): Maximum disparity value

**Returns**:
- `numpy.ndarray`: 3D array where `SD[i,r,c]` contains squared difference for disparity `i` at pixel `(r,c)`

**Algorithm**:
1. Converts images to float64 to prevent overflow
2. For each disparity value `d`:
   - Shifts right image by `d` pixels horizontally
   - Computes L2 norm squared difference between left and shifted right image
   - Stores result in `SD[d-minimum, :, :]`

#### `viterbi_7b(im_left, im_right, d_min, d_max, w, window_size)`
**Purpose**: Main Viterbi algorithm with window smoothing for stereo vision.

**Parameters**:
- `im_left` (numpy.ndarray): Left image (reference)
- `im_right` (numpy.ndarray): Right image
- `d_min` (int): Minimum disparity value
- `d_max` (int): Maximum disparity value
- `w` (float): Regularization parameter (penalty for disparity changes)
- `window_size` (int): Window size for smoothing photo-consistency term

**Returns**:
- `numpy.ndarray`: 2D disparity map where each pixel contains the optimal disparity value

**Algorithm**:
1. **Data Cost Computation**: 
   - Computes squared differences for all disparities using `SD_array()`
   - Applies window smoothing to each disparity level using `windSum()`
   - Handles infinity values by replacing with large finite penalties

2. **Forward Pass**:
   - Initializes cost array `E_bar[d, row, col]` with infinity
   - For each pixel `(row, col)` and disparity `d_curr`:
     - Computes data cost: `D_p(d_curr) = smoothed_SD[d_curr, row, col]`
     - Computes transition cost: `V(d_prev, d_curr) = w * |d_curr - d_prev|`
     - Updates: `E_bar[d_curr, row, col] = D_p(d_curr) + min(E_bar[d_prev, row, col-1] + V(d_prev, d_curr))`

3. **Backward Pass**:
   - Finds optimal disparity for last column: `argmin(E_bar[:, row, -1])`
   - Backtracks through columns to find optimal path
   - Returns complete disparity map

#### `test_viterbi_7b(im_left, im_right, d_min, d_max, window_sizes, regularization_params)`
**Purpose**: Test function for viterbi_7b with different parameters.

**Parameters**:
- `im_left` (numpy.ndarray): Left image
- `im_right` (numpy.ndarray): Right image
- `d_min` (int): Minimum disparity
- `d_max` (int): Maximum disparity
- `window_sizes` (list): List of window sizes to test (default: [1, 3, 5])
- `regularization_params` (list): List of regularization parameters to test (default: [0.0, 0.1, 1.0, 5.0])

**Returns**:
- `dict`: Dictionary containing results for each parameter combination

## Key Variables and Concepts

### Disparity
- **Definition**: Horizontal displacement between corresponding pixels in left and right images
- **Range**: `[d_min, d_max]` - typically 0 to 15 for Tsukuba dataset
- **Units**: Pixels

### Photo-consistency Term
- **Formula**: `D_p(d) = ||I_left(p) - I_right(p+d)||²`
- **Purpose**: Measures how well pixels match at disparity `d`
- **Window Smoothing**: `D_p(d) = windSum(SD_array[d], h)` where `h` is window size

### Regularization Term
- **Formula**: `V(d_p, d_q) = w * |d_p - d_q|`
- **Purpose**: Penalizes large disparity changes between neighboring pixels
- **Parameter `w`**: Controls smoothness vs. detail preservation trade-off

### Cost Function
- **Total Cost**: `E = Σ D_p(d_p) + Σ V(d_p, d_q)`
- **Optimization**: Minimize total cost using Viterbi algorithm
- **Scan-line**: Optimize along horizontal scan lines independently

## Algorithm Properties

### Complexity
- **Time**: O(W × H × D²) where W=width, H=height, D=disparity range
- **Space**: O(W × H × D) for cost array storage

### Parameters
- **Window Size (`h`)**: 
  - `h=1`: No smoothing (equivalent to basic Viterbi)
  - `h=3,5`: Moderate smoothing, reduces noise
  - Larger `h`: More smoothing, less detail
- **Regularization (`w`)**:
  - `w=0`: No regularization (equivalent to window-based stereo)
  - `w>0`: Smooth disparity maps, penalizes jumps
  - Larger `w`: Smoother results, less detail preservation

## Usage Examples

### Basic Usage
```python
from viterbi_7b import viterbi_7b
import matplotlib.image as image

# Load images
im_left = image.imread("images/stereo_pairs/tsukuba/scene1.row3.col3.ppm")
im_right = image.imread("images/stereo_pairs/tsukuba/scene1.row3.col4.ppm")

# Compute disparity map
disparity_map = viterbi_7b(im_left, im_right, d_min=0, d_max=15, w=1.0, window_size=3)

# Display result
import matplotlib.pyplot as plt
plt.imshow(disparity_map, cmap='gray')
plt.colorbar()
plt.show()
```

### Parameter Testing
```python
from viterbi_7b import test_viterbi_7b

# Test different parameters
results = test_viterbi_7b(im_left, im_right, d_min=0, d_max=15, 
                         window_sizes=[1, 3, 5], 
                         regularization_params=[0.0, 0.1, 1.0])

# Access specific result
result_h3_w1 = results["h=3_w=1.0"]
```

## Notebook Structure

The main notebook (`MyStereo - windows+scanlines-Copy1.ipynb`) is organized into functional components:

1. **Problem 1-2**: Basic squared difference computation
2. **Problem 3**: Integral image implementation
3. **Problem 4**: Window sum computation
4. **Problem 5**: Disparity map from SSD arrays
5. **Problem 6**: Window-based stereo on Tsukuba dataset
6. **Problem 7**: Scan-line stereo with Viterbi algorithm
7. **Problem 8**: Adaptive regularization weights
8. **Problem 9**: Alternative regularization terms

Each problem builds upon the previous ones, creating a comprehensive stereo vision pipeline.

## Dependencies

- **numpy**: Numerical computations
- **matplotlib**: Visualization and image I/O
- **scikit-image**: Image processing utilities
- **jupyter**: Notebook environment

## References

- Middlebury Stereo Dataset: http://vision.middlebury.edu/stereo/
- Tsukuba Dataset: University of Tsukuba (2001)
- Viterbi Algorithm: Dynamic programming for optimal path finding
- Stereo Vision: Computer vision technique for depth estimation
