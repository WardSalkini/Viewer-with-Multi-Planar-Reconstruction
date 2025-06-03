# MPRViewer - Viewer with Multi-Planar Reconstruction
## Introduction
### MPRViewer is an interactive application designed for viewing medical imaging data in the NIfTI format. It allows users to view multi-planar reconstructions (MPR) of 3D volume data with three orthogonal slices: Axial, Coronal, and Sagittal. The application also provides volume rendering capabilities using VTK, enabling users to visualize 3D structures from the 2D slices. Additionally, users can adjust brightness, contrast, and select different colormaps to enhance the visual representation of the data.
### This tool is useful for medical imaging analysis, especially when working with volumetric data like MRI and CT scans.

###### image of program:
<div>
  <img src ="images/{A05E05A5-0736-438E-87C4-186A76D4C8E6}.png" >
</div>

## Features
- Load NIfTI file: Click on "Open NIfTI File" to browse and open a NIfTI file from your computer.

- Zoom (in-out) using the mouse.

- Scroll through slices in each view.

- Slice Navigation: Adjust the slices by dragging or scrolling.

- Adjust Brightness and Contrast: Use the sliders to fine-tune the slice appearance.

- Add Points: Enable the "Add Point" checkbox and click on the image slices to mark points, which will be reflected in all views.

- Pan and Zoom: Use the "Pan" button to drag and move the views, and the mouse scroll to zoom in/out.



## Requirements

- Python 3.8 or higher

- A CUDA-compatible GPU is recommended for faster training.

- Required libraries (listed in [requirement.txt](https://github.com/ziad0nassif/MPRViewer/blob/3eed246afeb3e14bb025fcab0a09f595ba1815a1/requirements.txt) )



## Logging
The program logs user interactions and critical steps, aiding in debugging and problem resolution. Log files are generated to provide insights into the development process.

### Feel free to fork this repository, make improvements, and submit a pull request if you have any enhancements or bug fixes.
