# MorphPy: 3D Morphological Segmentation for Binary Voxel Data
**Current Release:** v1.0.0

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/Status-Stable-success)
![Research](https://img.shields.io/badge/Research-York%20University-darkgreen)

MorphPy is an open research software package for three-dimensional morphological segmentation of binary voxel data. Using graph-based connectivity analysis, it classifies connected voxel structures into interpretable morphological classes, enabling quantitative analysis of structural connectivity in ecological, geospatial, and spatiotemporal datasets.
---

## Overview

Morphological Spatial Pattern Analysis (MSPA) is widely used in two dimensions to classify landscape structure. MorphPy extends this concept into a three-dimensional voxel framework for connected pattern analysis.

The third dimension can represent:

* **Time** `(x, y, t)`
* **Height** `(x, y, z)`
* Any ordered third dimension in binary cube data

One application of MorphPy is the analysis of annual disturbance layers stacked through time to evaluate post-disturbance boreal forest regeneration as a connected structural process.

---

## Morphological Classes

| Code | Class       | Description                                 |
| ---- | ----------- | ------------------------------------------- |
| 1    | OUTSIDE     | Background voxels coded 0 / non-feature voxels|
| 2    | MASS        | Interior voxels completely surrounded on all six sides by neighbouring feature voxels, representing the core of 3D object|
| 3    | SKIN        | Boundary voxels adjacent to MASS            |
| 4    | CRUMB       | Small isolated objects without MASS or connected to other feature voxels|
| 5    | CIRCUIT     | Connector voxels forming a closed loop by linking a MASS back to itself|
| 6    | ANTENNA     | Connector voxels attached to a MASS at a single point, forming protrusions|
| 7    | BOND        | Connector voxels linking two or more separate MASS clusters|
| 8    | VOID-VOLUME | Fully enclosed empty voxels within the feature voxels |
| 9    | VOID        | Voxels representing the internal boundary of an empty space (VOID - VOLUME) located inside the object|

---

## Features

* 3D morphological segmentation
* Graph-based topology analysis
* 6-neighbour connectivity
* Internal cavity detection
* 3D visualization
* Morphological summary statistics
* CSV export of morphology results
* Timestamped autosave folders

---
## Applications

MorphPy can be applied to binary voxel datasets in a variety of domains, including:

- Forest ecology
- Landscape ecology
- Remote sensing
- GIS and GIScience
- Environmental monitoring
- Spatiotemporal change analysis
- Any binary three-dimensional voxel dataset

## Installation

### Clone the repository

```bash
git clone https://github.com/tejumade-ojo/morphpy.git
cd morphpy
```

### Install required packages

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from Demodata import demo_data
from morph3d import morph3d

cube = demo_data()

result = morph3d(
    DATACUBE=cube,
    VERBOSE=True,
    PLOT=True,
    FINALPLOT=True
)

print(result["Summary"])
```

---
## Research Background
MorphPy was originally developed as part of MSc research in the Department of Geography, Faculty of Environmental and Urban Change, York University.
Although developed to support research on boreal forest regeneration, MorphPy is designed as a general framework for three-dimensional morphological segmentation of binary voxel datasets. The third dimension may represent time, height, or any ordered dimension depending on the application.

### Associated Thesis

**Comparing Boreal Forest Regeneration Structures Between Harvesting and Wildfire Disturbance with 3D Morphology**

Tejumade A. Ojo  
Master of Science, York University

## Authors

### Original 3D Morphology Framework (R)

Dr. Tarmo K. Remmel  
Department of Geography  
York University

### MorphPy (Python)

Tejumade A. Ojo  
MSc Researcher  
Faculty of Environmental and Urban Change  
York University

## Citation

If you use MorphPy in academic research, publications, or derived analyses, please cite both the software and the original methodology:

Ojo, T. A. (2026).
*Comparing Boreal Forest Regeneration Structures Between Harvesting and Wildfire Disturbance with 3D Morphology.*
Master's thesis, York University.

Repository:
https://github.com/tejumade-ojo/morphpy

Remmel, T. K. (2022).
*Extending Morphological Pattern Segmentation to 3D Voxels.*
Landscape Ecology.
https://doi.org/10.1007/s10980-021-01384-7

## License and Use

MorphPy is made available for academic, research, and educational use. Please refer to the `NOTICE.md` file for the terms governing use, attribution, modification, and redistribution.

## Acknowledgements

MorphPy builds upon the three-dimensional morphological framework originally developed by Dr. Tarmo K. Remmel. The Python implementation extends this framework to support reproducible scientific workflows and broader accessibility within the geospatial research community.

A machine-readable citation file (CITATION.cff) is included in this repository.