# Screw it right

This is the repository for the "Screw It Right" project from HackaTUM 2022

---

## Table of Contents

1. [Project structure](#project-structure)
1. [Usage](#usage)
1. [Requirements](#requirements)

---

## Project structure

The project is rather simple and can be summarized as follows:

- `main.py` - contains the main logic for the extraction of useful features regarding the screws' angles from the provided image data
- `force_calculation.py` - represents a specifically constructed model of a mechanical system which aims to calculate the maximum allowed force that can be applied to the screw before any damage has been done to the wall and/or screw
- **Note:** The mechanical system is idealized and currently tailored to wooden walls with specific qualitative coefficients assumed. The coefficients were taken from the German handbook for civil engineers

---

## Usage

To use the program, simply pass it the folder with the image data of interest, like so:

```
$ python3 main.py --input <path-to-image-data-folder>
```

---

## Requirements

Python 3.9+ is recommended as a requirement
Further packages that might require additional installation are:

- `cv2`
- `numpy`
- `matplotlib`
- `pyransac3d`
