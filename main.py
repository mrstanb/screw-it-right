import os
import json
import nibabel
import numpy as np
import cv2
import pyransac3d as pyrsc
import force_calculation as fc

from matplotlib import pyplot as plt
from argparse import ArgumentParser
from pathlib import Path


def import_volume(file_path):
    """Import 3D volumetric data from file.

    Args:
        file_path (basestring): Absolute path for .img, .hdr or .json file.

    Returns:
        The volume definition given as the coordinate vectors in x, y, and z-direction.
    """

    path, filename = os.path.split(file_path)
    filename = os.path.splitext(filename)[0]

    file_path_img = os.path.join(path, f"{filename}.img")
    file_path_hdr = os.path.join(path, f"{filename}.hdr")
    file_path_json = os.path.join(path, f"{filename}.json")

    if not os.path.exists(file_path_img):
        raise Exception(f"Does not exist file: {file_path_img}")
    if not os.path.exists(file_path_hdr):
        raise Exception(f"Does not exist file: {file_path_hdr}")
    if not os.path.exists(file_path_json):
        raise Exception(f"Does not exist file: {file_path_json}")

    v_mag_phase = nibabel.load(file_path_hdr)
    _volume = v_mag_phase.dataobj[:, :, : v_mag_phase.shape[2] // 2] * np.exp(
        1j * v_mag_phase.dataobj[:, :, v_mag_phase.shape[2] // 2:]
    )
    if len(_volume.shape) > 3:
        _volume = np.squeeze(_volume)

    with open(file_path_json, "rb") as vol_definition_file:
        def_json = json.load(vol_definition_file)
        _x_vec = def_json["origin"]["x"] + np.arange(def_json["dimensions"]["x"]) * def_json["spacing"]["x"]
        _y_vec = def_json["origin"]["y"] + np.arange(def_json["dimensions"]["y"]) * def_json["spacing"]["y"]
        _z_vec = def_json["origin"]["z"] + np.arange(def_json["dimensions"]["z"]) * def_json["spacing"]["z"]

        return _volume, _x_vec, _y_vec, _z_vec


def compute_slice(_volume, _z_idx=0):
    """Compute a single slice of the given 3D volumetric data in z-direction.

    Args:
        _volume: 3D volumetric data.
        _z_idx: Index on z-direction.

    Returns:
        A single slice (img) of the given 3D volumetric data in z-direction.
    """

    img = _volume[:, :, _z_idx]
    return img


def display(
        img,
        color_map=plt.get_cmap("magma"),
        img_title=None,
        cmap_label=None,
        alphadata=None,
        xvec=None,
        yvec=None,
        dynamic_range=None,
        clim=None,
        xlabel=None,
        ylabel=None,
):
    """Helper function to display data as an image, i.e., on a 2D regular raster.

    Args:
        img: 2D array.
        color_map: The Colormap instance or registered colormap name to map scalar data to colors.
        img_title: Text to use for the title.
        cmap_label: Set the label for the x-axis.
        alphadata: The alpha blending value, between 0 (transparent) and 1 (opaque).
        xvec: coordinate vectors in x.
        yvec: coordinate vectors in y.
        dynamic_range: The dynamic range that the colormap will cover.
        clim: Set the color limits of the current image.
    """

    max_image = np.max(img)
    if dynamic_range is None:
        imshow_args = {}
    else:
        imshow_args = {"vmin": max_image - dynamic_range, "vmax": max_image}

    if xvec is None or yvec is None:
        plt.imshow(img, cmap=color_map, alpha=alphadata, origin="lower", **imshow_args)
    else:
        plt.imshow(
            img,
            cmap=color_map,
            alpha=alphadata,
            extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
            origin="lower",
            **imshow_args,
        )

    if clim is not None:
        plt.clim(clim)

    if img_title is not None:
        plt.title(img_title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel(cmap_label)
    plt.show()


def transform(img):
    lower_bound = -200
    # Crop image (expect that screw is in the center)
    img = img[75:175, 75:175]
    orig = img.copy()

    # Replace noise by lower bound value
    img[img == float("-inf")] = lower_bound

    # Normalize for OpenCV
    img_normalized = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    # Apply Gaussian blur to enhance bright spots
    img_gauss = cv2.GaussianBlur(img_normalized, (7, 7), 0)

    # Apply MinMaxLoc to get the brightest spot in the image (where we expect to have a screw)
    _, _, _, max_loc = cv2.minMaxLoc(img_gauss)
    # Draw circle around the point for visualization purposes
    cv2.circle(orig, max_loc, 5, (255, 0, 0), 1)
    cv2.circle(img_gauss, max_loc, 5, (255, 0, 0), 1)

    return orig, img_gauss, max_loc


def main():
    usage = "Detect load that a screw can withstand."
    parser = ArgumentParser(usage=usage)
    parser.add_argument("--input", type=str, help="path to folder with input microwave data", required=True)
    np.seterr(divide='ignore')

    args = parser.parse_args()
    input_file_path = args.input

    data_sample_path = input_file_path + "/" + os.path.basename(os.path.normpath(input_file_path)) + "_reco.img"
    if not Path(data_sample_path).is_file():
        raise ValueError(f"Path '{data_sample_path}' is invalid!")

    volume, x_vec, y_vec, z_vec = import_volume(data_sample_path)
    # We empirically chose slices where we can see the screw
    chosen_slices = [i for i in range(5, 20)]

    # From each slice, collect the central point of the screw
    point_cloud = []

    for i in chosen_slices:
        s = compute_slice(np.abs(volume), i)
        if np.max(s) == 0:
            continue

        image_raw = 30 * np.log10(s / np.max(s))
        orig, img_gauss, max_loc = transform(image_raw)

        # display(
        #     img_gauss,
        #     img_title=f"Slice {i} with Gaussian Blur"
        # )

        point_cloud.append([max_loc[0], max_loc[1], i])

    pc_array = np.array(point_cloud)

    rsc_line = pyrsc.Line()
    a, b, _ = rsc_line.fit(pc_array, 3)

    # Compute x and y for intersection with plane at z = 5
    x_5 = a[0] * ((5 - b[2]) / a[2]) + b[0]
    y_5 = a[1] * ((5 - b[2]) / a[2]) + b[1]

    # Compute x and y for intersection with plane at z = 20
    x_20 = a[0] * ((20 - b[2]) / a[2]) + b[0]
    y_20 = a[1] * ((20 - b[2]) / a[2]) + b[1]

    # print(f"Values for x_5 and y_5: {x_5} and {y_5}")
    # print(f"Values for x_20 and y_20: {x_20} and {y_20}")

    p_5 = np.array((x_5, y_5))
    p_20 = np.array((x_20, y_20))
    cat = np.linalg.norm(p_5 - p_20)
    angle = np.degrees(np.arctan(cat / 30))     # 30 mm are between slices 5 and 20

    theta = np.radians(45)
    # Rotate because our piece of wood is in a 45 deg angle in the image
    # This eventually serves the purpose of determining if the screw is facing upwards or downwards
    # This will be later used for the force calculation
    rotation_matrix = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    xy_5_prime = rotation_matrix @ np.array((x_5, y_5))
    xy_20_prime = rotation_matrix @ np.array((x_20, y_20))
    if xy_5_prime[0] < xy_20_prime[0]:
        angle *= -1

    # Force computation (in kg)
    force_kg = fc.compute_max_influential_force(float(angle))
    print(f"Maximum force under angle {round(angle, 2)}°: {round(force_kg, 2)} KG")


if __name__ == '__main__':
    main()
