import os
import json
import nibabel
import numpy as np

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
        color_map=plt.get_cmap("viridis"),
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
    # TODO: image pre-processing
    return img[75:175, 75:175]


def main():
    usage = "Detect load that a screw can withstand."
    parser = ArgumentParser(usage=usage)
    parser.add_argument("--input", type=str, help="path to folder with input microwave data", required=True)

    args = parser.parse_args()
    input_file_path = args.input

    data_sample_path = input_file_path + "/" + os.path.basename(os.path.normpath(input_file_path)) + "_reco.img"
    if not Path(data_sample_path).is_file():
        raise ValueError(f"Path '{data_sample_path}' is invalid!")

    volume, x_vec, y_vec, z_vec = import_volume(data_sample_path)
    chosen_slices = [10, 30]

    for i in chosen_slices:
        s = compute_slice(np.abs(volume), i)
        if np.max(s) == 0:
            continue

        image_raw = 30 * np.log10(s / np.max(s))
        display(
            transform(image_raw),
            img_title=f"Slice {i}",
            cmap_label="Normalized magnitude in dB",
            xvec=x_vec,
            yvec=y_vec,
            dynamic_range=50,
            xlabel="$x$ in m",
            ylabel="$y$ in m",
        )


if __name__ == '__main__':
    main()
