from math import pi, tan, atan
from typing import Literal

import numpy as np
import torch


def fov_to_focal_length(fov):
    return 0.5 / tan(fov / 2)


def focal_length_to_fov(focal_length, hole_rad=0.5):
    return 2 * atan(hole_rad / focal_length)


CAM_TRANSFORMATIONS = {
    "RUF": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "Unity": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "Pytorch3D": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "LUF": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "OpenGL": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    "RUB": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    "OpenCV": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    "RDF": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
}


def convert_camera_convention(mtx, from_convention, to_convention):
    """
    Converts camera coordinate conventions for given transformation or rotation matrices.

    Parameters:
    mtx (numpy.ndarray): Transformation matrices.
                              Can be of shape (4, 4) for a single full transformation matrix,
                              (3, 3) for a single rotation matrix,
                              (B, 4, 4) for a batch of full transformation matrices, or
                              (B, 3, 3) for a batch of rotation matrices.
    from_convention (str): The source camera convention.
    to_convention (str): The target camera convention.

    Returns:
    numpy.ndarray: Transformed matrices with the target camera convention.
    """

    from_ruf = CAM_TRANSFORMATIONS[from_convention]
    ruf_to = CAM_TRANSFORMATIONS[to_convention]
    from_to = np.linalg.inv(from_ruf) @ ruf_to

    new_mtx = mtx.copy()
    if mtx.ndim == 2 and (mtx.shape == (4, 4) or mtx.shape == (3, 3)):
        new_mtx[:3, :3] = new_mtx[:3, :3] @ from_to
    elif mtx.ndim == 3 and (mtx.shape[1:] == (4, 4) or mtx.shape[1:] == (3, 3)):
        new_mtx[:, :3, :3] = new_mtx[:, :3, :3] @ from_to
    else:
        raise ValueError(
            "Invalid shape for mtx. Must be (4, 4), (3, 3), (B, 4, 4), or (B, 3, 3)."
        )
    return new_mtx


def generate_camera_params(
    dist,
    elev,
    azim,
    fov,
    height,
    width,
    up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z",
    convention: Literal[
        "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
    ] = "OpenCV",
):
    # If input is tensor, convert to float
    if isinstance(dist, torch.Tensor):
        dist = dist.item()
    if isinstance(elev, torch.Tensor):
        elev = elev.item()
    if isinstance(azim, torch.Tensor):
        azim = azim.item()

    elev = elev * np.pi / 180
    azim = azim * np.pi / 180
    fov = fov * np.pi / 180

    world_up, world_front, world_right = {
        "x": (np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0])),
        "y": (np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])),
        "z": (np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0])),
        "-x": (np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
        "-y": (np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])),
        "-z": (np.array([0, 0, -1]), np.array([1, 0, 0]), np.array([0, 1, 0])),
    }[up_vec]

    cam_pos = dist * (
        world_up * np.sin(elev)
        - world_front * np.cos(elev) * np.cos(azim)
        + world_right * np.cos(elev) * np.sin(azim)
    )

    lookat = -cam_pos / np.linalg.norm(cam_pos)
    fake_up = world_up

    right = np.cross(lookat, fake_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, lookat)

    R = np.stack([right, up, lookat], axis=1)
    R = convert_camera_convention(R, "RUF", convention)

    invert_y = not (convention == "RDF" or convention == "OpenCV")

    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_pos

    focal_length = fov_to_focal_length(fov)
    fx = fy = focal_length * height
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    if invert_y:
        K[1, 1] *= -1

    return c2w, K


def generate_camera(
    dist,
    elev,
    azim,
    fov,
    height,
    width,
    up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z",
    convention: Literal[
        "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
    ] = "RDF",
    device="cpu",
):
    c2w, K = generate_camera_params(
        dist,
        elev,
        azim,
        fov,
        height,
        width,
        up_vec=up_vec,
        convention=convention,
    )
    c2w = torch.tensor(c2w, dtype=torch.float32).to(device)
    K = torch.tensor(K, dtype=torch.float32).to(device)

    return {
        "num": 1,
        "c2w": c2w.unsqueeze(0),
        "K": K.unsqueeze(0),
        "height": height,
        "width": width,
        "fov": fov,
        "azimuth": [azim],
        "elevation": [elev],
        "dist": [dist],
    }


def merge_camera(cam_list):
    assert all(
        cam_list[0]["height"] == cam["height"] for cam in cam_list
    ), "All cameras must have the same height."
    assert all(
        cam_list[0]["width"] == cam["width"] for cam in cam_list
    ), "All cameras must have the same width."
    assert all(
        cam_list[0]["fov"] == cam["fov"] for cam in cam_list
    ), "All cameras must have the same fov."

    c2w = torch.cat([cam["c2w"] for cam in cam_list], dim=0)
    K = torch.cat([cam["K"] for cam in cam_list], dim=0)
    width = cam_list[0]["width"]
    height = cam_list[0]["height"]
    fov = cam_list[0]["fov"]
    azimuth = [azim for cam in cam_list for azim in cam["azimuth"]]
    elevation = [elev for cam in cam_list for elev in cam["elevation"]]
    dist = [d for cam in cam_list for d in cam["dist"]]

    assert len(azimuth) == len(elevation) == len(c2w) == len(K), "Invalid camera list."

    return {
        "num": len(c2w),
        "c2w": c2w,
        "K": K,
        "width": width,
        "height": height,
        "fov": fov,
        "azimuth": azimuth,
        "elevation": elevation,
        "dist": dist,
    }


def index_camera(cam_list, idx):
    assert idx < cam_list["num"], "Index out of range."

    return {
        "num": 1,
        "c2w": cam_list["c2w"][idx : idx + 1],
        "K": cam_list["K"][idx : idx + 1],
        "width": cam_list["width"],
        "height": cam_list["height"],
        "fov": cam_list["fov"],
        "azimuth": [cam_list["azimuth"][idx]],
        "elevation": [cam_list["elevation"][idx]],
        "dist": [cam_list["dist"][idx]],
    }


def camera_hash(cam):
    azimuth_int = [int(azim * 100) for azim in cam["azimuth"]]
    elevation_int = [int(elev * 100) for elev in cam["elevation"]]
    dist_int = [int(d * 100) for d in cam["dist"]]
    return hash(
        (
            cam["width"],
            cam["height"],
            cam["fov"],
            tuple(azimuth_int),
            tuple(elevation_int),
            tuple(dist_int),
        )
    )
