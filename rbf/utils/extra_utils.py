from math import log, pi
import numpy as np
from scipy.ndimage import distance_transform_edt
from dataclasses import fields
import importlib
import torch
from torch.nn.functional import interpolate
import functools
import weakref
import sys
import contextlib
from tqdm.contrib import DummyTqdmFile
import tqdm


def accumulate_tensor(tensor, index, value):
    assert (
        tensor.shape[:-1] == value.shape[:-1]
    ), "tensor and value must have the same shape, except for the last dimension"
    assert (
        index.shape == value.shape[-1:]
    ), "index must be a 1D tensor with the same length as the last dimension of value"

    accumulated_tensor = torch.zeros_like(tensor)
    counter = torch.zeros_like(tensor)

    accumulated_tensor.index_add_(-1, index, value)
    counter.index_add_(-1, index, torch.ones_like(value))

    return tensor + accumulated_tensor, counter


def attach_direction_prompt(prompt, elevs, azims):
    if type(elevs) == float:
        elevs = [elevs]
    if type(azims) == float:
        azims = [azims]

    output_prompts = []

    for elev, azim in zip(elevs, azims):
        direction_prompt = ""
        if elev > 60:
            direction_prompt = "overhead view"
        elif azim < 45 or azim > 315:
            direction_prompt = "side view"
        elif azim < 135:
            direction_prompt = "front view"
        elif azim < 225:
            direction_prompt = "side view"
        else:
            direction_prompt = "back view"
        output_prompts.append(f"{prompt}, {direction_prompt}")

    return output_prompts


def attach_detailed_direction_prompt(prompt, elevs, azims):
    if type(elevs) == float:
        elevs = [elevs]
    if type(azims) == float:
        azims = [azims]

    output_prompts = []

    for elev, azim in zip(elevs, azims):
        # make sure the values are in the range of 0~360
        azim = (azim + 360) % 360
        direction_prompt = ""
        if elev > 60:
            direction_prompt = "overhead view"
        elif azim < 22.5 or azim > 337.5:
            direction_prompt = "side view"
        elif azim < 67.5:
            direction_prompt = "front-side view"
        elif azim < 112.5:
            direction_prompt = "front view"
        elif azim < 157.5:
            direction_prompt = "front-side view"
        elif azim < 202.5:
            direction_prompt = "side view"
        elif azim < 247.5:
            direction_prompt = "back-side view"
        elif azim < 292.5:
            direction_prompt = "back view"
        else:
            direction_prompt = "back-side view"
        output_prompts.append(f"{prompt}, {direction_prompt}")

    return output_prompts


def attach_elevation_prompt(prompt, elevs):
    if type(elevs) == float:
        elevs = [elevs]

    output_prompts = []

    for elev in elevs:
        direction_prompt = ""
        # elev -90~-80: "viewed from directly above"
        # elev -80~-45: "high-angle downward view"
        # elev -45~-10: "elevated downward view"
        # elev -10~10: "eye-level view"
        # elev 10~45: "low-angle upward view"
        # elev 45~80: "steep upward view"
        # elev 80~: "viewed from directly below"
        if elev < -80:
            direction_prompt = "viewed from directly above"
        elif elev < -45:
            direction_prompt = "high-angle downward view"
        elif elev < -10:
            direction_prompt = "elevated downward view"
        elif elev < 10:
            direction_prompt = "eye-level view"
        elif elev < 45:
            direction_prompt = "low-angle upward view"
        elif elev < 80:
            direction_prompt = "steep upward view"
        else:
            direction_prompt = "viewed from directly below"
        output_prompts.append(f"{prompt}, {direction_prompt}")

    return output_prompts


def ignore_kwargs(cls):
    original_init = cls.__init__

    def init(self, *args, **kwargs):
        expected_fields = {field.name for field in fields(cls)}
        expected_kwargs = {
            key: value for key, value in kwargs.items() if key in expected_fields
        }
        original_init(self, *args, **expected_kwargs)

    cls.__init__ = init
    return cls


def get_class_filename(cls):
    """
    Get the filename of the module containing the given class.

    Args:
    cls (type): The class for which to find the filename.

    Returns:
    str: The filename of the module containing the class.
    """
    # Get the module name where the class is defined
    module_name = cls.__module__

    # Import the module dynamically
    module = importlib.import_module(module_name)

    # Retrieve the filename of the module
    filename = module.__file__

    return filename


def weak_lru(maxsize=128, typed=False):
    'LRU Cache decorator that keeps a weak reference to "self"'

    def wrapper(func):

        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper


@contextlib.contextmanager
def redirect_stdout_to_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


ORIG_STDOUT = sys.stdout
ORIG_STDERR = sys.stderr


def redirected_tqdm(*args, **kwargs):
    return tqdm.tqdm(*args, file=ORIG_STDOUT, **kwargs)


def redirected_trange(*args, **kwargs):
    return tqdm.trange(*args, file=ORIG_STDOUT, **kwargs)


if __name__ == "__main__":

    def dummy():
        print("dummy")

    def dummy_loop():
        for j in redirected_trange(3, desc="inner", leave=False, position=1):
            print(i, j)
            sleep(0.5)
            dummy()
            sleep(0.5)

    from time import sleep

    with redirect_stdout_to_tqdm():
        for i in redirected_trange(2, desc="outer", leave=False, position=0):
            dummy_loop()


def calculate_distance_to_zero_level(mask_tensor):
    mask_np = mask_tensor.cpu().numpy().astype(np.bool_)
    distance_np = np.zeros_like(mask_np, dtype=np.float32)
    for i in range(mask_np.shape[0]):
        distance_np[i] = distance_transform_edt(~mask_np[i])
    distance_tensor = torch.from_numpy(distance_np).float()

    distance_tensor = distance_tensor.to(mask_tensor.device)

    return distance_tensor
