"""
fs_travel.py

Utility functions for recursively traversing directories and processing files.
"""

import os
from natsort import natsorted
from tqdm import tqdm

from .print_utils import print_info, print_warning, print_error

def resolve_name_collision(dest_path, dest_path_set, max_count=100):
    """
    Resolve name collisions for `dest_path` by appending numeric suffixes if the path already exists in `dest_path_set`.

    Example:
    dest_path = "/path/to/file.txt"
    dest_path_set = {"/path/to/file.txt", "/path/to/file_1.txt", "/path/to/file_2.txt"}
    resolved_path = resolve_name_collision(dest_path, dest_path_set)
    print(resolved_path)  # Output: "/path/to/file_3.txt"
    """
    count = 0
    if dest_path in dest_path_set:
        print_warning(f"dest_path {dest_path} already exists. Resolving name collision.")
    
        orig_dest_path = dest_path
        while dest_path in dest_path_set:
            if count >= max_count:
                raise ValueError(f"Cannot resolve name collision for {dest_path}")
            count += 1

            base, ext = os.path.splitext(dest_path)
            last_digits = base.split('_')[-1]
            if last_digits.isdigit():
                base = base[:-len(last_digits)] + str(int(last_digits) + 1)
            else:
                base += "_1"
            dest_path = base + ext

        print_info(f"Resolved name collision: {orig_dest_path} -> {dest_path}")

    return dest_path

def fs_travel(input_dir, output_dir, process_func, filter_func=None, rename_func=None, force=False):
    """
    Recursively traverse `input_dir`, process files filtered by `filter_func` using `process_func`, and write them to `output_dir`.
    
    Input:
    - `input_dir`: The source directory to traverse recursively.
    - `output_dir`: The destination directory where processed files will be saved.
    - `process_func`: A function that defines how each file will be processed and copied.
    - `filter_func` (optional): A function to filter files before processing.
    - `rename_func` (optional): A function to rename files before writing to the output directory.
    - `force` (optional): If `True`, overwrite existing files in the output directory.

    Output:
    - Files processed according to the provided `process_func` will be saved in the `output_dir`.
    - No explicit return value from `fs_travel`, as it performs side effects (file processing and copying).
    """
    filter_func = filter_func or (lambda x: True)
    rename_func = rename_func or (lambda x: x)
    dest_path_set = set()  # for name collision detection

    # First, put every files in the output_dir to dest_path_set
    if output_dir is not None:
        for dest_dir, dirs, files in os.walk(output_dir):
            elems = dirs + files
            elems = [os.path.join(dest_dir, elem) for elem in elems]
            dest_path_set.update(elems)
    else:
        print_warning("output_dir is None. Processing files without writing to the output directory.")

    # Process files recursively from input_dir to output_dir
    for src_dir, dirs, files in tqdm(os.walk(input_dir)):
        elems = dirs + files
        elems = [os.path.join(src_dir, elem) for elem in elems]
        elems = [elem for elem in elems if filter_func(elem)]
        if not elems:
            continue
        elems = natsorted(elems)
        for src_path in elems:
            src_dir, src_fname = os.path.split(src_path)
            if output_dir is not None:
                dest_dir = os.path.join(output_dir, os.path.relpath(src_dir, input_dir))
                dest_fname = rename_func(src_fname)
                dest_path = os.path.normpath(os.path.join(dest_dir, dest_fname))
                dest_path = resolve_name_collision(dest_path, dest_path_set)
                dest_path_set.add(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                try:
                    process_func(src_path, dest_path)
                except Exception as e:
                    print(f"Error: {e} while processing {src_path}: Ignoring this file")
            else:
                try:
                    process_func(src_path)
                except Exception as e:
                    print(f"Error: {e} while processing {src_path}: Ignoring this file")
    
    # Process the root, which is not included in os.walk
    if filter_func(input_dir):
        src_path = input_dir
        src_fname = os.path.basename(os.path.normpath(input_dir))  # Exceptional src_fname only for root to avoid going out of output_dir
        if output_dir is not None:
            dest_dir = output_dir
            dest_fname = rename_func(src_fname)
            dest_path = os.path.normpath(os.path.join(dest_dir, dest_fname))
            dest_path = resolve_name_collision(dest_path, dest_path_set)
            try:
                process_func(src_path, dest_path)
            except Exception as e:
                print(f"Error: {e} while processing {src_path}: Ignoring this file")
        else:
            try:
                process_func(src_path)
            except Exception as e:
                print(f"Error: {e} while processing {src_path}: Ignoring this file")