import os
import base64
import numpy as np
import torch
from PIL import Image
import imageio
from natsort import natsorted
from tqdm import tqdm
from .print_utils import print_info, print_warning, print_error
from .fs_travel_utils import fs_travel

### Constants ###
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
VIDEO_EXTENSIONS = [".mp4", ".gif"]

### Image-Video Processing Functions ###


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_images(input_path):
    """
    Load images from a directory, video file, or list of image paths.

    Args:
        input_path (str or list): Path to a directory, video file, or list of image paths.

    Returns:
        list: List of PIL Image objects.
    """
    assert isinstance(
        input_path, (str, list)
    ), "input_path should be a string or a list."

    if isinstance(input_path, str):
        if input_path.endswith(tuple(VIDEO_EXTENSIONS)):
            reader = imageio.get_reader(input_path)
            images = [Image.fromarray(img) for img in reader]
        elif os.path.isdir(input_path):
            all_paths = natsorted(os.listdir(input_path))
            img_paths = [
                os.path.join(input_path, img)
                for img in all_paths
                if img.lower().endswith(tuple(SUPPORTED_IMAGE_EXTENSIONS))
            ]
            images = [
                Image.fromarray(imageio.imread(img_path)) for img_path in img_paths
            ]
        else:
            raise ValueError("input_path should be a directory or a video file.")
    else:
        images = [Image.fromarray(imageio.imread(img_path)) for img_path in input_path]

    return images


def save_video(images, filename, fps=10):
    """
    Save a list of PIL images as a video file.

    Args:
        images (list): List of PIL Image objects.
        filename (str): Output video filename.
        fps (int, optional): Frames per second. Default is 10.
    """
    with imageio.get_writer(filename, fps=fps) as writer:
        for img in images:
            writer.append_data(np.array(img))


def save_gif(images, filename, fps=10):
    """
    Save a list of PIL images as a GIF file.

    Args:
        images (list): List of PIL Image objects.
        filename (str): Output GIF filename.
        fps (int, optional): Frames per second. Default is 10.
    """
    duration = int(1000 // fps)
    images[0].save(
        filename, save_all=True, append_images=images[1:], loop=0, duration=duration
    )


def concat_images(images, row_size, col_size):
    """
    Concatenate multiple images into a single image.

    Args:
        images (list): List of PIL Image objects.
        row_size (int): Number of rows.
        col_size (int): Number of columns.

    Returns:
        PIL.Image: Concatenated image.
    """
    assert (
        len(images) <= row_size * col_size
    ), f"Too many images: {len(images)} > {row_size * col_size}"
    w, h = images[0].size
    concatenated_image = Image.new("RGB", (w * col_size, h * row_size))
    for i, image in enumerate(images):
        concatenated_image.paste(image, (i % col_size * w, i // col_size * h))
    return concatenated_image


def check_existence(path, force=False):
    """
    Check if a path exists and handle based on the force flag.

    Args:
        path (str): Path to check.
        force (bool, optional): If True, overwrite existing path. Default is False.

    Returns:
        bool: True if path exists and should not be overwritten, False otherwise.
    """
    if os.path.exists(path):
        if not force:
            print_error(f"Path {path} already exists. Skipping.")
            return True
        else:
            print_warning(f"Path {path} already exists. Overwriting.")
    return False


def convert_to_video(input_path, output_path, fps=10, force=False):
    """
    Convert images or a video file to a video.

    Args:
        input_path (str or list): Input path to images or video file.
        output_path (str): Output video file path.
        fps (int, optional): Frames per second. Default is 10.
        force (bool, optional): If True, overwrite existing file. Default is False.
    """
    if check_existence(output_path, force):
        return
    images = load_images(input_path)
    save_video(images, output_path, fps=fps)


def convert_to_gif(input_path, output_path, fps=10, force=False):
    """
    Convert images or a video file to a GIF.

    Args:
        input_path (str or list): Input path to images or video file.
        output_path (str): Output GIF file path.
        fps (int, optional): Frames per second. Default is 10.
        force (bool, optional): If True, overwrite existing file. Default is False.
    """
    if check_existence(output_path, force):
        return
    images = load_images(input_path)
    save_gif(images, output_path, fps=fps)


def convert_to_video_recursive(
    input_dir, output_dir, fps=10, force=False, filter_func=None
):
    """
    Recursively convert all images and GIFs in a directory to videos.

    Args:
        input_dir (str): Input directory containing images and GIFs.
        output_dir (str): Output directory for videos.
        fps (int, optional): Frames per second. Default is 10.
        force (bool, optional): If True, overwrite existing files. Default is False.
        filter_func (callable, optional): Function to filter files. Default is None.
        rename_func (callable, optional): Function to rename output files. Default is None.
    """

    def process_func(src_path, dest_path):
        images = load_images(src_path)
        if len(images) < 2:
            return
        save_video(images, dest_path, fps=fps)

    filter_func = (filter_func) or (lambda x: True)

    filter_func1 = lambda x: os.path.isdir(x) and filter_func(x)
    rename_func1 = lambda x: x + ".mp4"
    fs_travel(
        input_dir,
        output_dir,
        process_func,
        filter_func=filter_func1,
        rename_func=rename_func1,
        force=force,
    )

    filter_func2 = lambda x: x.endswith(".gif") and filter_func(x)
    rename_func2 = lambda x: x[:-4] + ".mp4"
    fs_travel(
        input_dir,
        output_dir,
        process_func,
        filter_func=filter_func2,
        rename_func=rename_func2,
        force=force,
    )


def convert_to_gif_recursive(
    input_dir, output_dir, fps=10, force=False, filter_func=None
):
    """
    Recursively convert all images and videos in a directory to GIFs.

    Args:
        input_dir (str): Input directory containing images and videos.
        output_dir (str): Output directory for GIFs.
        fps (int, optional): Frames per second. Default is 10.
        force (bool, optional): If True, overwrite existing files. Default is False.
    """

    def process_func(src_path, dest_path):
        images = load_images(src_path)
        if len(images) < 2:
            return
        save_gif(images, dest_path, fps=fps)

    filter_func = (filter_func) or (lambda x: True)

    filter_func1 = lambda x: os.path.isdir(x) and filter_func(x)
    rename_func1 = lambda x: x + ".gif"
    fs_travel(
        input_dir,
        output_dir,
        process_func,
        filter_func=filter_func1,
        rename_func=rename_func1,
        force=force,
    )

    filter_func2 = lambda x: x.endswith(".mp4") and filter_func(x)
    rename_func2 = lambda x: os.path.splitext(x)[0] + ".gif"
    fs_travel(
        input_dir,
        output_dir,
        process_func,
        filter_func=filter_func2,
        rename_func=rename_func2,
        force=force,
    )


def convert_to_concat_video(input_paths, output_path, fps=10, margin=0, fill_color=0):
    """
    Concatenate multiple videos into a single video.

    Args:
        input_paths (list): List of input video file paths.
        output_path (str): Output video file path.
        fps (int, optional): Frames per second. Default is 10.
        margin (int, optional): Margin between videos in pixels. Default is 0.
        fill_color (int or tuple, optional): Fill color for the margin. Default is 0 (black).
    """
    readers = [imageio.get_reader(input_path) for input_path in input_paths]
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frames in zip(*readers):
            if margin > 0:
                frames = [
                    np.pad(
                        frame,
                        ((0, 0), (0, margin), (0, 0)),
                        mode="constant",
                        constant_values=fill_color,
                    )
                    for frame in frames[:-1]
                ] + [frames[-1]]
            frame = np.concatenate(frames, axis=1)
            writer.append_data(frame)


### Torch-PIL-Image&Video Conversion Functions ###


def pil_to_torch(pil_img):
    _np_img = np.array(pil_img).astype(np.float32) / 255.0
    _torch_img = torch.from_numpy(_np_img).permute(2, 0, 1).unsqueeze(0)
    return _torch_img


def torch_to_pil(tensor, is_grayscale=False, cmap=None):
    # Convert a torch image tensor to a PIL image.
    # Input: tensor (HW or 1HW or 13HW or 3HW), is_grayscale (bool), cmap (str)
    # Output: PIL image

    if is_grayscale:
        assert tensor.dim() == 2 or (
            tensor.dim() == 3 and tensor.shape[0] == 1
        ), f"Grayscale tensor should be one of HW or 1HW: got {tensor.shape}."  # HW or 1HW
        # Make them all 3D tensor
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # 1HW

        # colormap
        if cmap is not None:
            raise NotImplementedError("Not implemented yet")
        else:
            tensor = tensor.repeat(3, 1, 1)
    else:
        assert (tensor.dim() == 3 and tensor.shape[0] == 3) or (
            tensor.dim() == 4 and tensor.shape[0] == 1 and tensor.shape[1] == 3
        ), f"Color tensor should be one of 3HW or 13HW: got {tensor.shape}."  # 3HW or 13HW
        # Make them all 3D tensor
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

    tensor = tensor.clamp(0, 1)  # 3HW
    assert (
        tensor.dim() == 3 and tensor.shape[0] == 3
    ), f"Invalid tensor shape: {tensor.shape}"
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()
    tensor = (tensor * 255.0).astype(np.uint8)
    pil_tensor = Image.fromarray(tensor)
    return pil_tensor


def torch_to_pil_batch(tensor, is_grayscale=False, cmap=None):
    # Convert a batch of torch image tensor to a list of PIL images, agnostic to the number of channels and batch size.
    # Input: tensor (HW or BHW or B1HW or 3HW or B3HW), is_grayscale (bool), cmap (str)
    # Output: list of PIL images

    if is_grayscale:
        assert (
            tensor.dim() == 2
            or tensor.dim() == 3
            or (tensor.dim() == 4 and tensor.shape[1] == 1)
        ), f"Grayscale tensor should be one of HW, BHW, B1HW: got {tensor.shape}."  # HW or BHW or B1HW
        # Make them all 4D tensor
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # 11HW
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)  # B1HW

        # colormap
        if cmap is not None:
            raise NotImplementedError("Not implemented yet")
        else:
            tensor = tensor.repeat(1, 3, 1, 1)
    else:
        assert (tensor.dim() == 3 and tensor.shape[0] == 3) or (
            tensor.dim() == 4 and tensor.shape[1] == 3
        ), f"Color tensor should be one of 3HW or B3HW: got {tensor.shape}."  # 3HW or B3HW
        # Make them all 4D tensor
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # 13HW

    tensor = tensor.clamp(0, 1)  # B3HW
    assert (
        tensor.dim() == 4 and tensor.shape[1] == 3
    ), f"Invalid tensor shape: {tensor.shape}"
    tensor = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    tensor = (tensor * 255.0).astype(np.uint8)
    pil_tensors = [Image.fromarray(image) for image in tensor]
    return pil_tensors


def save_tensor(
    input_tensor,
    output_path,
    is_grayscale=False,
    save_type="images",
    fps=10,
    row_size=None,
    col_size=None,
    fns=None,
):
    def attach_ext(path, ext):
        if not path.endswith(ext):
            path += ext
        return path

    dim = input_tensor.dim()
    shape = input_tensor.shape
    type_4d = ("images", "video", "gif", "cat_image")
    type_5d = ("cat_images", "cat_video")
    if save_type in type_4d:
        if is_grayscale:
            # Normalize tensor
            input_tensor = (input_tensor - input_tensor.min()) / (
                input_tensor.max() - input_tensor.min()
            )
            images = torch_to_pil_batch(input_tensor, is_grayscale=True)
        else:
            images = torch_to_pil_batch(input_tensor, is_grayscale=False)
    elif save_type in type_5d:
        raise NotImplementedError(f"Invalid save_type: {save_type}")
    else:
        raise ValueError(f"Invalid save_type: {save_type}")

    if save_type == "images":
        if len(images) == 1:
            output_path = attach_ext(output_path, ".png")
            images[0].save(output_path)
        else:
            if fns is None:
                fns = [f"{i:03d}" for i in range(len(images))]
            
            for (fn, img) in zip(fns, images):
                if output_path.endswith(".png"):
                    output_full_path = output_path[:-4] + f"_{fn}.png"
                else:
                    assert not os.path.exists(output_path) or os.path.isdir(
                        output_path
                    ), f"output_path {output_path} should be a directory or non-existing file."
                    os.makedirs(output_path, exist_ok=True)
                    output_full_path = os.path.join(output_path, f"{fn}.png")
                img.save(output_full_path)

    elif save_type == "video":
        output_path = attach_ext(output_path, ".mp4")  # attach extension if not exists
        save_video(images, output_path, fps=fps)

    elif save_type == "gif":
        output_path = attach_ext(output_path, ".gif")  # attach extension if not exists
        save_gif(images, output_path, fps=fps)

    elif save_type == "cat_image":
        num_imgs = len(images)
        if row_size is None and col_size is None:
            row_size = int(np.sqrt(num_imgs))
            col_size = int(np.ceil(num_imgs / row_size))
        elif row_size is None:
            col_size = max(col_size, 1)
            row_size = int(np.ceil(num_imgs / col_size))
        elif col_size is None:
            row_size = max(row_size, 1)
            col_size = int(np.ceil(num_imgs / row_size))
        elif row_size * col_size < num_imgs:
            print_warning("row_size * col_size < num_imgs. Adjusting row_size.")
            col_size = max(col_size, 1)
            row_size = int(np.ceil(num_imgs / col_size))

        img = concat_images(images, row_size, col_size)
        output_path = attach_ext(output_path, ".png")  # attach extension if not exists
        img.save(output_path)

    elif save_type == "cat_images":
        raise NotImplementedError("Not implemented yet")
    elif save_type == "cat_video":
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError(f"Invalid save_type: {save_type}")


### Torch-PIL-Image&Video Conversion Functions End ###

import os
import imageio
from PIL import Image, ImageDraw, ImageFont


def convert_to_labeled_video(
    input_path, output_path, label, org=(30, 30), color=(0, 0, 0)
):
    """
    Label a video with a text.

    Args:
        input_path (str): Input video path.
        output_path (str): Output video path.
        label (str): Text to label.
        org (tuple): Origin of the text from the top-left corner.
        color (tuple): Color of the text.

    Returns:
        None
    """
    # Check if input path is valid
    if not os.path.isfile(input_path):
        raise ValueError("Input path is not a valid file.")

    # Initialize video reader and writer
    reader = imageio.get_reader(input_path)
    writer = imageio.get_writer(output_path, fps=reader.get_meta_data()["fps"])

    font_path = os.path.join(os.path.dirname(__file__), "arial.ttf")
    font = ImageFont.truetype(font_path, 50)

    for frame in reader:
        # Convert frame to PIL Image
        pil_img = Image.fromarray(frame)
        ImageDraw.Draw(pil_img).text(org, label, fill=color, font=font)
        labeled_frame = np.array(pil_img)

        # Append the labeled frame to the output video
        writer.append_data(labeled_frame)

    # Close the writer to finalize the video
    writer.close()
    reader.close()


def convert_to_labeled_grid_video(
    input_paths,
    output_path,
    labels,
    org=(30, 30),
    color=(0, 0, 0),
    text_size=32,
    row_size=None,
    col_size=None,
    fps=10,
):
    """
    Label a grid of videos with text.

    Args:
        input_paths (list): List of input video paths.
        output_path (str): Output video path.
        labels (list): List of text labels.
        org (tuple): Origin of the text from the top-left corner.
        color (tuple): Color of the text.
        row_size (int): Number of rows.
        col_size (int): Number of columns.

    Returns:
        None
    """

    def adjust_grid_size(num_videos, row_size=None, col_size=None):
        if row_size is None and col_size is None:
            row_size = int(np.sqrt(num_videos))
            col_size = int(np.ceil(num_videos / row_size))
        elif row_size is None:
            col_size = max(col_size, 1)
            row_size = int(np.ceil(num_videos / col_size))
        elif col_size is None:
            row_size = max(row_size, 1)
            col_size = int(np.ceil(num_videos / row_size))
        elif row_size * col_size < num_videos:
            print_warning("row_size * col_size < num_videos. Adjusting batch_size.")
            row_size = max(row_size, 1)
            col_size = max(col_size, 1)
        return row_size, col_size

    def dir_reader(path):
        for img_path in natsorted(os.listdir(path)):
            yield imageio.imread(os.path.join(path, img_path))

    assert all(
        [os.path.exists(path) for path in input_paths]
    ), "Input path does not exist."
    assert len(input_paths) == len(
        labels
    ), f"Number of input paths and labels should be the same, but got {len(input_paths)} and {len(labels)}."

    # Adjust grid size
    num_videos = len(input_paths)
    row_size, col_size = adjust_grid_size(num_videos, row_size, col_size)
    num_per_batch = row_size * col_size

    # Initialize video readers
    readers = [
        (
            imageio.get_reader(path)
            if path.endswith(tuple(VIDEO_EXTENSIONS))
            else dir_reader(path)
        )
        for path in input_paths
    ]

    # Initialize video writer
    writer = imageio.get_writer(output_path, fps=fps)

    # Initialize font
    font_path = os.path.join(os.path.dirname(__file__), "arial.ttf")
    font = ImageFont.truetype(font_path, text_size)

    # batch reader by batch_size
    readers_batched = [
        readers[i : min(i + num_per_batch, len(readers))]
        for i in range(0, len(readers), num_per_batch)
    ]
    labels_batched = [
        labels[i : min(i + num_per_batch, len(labels))]
        for i in range(0, len(labels), num_per_batch)
    ]

    for readers, labels in zip(readers_batched, labels_batched):
        for frames in zip(*readers):
            # Convert frames to PIL Images
            pil_imgs = [Image.fromarray(frame) for frame in frames]

            # Draw text on each image with text size
            for pil_img, label in zip(pil_imgs, labels):
                ImageDraw.Draw(pil_img).text(org, label, fill=color, font=font)

            # Concatenate the images into a single image
            img = concat_images(pil_imgs, row_size, col_size)

            # Convert the image back to a numpy array
            labeled_frame = np.array(img)

            # Append the labeled frame to the output video
            writer.append_data(labeled_frame)

    # Close the writer to finalize the video
    writer.close()
    # Close the readers
    for reader in readers:
        reader.close()


def convert_to_labeled_grid_image(
    input_paths,
    output_path,
    labels,
    org=(30, 30),
    color=(0, 0, 0),
    text_size=32,
    row_size=None,
    col_size=None,
):
    """
    Label a grid of images with text.

    Args:
        input_paths (list): List of input image paths.
        output_path (str): Output image path.
        labels (list): List of text labels.
        org (tuple): Origin of the text from the top-left corner.
        color (tuple): Color of the text.
        row_size (int): Number of rows.
        col_size (int): Number of columns.

    Returns:
        None
    """

    def adjust_grid_size(num_images, row_size=None, col_size=None):
        if row_size is None and col_size is None:
            row_size = int(np.sqrt(num_images))
            col_size = int(np.ceil(num_images / row_size))
        elif row_size is None:
            col_size = max(col_size, 1)
            row_size = int(np.ceil(num_images / col_size))
        elif col_size is None:
            row_size = max(row_size, 1)
            col_size = int(np.ceil(num_images / row_size))
        elif row_size * col_size < num_images:
            print_warning("row_size * col_size < num_images.")
            col_size = max(col_size, 1)
            row_size = int(np.ceil(num_images / col_size))
        return row_size, col_size

    # Check if input paths are valid
    for path in input_paths:
        assert os.path.exists(path), f"Input path {path} does not exist."

    # Check if input paths and labels have the same length
    assert len(input_paths) == len(
        labels
    ), f"Number of input paths and labels should be the same, but got {len(input_paths)} and {len(labels)}."

    # Adjust grid size
    num_images = len(input_paths)
    row_size, col_size = adjust_grid_size(num_images, row_size, col_size)

    # Initialize images
    images = [Image.open(path) for path in input_paths]

    # Initialize font
    # font = ImageFont.truetype("arial.ttf", text_size)
    # path is relative to this source file
    font_path = os.path.join(os.path.dirname(__file__), "arial.ttf")
    font = ImageFont.truetype(font_path, text_size)

    # Draw text on each image with text size
    for i, (img, label) in enumerate(zip(images, labels)):
        #ImageDraw.Draw(img).text(org, label, fill=color, font=font)
        # black text with white outline
        ImageDraw.Draw(img).text(org, label, fill=(0,0,0), font=font, stroke_width=max(int(text_size//16),1), stroke_fill=(255, 255, 255))

    # Concatenate the images into a single image
    img = concat_images(images, row_size, col_size)

    # Save the labeled image
    img.save(output_path)
