from importlib import import_module
import requests
from utils import rotation_utils as rot_utils
import torch


def get_object_from_path(path):
    assert type(path) is str
    mod_path = '.'.join(path.split('.')[:-1])
    object_name = path.split('.')[-1]
    mod = import_module(mod_path)
    target_obj = getattr(mod, object_name)
    return target_obj


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def preprocess_input_data(images, labels, rotation=True):
    """Preprocess a mini-batch of images."""
    if rotation:
        # Create the 4 rotated version of the images; this step increases
        # the batch size by a multiple of 4.
        batch_size_in = images.size(0)
        images = rot_utils.create_4rotations_images(images)
        labels_rotation = rot_utils.create_rotations_labels(batch_size_in, images.device)
        labels = labels.repeat(4)
    return images, labels, labels_rotation


def load_vissl_weights(model, checkpoints_path):
    checkpoint = torch.load(checkpoints_path)
    updated_checkpoints_dict = {}
    for key in checkpoint:
        updated_checkpoints_dict[f"model.{key}"] = checkpoint[key]
    model.load_state_dict(updated_checkpoints_dict, strict=False)

    return model
