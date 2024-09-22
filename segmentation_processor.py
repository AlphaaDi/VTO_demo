from gradio_client import Client, handle_file
import json
import numpy as np
import tempfile

def request_segmentation_results(url, image, model_name="1b", api_name="/process_image"):
    client = Client(url)

    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        image.save(temp_file.name, format="PNG")
        _, npy_path, json_path = client.predict(
                image=handle_file(temp_file.name),
                model_name=model_name,
                api_name=api_name
        )

    segmentation_map = np.load(npy_path)

    with open(json_path, 'r') as f:
        classes_mapping = json.load(f)

    return segmentation_map, classes_mapping

def extract_submask(segmentation_map, submask_classes, classes_mapping):
    """
    Creates a submask of the union of the classes to preserve.

    Args:
    segmentation_map (numpy.ndarray): A 2D array of argmax indices representing the segmentation map.
    submask_classes (list): List of class.
    classes_mapping (dict): A dictionary mapping class names to their corresponding index values.

    Returns:
    numpy.ndarray: A binary submask where the classes to preserve are set to 1 and the rest are 0.
    """
    # Convert the class names to indices using the class mapping
    class_indices_to_preserve = [classes_mapping[cls] for cls in submask_classes]

    # Create a binary mask where classes to preserve are 1, and others are 0
    submask = np.isin(segmentation_map, class_indices_to_preserve).astype(np.uint8) > 0

    return submask