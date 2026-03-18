import pytest
import torch
from PIL import Image

from zerokey.models.molmo import Molmo


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Molmo requires a GPU")
def test_molmo_minimal():
    model_id = "allenai/Molmo-7B-D-0924"
    print(f"Loading model {model_id}...", flush=True)

    molmo = Molmo(model_path=model_id)
    print("Model loaded successfully.", flush=True)

    image = Image.new("RGB", (100, 100), color="white")

    prompt = "point to the center of this white image"
    print(f"Prompt: {prompt}", flush=True)

    result = molmo.generated_kps_points(image, prompt)
    print(f"Result: {result}", flush=True)

    try:
        kps, alt = molmo.parse_points_str(result)
        print(f"Parsed KPs: {kps}", flush=True)
        print(f"Alt: {alt}", flush=True)
    except Exception as e:
        print(f"Parsing note (output may not be XML): {e}", flush=True)
