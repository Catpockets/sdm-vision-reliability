"""Import the research dependencies and exercise every available device."""

import json
import platform

import numpy as np
import PIL
from PIL import Image
import scipy
import torch
import torchvision


def main():
    devices = ["cpu"]
    devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    if torch.backends.mps.is_available():
        devices.append("mps")

    # Exercise NumPy/Pillow/torchvision interop and a real tensor operation.
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    tensor = torchvision.transforms.functional.to_tensor(image)
    for device in devices:
        result = (tensor.to(device) + 1).sum().item()
        if result != 3 * 32 * 32:
            raise RuntimeError(f"Tensor smoke check failed on {device}: {result}")

    print(json.dumps({
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(i)
                         for i in range(torch.cuda.device_count())],
        "tested_devices": devices,
        "image_shape": list(tensor.shape),
        "status": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
