
import time
from transformers import pipeline
from PIL import Image
import requests

image = Image.open("segmentation_input.jpg")
image

#semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
model_path = "/cluster/home/wueestm/.cache/huggingface/hub/models--nvidia--segformer-b1-finetuned-cityscapes-1024-1024/snapshots/ec86afeba68e656629ccf47e0c8d2902f964917b"
semantic_segmentation = pipeline("image-segmentation", model=model_path)


# Measure inference time
start_time = time.time()
results = semantic_segmentation(image)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")