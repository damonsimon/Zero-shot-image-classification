from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 이미지 로드
image = Image.open("horse.jpg")  # 이미지 파일 경로

# Processor 호출
inputs = processor(
    text=["a photo of a horse", "a photo of a dog", 
          "a photo of a bear", "a photo of a person"], 
    images=image, return_tensors="pt", padding=True
)

