import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

# 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 이미지 로드
image = plt.imread("horse.jpg")  # horse.jpg 파일 필요

# 입력 텍스트 및 이미지 처리
inputs = processor(
    text=["a photo of a horse", "a photo of a dog", 
          "a photo of a bear", "a photo of a person"], 
    images=image, return_tensors="pt", padding=True
)

# 모델 추론
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# 결과 시각화
plt.imshow(image)
plt.title(f"Probabilities: horse: {probs[0,0]:.2f}, dog: {probs[0,1]:.2f}, bear: {probs[0,2]:.2f}, person: {probs[0,3]:.2f}")
plt.show()
