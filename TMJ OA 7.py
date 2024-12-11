from PIL import Image
import matplotlib.pyplot as plt

# 이미지 로드
image_path = 'C:/TMJ OA/masked_data/20703917 1.JPG'
img = Image.open(image_path)

# 크롭 영역 정의 (좌표: (left, upper, right, lower))
crop_box = (110, 45, 265, 250)

# 크롭 수행
cropped_img = img.crop(crop_box)

# 크롭된 이미지 시각화
plt.imshow(cropped_img)
plt.title("Cropped Image")
plt.axis("off")
plt.show()