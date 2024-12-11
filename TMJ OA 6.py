import os
from PIL import Image

# 폴더 경로 설정
input_folder = 'C:/TMJ OA/masked_data'  # 원본 이미지 폴더
output_folder = 'C:/TMJ OA/masked_2_data' # 결과 저장 폴더

# 크롭 영역 정의 (모든 이미지에 동일한 영역 적용)
crop_box = (110, 45, 265, 250)

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 이미지 일괄 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # 이미지 파일 필터링
        # 이미지 로드
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)

        # 크롭 수행
        cropped_img = img.crop(crop_box)

        # 크롭된 이미지 저장
        output_path = os.path.join(output_folder, filename)
        cropped_img.save(output_path)

print(f"All images cropped and saved to {output_folder}")