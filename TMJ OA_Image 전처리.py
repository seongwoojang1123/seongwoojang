import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 원본 및 출력 디렉토리 설정
source_dir = r'C:/TMJ OA/source_data'
output_base_dir = r'C:/TMJ OA/masked_data'


# 텍스트 영역 마스킹 함수
def remove_text_only(image):
    masked_image = image.copy()
    masked_image[0:20, :] = 0  # 상단 10~30 픽셀 범위 마스킹
    height = masked_image.shape[0]
    masked_image[height - 20:height - 0, :] = 0  # 하단 30~10 픽셀 범위 마스킹
    masked_image[:, 0:95] = 0  # 좌측 0~90 픽셀 범위 마스킹
    width = masked_image.shape[1]
    masked_image[:, width - 95:] = 0  # 우측 90 픽셀 범위 마스킹
    return masked_image


# 이미지 전처리 및 저장 함수
def process_images_in_directory(input_dir, output_base_dir):
    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        output_dir = os.path.join(output_base_dir, relative_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"Processing directory: {output_dir}")

        for filename in files:
            # 확장자 조건 제거 (모든 파일을 시도)
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)

            if image is not None:
                # 텍스트 영역 제거
                masked_image = remove_text_only(image)
                output_path = os.path.join(output_dir, filename)
                success = cv2.imwrite(output_path, masked_image)

                if success:
                    print(f"Image successfully saved to {output_path}")
                else:
                    print(f"Failed to save image at: {output_path}")
            else:
                print(f"Failed to load image {image_path} - It might not be a valid image file.")


# 전체 이미지 전처리 수행
process_images_in_directory(source_dir, output_base_dir)
