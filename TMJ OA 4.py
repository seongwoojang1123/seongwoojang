import cv2
import os

# 원본 및 출력 디렉토리 설정
source_dir = r'C:/TMJ OA_2/source_data'
output_base_dir = r'C:/TMJ OA_2/masked_data'

# 텍스트 영역을 수동으로 마스킹하는 함수
def manual_text_masking(image):
    masked_image = image.copy()

    # 텍스트가 위치한 영역을 수동으로 지정하여 마스킹 (픽셀 값은 예시로 조정 가능)
    # 상단 텍스트 영역 마스킹
    masked_image[0:20, :] = 0  # 상단 0~50 픽셀 범위 마스킹

    # 하단 텍스트 영역 마스킹
    height = masked_image.shape[0]
    masked_image[height-20:height, :] = 0  # 하단 마지막 50 픽셀 범위 마스킹

    # 좌측 텍스트 영역 마스킹
    masked_image[:, 0:80] = 0  # 좌측 0~100 픽셀 범위 마스킹

    # 우측 텍스트 영역 마스킹
    width = masked_image.shape[1]
    masked_image[:, width-80:] = 0  # 우측 마지막 100 픽셀 범위 마스킹

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
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)

            if image is not None:
                # 수동으로 텍스트 영역 마스킹
                masked_image = manual_text_masking(image)
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
