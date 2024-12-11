import os
import pandas as pd

# Excel 파일 경로 및 이미지 폴더 경로 설정
excel_path = 'C:/Users/mook6/OneDrive/Documents/장성우/대학원/프로젝트/TMJ_OA/2024 CBCT and MRI/20240819 CBCT MRI 데이터 정리_수정.xlsx'
base_dir = 'C:/TMJ OA_2/'
train_dir = os.path.join(base_dir, 'train_data')
validation_dir = os.path.join(base_dir, 'validation_data')
test_dir = os.path.join(base_dir, 'test_data')

# Excel 파일 로드 및 불필요한 텍스트와 NaN 제거
oa_data = pd.read_excel(excel_path, usecols=['Patient ID', 'NAME', 'CBCT_Rt OA (Osteoarthritis) (0=normal, 1=OA)', 'CBCT_Lt OA  (Osteoarthritis) (0=normal, 1=OA)'])

# Patient ID와 Name 모두에 대해 NaN 값이 없는 유효한 데이터만 필터링
oa_data = oa_data.dropna(subset=['Patient ID', 'NAME'])  # NaN 값 제거
oa_data = oa_data[pd.to_numeric(oa_data['Patient ID'], errors='coerce').notna()]  # Patient ID가 숫자인 행만 남김
oa_data['Patient ID'] = oa_data['Patient ID'].astype(str)  # 문자열로 변환하여 일관성 유지

# Patient ID와 Name 정보 딕셔너리 생성
oa_info = {row['Patient ID']: row for _, row in oa_data.iterrows()}
excel_patient_ids = set(oa_info.keys())

# 이미지 폴더에서 모든 Patient ID 추출
all_image_patient_ids = set()
for folder in [train_dir, validation_dir, test_dir]:
    if os.path.exists(folder):
        for label_type in os.listdir(folder):
            label_path = os.path.join(folder, label_type)
            if os.path.isdir(label_path):
                all_image_patient_ids.update(file.split(' ')[0] for file in os.listdir(label_path))

# Excel에 있지만 이미지 폴더에 없는 Patient ID와 NAME 정보
missing_in_images = excel_patient_ids - all_image_patient_ids
missing_in_images_info = {pid: oa_info[pid]['NAME'] for pid in missing_in_images}

# 이미지 폴더에 있지만 Excel에 없는 Patient ID
missing_in_excel = all_image_patient_ids - excel_patient_ids
missing_in_excel_count = len(missing_in_excel)

# 결과 출력
print(f"Excel에는 있지만 이미지 폴더에 없는 Patient ID 개수: {len(missing_in_images)}")
print("Excel에는 있지만 이미지 폴더에 없는 Patient ID와 NAME 정보:", missing_in_images_info)

print(f"\n이미지 폴더에는 있지만 Excel에 없는 Patient ID 개수: {missing_in_excel_count}")
print("이미지 폴더에는 있지만 Excel에 없는 Patient ID:", missing_in_excel)