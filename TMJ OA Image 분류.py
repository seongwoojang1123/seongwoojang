import os
import shutil
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# Excel 파일 경로 및 이미지 경로 설정
excel_path = 'C:/Users/mook6/Downloads/20240819 CBCT MRI 데이터 정리_수정1.xlsx'
source_dir = 'C:/TMJ OA/masked_data'
train_dir = 'C:/TMJ OA/train_data'
validation_dir = 'C:/TMJ OA/validation_data'
test_dir = 'C:/TMJ OA/test_data'

# Excel 파일 로드 및 OA 상태를 딕셔너리로 저장 (ID별로 분류)
oa_data = pd.read_excel(excel_path)
oa_dict = {row['Patient ID']: (row['CBCT_Rt OA (Osteoarthritis) (0=normal, 1=OA)'],
                               row['CBCT_Lt OA  (Osteoarthritis) (0=normal, 1=OA)'])
           for _, row in oa_data.iterrows()}

# 이미지 필터링 (우측: 1.JPG, 좌측: 3.JPG 이미지만 선택)
all_files = [f for f in os.listdir(source_dir) if f.endswith('1.JPG') or f.endswith('3.JPG')]

# 레이블에 따른 분류
rt_oa_files = [f for f in all_files if f.endswith('1.JPG') and oa_dict.get(int(f.split(' ')[0]), (0, 0))[0] == 1]
rt_normal_files = [f for f in all_files if f.endswith('1.JPG') and oa_dict.get(int(f.split(' ')[0]), (0, 0))[0] == 0]
lt_oa_files = [f for f in all_files if f.endswith('3.JPG') and oa_dict.get(int(f.split(' ')[0]), (0, 0))[1] == 1]
lt_normal_files = [f for f in all_files if f.endswith('3.JPG') and oa_dict.get(int(f.split(' ')[0]), (0, 0))[1] == 0]


# 함수: 각 레이블 비율에 맞춰 데이터를 분할
def split_data(files, train_ratio=0.8, val_ratio=0.1):
    train_files, temp_files = train_test_split(files, train_size=train_ratio, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=val_ratio / (1 - train_ratio), random_state=42)
    return train_files, val_files, test_files


# 각 레이블에 따라 분할
train_rt_oa, val_rt_oa, test_rt_oa = split_data(rt_oa_files)
train_rt_normal, val_rt_normal, test_rt_normal = split_data(rt_normal_files)
train_lt_oa, val_lt_oa, test_lt_oa = split_data(lt_oa_files)
train_lt_normal, val_lt_normal, test_lt_normal = split_data(lt_normal_files)

# 분할된 파일들을 합치기
train_files = train_rt_oa + train_rt_normal + train_lt_oa + train_lt_normal
validation_files = val_rt_oa + val_rt_normal + val_lt_oa + val_lt_normal
test_files = test_rt_oa + test_rt_normal + test_lt_oa + test_lt_normal

# 학습, 검증, 테스트 폴더 생성
for label_type in ['Rt_Normal', 'Lt_Normal', 'Rt_OA', 'Lt_OA']:
    os.makedirs(os.path.join(train_dir, label_type), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, label_type), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label_type), exist_ok=True)


# 파일 복사 함수
def copy_files(file_list, target_dir, oa_dict):
    for filename in file_list:
        file_path = os.path.join(source_dir, filename)
        img_id = int(filename.split(' ')[0])  # ID가 첫 부분에 있다고 가정

        # 우측/좌측 및 OA 상태에 따른 폴더 설정
        if filename.endswith('1.JPG'):  # 우측
            oa_status = oa_dict.get(img_id, (0, 0))[0]  # 우측 OA 상태
            label = 'Rt_OA' if oa_status == 1 else 'Rt_Normal'
        elif filename.endswith('3.JPG'):  # 좌측
            oa_status = oa_dict.get(img_id, (0, 0))[1]  # 좌측 OA 상태
            label = 'Lt_OA' if oa_status == 1 else 'Lt_Normal'

        # 파일 복사
        target_label_dir = os.path.join(target_dir, label)
        shutil.copy(file_path, os.path.join(target_label_dir, filename))


# 파일 복사 수행
copy_files(train_files, train_dir, oa_dict)
copy_files(validation_files, validation_dir, oa_dict)
copy_files(test_files, test_dir, oa_dict)

# 결과 출력
print(f"학습 데이터로 복사된 파일 수: {len(train_files)}")
print(f"검증 데이터로 복사된 파일 수: {len(validation_files)}")
print(f"테스트 데이터로 복사된 파일 수: {len(test_files)}")
