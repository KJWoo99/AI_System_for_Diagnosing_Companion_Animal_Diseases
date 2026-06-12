import os

def print_folder_structure(path, indent=0):
    """
    폴더 구조를 재귀적으로 출력 (파일 제외)
    마지막 폴더에는 파일 개수 표시
    """
    try:
        items = os.listdir(path)
    except PermissionError:
        return

    # 폴더만 필터
    folders = [f for f in items if os.path.isdir(os.path.join(path, f))]

    for folder in folders:
        folder_path = os.path.join(path, folder)

        try:
            sub_items = os.listdir(folder_path)
        except PermissionError:
            continue

        sub_folders = [f for f in sub_items if os.path.isdir(os.path.join(folder_path, f))]

        # 마지막 폴더(leaf folder)인지 확인
        if not sub_folders:
            # 파일 개수 세기
            file_count = len([
                f for f in sub_items
                if os.path.isfile(os.path.join(folder_path, f))
            ])

            print('│   ' * indent + f'├── {folder} --- {file_count}개')
        else:
            print('│   ' * indent + '├── ' + folder)
            print_folder_structure(folder_path, indent + 1)


# 시작 경로
root_path = "."

print(root_path)
print_folder_structure(root_path)