import os

def count_images(data_path, magnification='40X'):
    root = os.path.join(data_path, magnification)
    count = {
        'benign_total': 0,
        'malignant_total': 0,
        'benign_subtypes': {},
        'malignant_subtypes': {}
    }

    for category in ['benign', 'malignant']:
        category_path = os.path.join(root, category)
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            n = len(files)

            if category == 'benign':
                count['benign_total'] += n
                count['benign_subtypes'][subfolder] = n
            else:
                count['malignant_total'] += n
                count['malignant_subtypes'][subfolder] = n

    total = count['benign_total'] + count['malignant_total']
    print(f"Total images: {total}")
    print(f"Benign total: {count['benign_total']}")
    print(f"Malignant total: {count['malignant_total']}")
    return count

# 使用你的資料路徑
data_path = r"C:\linbo\Structural_Machine_Learning_Models_and_Their_Applications\cmba\BreaKHis_v1\BreaKHis_v1\histology_slides"  # <- 換成你的實際路徑
count_images(data_path, magnification='40X')
