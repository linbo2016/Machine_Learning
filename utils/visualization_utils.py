import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms

def get_model_attention(model, image_tensor, model_type):
    """獲取模型的注意力圖
    
    參數:
        model: 訓練好的模型
        image_tensor: 輸入圖像張量 (1, C, H, W)
        model_type: 'cbam' 或 'ra'
    
    返回:
        attention_maps: 注意力圖列表
    """
    device = next(model.parameters()).device
    model.eval()
    image_tensor = image_tensor.to(device)
    
    attention_maps = []
    
    with torch.no_grad():
        if model_type == 'cbam':
            # 對於 CBAM 模型，我們需要逐層處理
            
            # 處理第一個卷積塊
            x = model.conv_block1[0](image_tensor)  # 第一個卷積層
            x = model.conv_block1[1](x)             # 批標準化
            x = model.conv_block1[2](x)             # ReLU
            x = model.conv_block1[3](x)             # 第二個卷積塊
            
            # 獲取第一個 CBAM 注意力
            _, attn1 = model.conv_block1[4](x, return_attention=True)
            attention_maps.append(('block1', attn1))
            
            # 繼續通過模型的其他層
            x = model.conv_block1[5](model.conv_block1[4](x))  # MaxPool
            
            # 處理第二個卷積塊
            x = model.conv_block2[0](x)
            x = model.conv_block2[1](x)
            x = model.conv_block2[2](x)
            x = model.conv_block2[3](x)
            
            # 獲取第二個 CBAM 注意力
            _, attn2 = model.conv_block2[4](x, return_attention=True)
            attention_maps.append(('block2', attn2))
            
            # 可以繼續處理更多層獲取更多注意力圖...
            
        elif model_type == 'ra':
            # 對於 Residual Attention 模型
            
            # 處理第一個卷積層
            x = model.conv1(image_tensor)
            
            # 獲取第一階段注意力
            _, attn1 = model.stage1(x, return_attention=True)
            attention_maps.append(('stage1', attn1))
            
            # 繼續通過模型的其他層
            x = model.downsample1(model.stage1(x))
            
            # 獲取第二階段注意力
            _, attn2 = model.stage2(x, return_attention=True)
            attention_maps.append(('stage2', attn2))
            
            # 可以繼續處理更多層...
    
    return attention_maps

def visualize_attention_maps(image, attention_maps, model_type, output_path=None):
    """可視化注意力圖
    
    參數:
        image: 原始圖像 (H, W, C), numpy array
        attention_maps: 從模型獲取的注意力圖
        model_type: 'cbam' 或 'ra'
        output_path: 保存可視化結果的路徑
    """
    plt.figure(figsize=(20, 10))
    
    # 顯示原始圖像
    plt.subplot(1, len(attention_maps) + 1, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    # 處理不同模型的注意力圖
    if model_type == 'cbam':
        for i, (name, attn_dict) in enumerate(attention_maps):
            # 對於 CBAM，我們主要關注空間注意力
            spatial_attn = attn_dict['spatial_attention'].squeeze().cpu().numpy()
            
            plt.subplot(1, len(attention_maps) + 1, i + 2)
            plt.title(f'{name} - Spatial Attention')
            plt.imshow(spatial_attn, cmap='jet')
            plt.axis('off')
            
            # 疊加在原圖上
            plt.subplot(1, len(attention_maps) + 1 + len(attention_maps), i + 2 + len(attention_maps))
            plt.title(f'Overlay - {name}')
            
            # 調整注意力圖的大小以匹配原圖
            resized_attn = cv2.resize(spatial_attn, (image.shape[1], image.shape[0]))
            plt.imshow(image)
            plt.imshow(resized_attn, alpha=0.5, cmap='jet')
            plt.axis('off')
    else:  # Residual Attention
        for i, (name, attn) in enumerate(attention_maps):
            attn = attn.squeeze().cpu().numpy()
            
            plt.subplot(1, len(attention_maps) + 1, i + 2)
            plt.title(f'{name} - Attention')
            plt.imshow(attn, cmap='jet')
            plt.axis('off')
            
            # 疊加在原圖上
            plt.subplot(1, len(attention_maps) + 1 + len(attention_maps), i + 2 + len(attention_maps))
            plt.title(f'Overlay - {name}')
            
            # 調整注意力圖的大小以匹配原圖
            resized_attn = cv2.resize(attn, (image.shape[1], image.shape[0]))
            plt.imshow(image)
            plt.imshow(resized_attn, alpha=0.5, cmap='jet')
            plt.axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"保存可視化結果到: {output_path}")
    plt.show()

def process_image_for_model(image_path, transform=None):
    """處理圖像以供模型使用
    
    參數:
        image_path: 圖像文件路徑
        transform: 圖像轉換
    
    返回:
        image_tensor: 處理後的圖像張量
        original_image: 原始圖像
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 讀取圖像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # PIL 轉換
    from PIL import Image
    pil_image = Image.fromarray(image)
    
    # 應用轉換
    image_tensor = transform(pil_image).unsqueeze(0)  # 添加批次維度
    
    return image_tensor, original_image

def compare_model_attentions(image_path, cbam_model, ra_model, output_dir, transform=None):
    """比較兩個模型的注意力機制
    
    參數:
        image_path: 圖像文件路徑
        cbam_model: CBAM 模型
        ra_model: Residual Attention 模型
        output_dir: 輸出目錄
        transform: 圖像轉換
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 處理圖像
    image_tensor, original_image = process_image_for_model(image_path, transform)
    
    # 獲取 CBAM 注意力圖
    cbam_attention_maps = get_model_attention(cbam_model, image_tensor, 'cbam')
    
    # 獲取 Residual Attention 注意力圖
    ra_attention_maps = get_model_attention(ra_model, image_tensor, 'ra')
    
    # 獲取圖像文件名（不帶路徑和擴展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 可視化 CBAM 注意力圖
    cbam_output_path = os.path.join(output_dir, f'{image_name}_cbam_attention.png')
    visualize_attention_maps(original_image, cbam_attention_maps, 'cbam', cbam_output_path)
    
    # 可視化 Residual Attention 注意力圖
    ra_output_path = os.path.join(output_dir, f'{image_name}_ra_attention.png')
    visualize_attention_maps(original_image, ra_attention_maps, 'ra', ra_output_path)
    
    return cbam_attention_maps, ra_attention_maps

def process_test_images(cbam_model, ra_model, test_loader, output_dir, num_samples=5):
    """處理測試集中的圖像以進行注意力可視化比較"""
    os.makedirs(output_dir, exist_ok=True)
    device = next(cbam_model.parameters()).device
    
    # 獲取樣本和標籤
    all_samples = []
    all_labels = []
    
    for images, labels in test_loader:
        for i in range(len(images)):
            all_samples.append(images[i])
            all_labels.append(labels[i].item())
            if len(all_samples) >= num_samples:
                break
        if len(all_samples) >= num_samples:
            break
    
    # 為每個樣本生成注意力可視化
    for i, (image, label) in enumerate(zip(all_samples, all_labels)):
        print(f"處理樣本 {i+1}/{num_samples} (標籤: {label} - {'良性' if label == 0 else '惡性'})")
        
        # 轉換為 numpy 圖像以便可視化
        img_np = image.permute(1, 2, 0).cpu().numpy()
        
        # 反標準化 (如果在數據加載時應用了標準化)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        # 將圖像轉換為模型輸入格式
        img_tensor = image.unsqueeze(0).to(device)
        
        try:
            # 獲取 CBAM 注意力圖
            print("處理 CBAM 模型注意力...")
            cbam_attention_maps = get_model_attention(cbam_model, img_tensor, 'cbam')
            
            # 可視化 CBAM 注意力圖
            cbam_output_path = os.path.join(output_dir, f'sample_{i+1}_label_{label}_cbam.png')
            visualize_attention_maps(img_np, cbam_attention_maps, 'cbam', cbam_output_path)
            print(f"CBAM 注意力圖已保存: {cbam_output_path}")
            
            # 獲取 Residual Attention 注意力圖
            print("處理 Residual Attention 模型注意力...")
            ra_attention_maps = get_model_attention(ra_model, img_tensor, 'ra')
            
            # 可視化 Residual Attention 注意力圖
            ra_output_path = os.path.join(output_dir, f'sample_{i+1}_label_{label}_ra.png')
            visualize_attention_maps(img_np, ra_attention_maps, 'ra', ra_output_path)
            print(f"Residual Attention 注意力圖已保存: {ra_output_path}")
            
        except Exception as e:
            print(f"處理樣本 {i+1} 時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            continue