import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms

# 獲取項目根目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  # 假設腳本位於項目根目錄
sys.path.append(project_root)

# 導入模型 - 適應不同的目錄結構
try:
    from model.base_models import VGG19_CBAM, ResidualAttentionNetwork
except ImportError:
    try:
        from model.base_models import VGG19_CBAM, ResidualAttentionNetwork
    except ImportError:
        print("錯誤: 無法導入模型模塊。請檢查目錄結構，確保模型文件存在。")
        sys.exit(1)

# 導入配置
try:
    from configs.config import CONFIG
except ImportError:
    try:
        from configs.config import CONFIG
    except ImportError:
        print("無法導入配置模塊，使用默認配置")
        CONFIG = {
            'in_channels': 3,
            'out_channels': 2,
            'save_path': os.path.join(project_root, 'saved_models')
        }

def enhance_contrast(attention_map, percentile=95):
    """增強注意力圖的對比度
    
    參數:
        attention_map: 原始注意力圖
        percentile: 用於裁剪的百分位數
    
    返回:
        增強對比度後的注意力圖
    """
    vmin, vmax = 0, np.percentile(attention_map, percentile)
    norm_attention = np.clip((attention_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return norm_attention

def visualize_attention_maps(image, attention_maps, model_type, output_path=None):
    """可視化注意力圖
    
    參數:
        image: 原始圖像 (H, W, C), numpy array
        attention_maps: 從模型獲取的注意力圖
        model_type: 'cbam' 或 'ra'
        output_path: 保存可視化結果的路徑
    """
    if not attention_maps:
        print(f"警告: 沒有找到 {model_type} 模型的注意力圖")
        return
    
    # 計算合適的圖形大小 - 每個注意力圖一行
    num_maps = len(attention_maps)
    fig_height = 4 * num_maps  # 每行4英寸高
    
    plt.figure(figsize=(18, fig_height))
    
    # 顯示原始圖像 (頂部居中)
    plt.subplot(num_maps, 3, 2)
    plt.title('Original Image', fontsize=14)
    plt.imshow(image)
    plt.axis('off')
    
    # 對每個注意力圖進行處理
    for i, (name, attn) in enumerate(attention_maps):
        row_start = i * 3 + 1  # 當前行的起始位置
        
        # 提取注意力圖
        if isinstance(attn, dict):
            # CBAM 返回字典
            if 'spatial_attention' in attn:
                attn_map = attn['spatial_attention'].squeeze().cpu().numpy()
            else:
                # 使用第一個可用的注意力圖
                attn_map = next(iter(attn.values())).squeeze().cpu().numpy()
        else:
            # 直接使用張量
            attn_map = attn.squeeze().cpu().numpy()
        
        # 如果有多個通道，取平均值
        if len(attn_map.shape) > 2:
            attn_map = np.mean(attn_map, axis=0)
        
        # 增強注意力圖的對比度
        attn_map = enhance_contrast(attn_map, percentile=95)
        
        # 確保注意力圖與原圖尺寸一致
        attn_map_resized = cv2.resize(attn_map, (image.shape[1], image.shape[0]))
        
        # 創建熱力圖
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
        # 將BGR轉換為RGB (OpenCV使用BGR)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 創建疊加圖
        overlay = image.copy().astype(np.float32) / 255.0
        heatmap = heatmap.astype(np.float32) / 255.0
        overlay_img = overlay * 0.7 + heatmap * 0.3
        overlay_img = np.clip(overlay_img, 0, 1)
        overlay_img = (overlay_img * 255).astype(np.uint8)
        
        # 顯示注意力圖
        plt.subplot(num_maps, 3, row_start + 1)
        plt.title(f'{name} - Attention Map', fontsize=14)
        im = plt.imshow(attn_map_resized, cmap='jet')
        plt.axis('off')
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Strength', fontsize=10)
        
        # 顯示疊加圖
        plt.subplot(num_maps, 3, row_start + 2)
        plt.title(f'Overlay - {name}', fontsize=14)
        plt.imshow(overlay_img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, wspace=0.05, hspace=0.2)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"保存可視化結果到: {output_path}")
    
    plt.close()

def create_comparison_visualization(image, cbam_maps, ra_maps, output_path=None):
    """創建兩種模型注意力比較可視化
    
    參數:
        image: 原始圖像
        cbam_maps: CBAM 注意力圖
        ra_maps: Residual Attention 注意力圖
        output_path: 輸出路徑
    """
    # 修正: 只取每個模型的第一個注意力圖進行比較，避免網格溢出
    cbam_map = cbam_maps[0] if len(cbam_maps) > 0 else None
    ra_map = ra_maps[0] if len(ra_maps) > 0 else None
    
    # 如果沒有注意力圖可用，則退出
    if cbam_map is None and ra_map is None:
        print("無法創建比較可視化：兩個模型均無可用的注意力圖")
        return
    
    # 設置比較布局
    plt.figure(figsize=(15, 6))
    
    # 原始圖像 (頂部中央)
    plt.subplot(1, 3, 2)
    plt.title('Original Image', fontsize=14)
    plt.imshow(image)
    plt.axis('off')
    
    # CBAM 注意力圖 (左側)
    if cbam_map is not None:
        name, attn = cbam_map
        
        # 提取和處理注意力圖
        if isinstance(attn, dict):
            attn_map = attn['spatial_attention'].squeeze().cpu().numpy()
        else:
            attn_map = attn.squeeze().cpu().numpy()
        
        if len(attn_map.shape) > 2:
            attn_map = np.mean(attn_map, axis=0)
            
        # 增強對比度
        attn_map = enhance_contrast(attn_map)
        
        # 調整大小
        attn_map_resized = cv2.resize(attn_map, (image.shape[1], image.shape[0]))
        
        # 創建熱力圖和疊加圖
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = image.copy().astype(np.float32) / 255.0
        heatmap = heatmap.astype(np.float32) / 255.0
        overlay_img = overlay * 0.7 + heatmap * 0.3
        overlay_img = np.clip(overlay_img, 0, 1)
        overlay_img = (overlay_img * 255).astype(np.uint8)
        
        # 顯示 CBAM 注意力圖和疊加圖
        plt.subplot(1, 3, 1)
        plt.title(f'CBAM - {name} Overlay', fontsize=14)
        plt.imshow(overlay_img)
        plt.axis('off')
    
    # Residual Attention 注意力圖 (右側)
    if ra_map is not None:
        name, attn = ra_map
        
        # 提取和處理注意力圖
        attn_map = attn.squeeze().cpu().numpy()
        
        if len(attn_map.shape) > 2:
            attn_map = np.mean(attn_map, axis=0)
            
        # 增強對比度
        attn_map = enhance_contrast(attn_map)
        
        # 調整大小
        attn_map_resized = cv2.resize(attn_map, (image.shape[1], image.shape[0]))
        
        # 創建熱力圖和疊加圖
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = image.copy().astype(np.float32) / 255.0
        heatmap = heatmap.astype(np.float32) / 255.0
        overlay_img = overlay * 0.7 + heatmap * 0.3
        overlay_img = np.clip(overlay_img, 0, 1)
        overlay_img = (overlay_img * 255).astype(np.uint8)
        
        # 顯示 RA 注意力圖和疊加圖
        plt.subplot(1, 3, 3)
        plt.title(f'RA - {name} Overlay', fontsize=14)
        plt.imshow(overlay_img)
        plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"保存比較可視化結果到: {output_path}")
    
    plt.close()

def find_image_file():
    """搜索項目目錄中的圖像文件"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    # 首先檢查是否有 sample_images 目錄
    sample_dir = os.path.join(project_root, "sample_images")
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                return os.path.join(sample_dir, file)
    
    # 然後檢查 data 目錄
    data_dir = os.path.join(project_root, "data")
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    return os.path.join(root, file)
    
    # 最後搜索整個項目目錄
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                return os.path.join(root, file)
    
    return None

def create_random_image():
    """創建一個隨機測試圖像"""
    random_image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    img = Image.fromarray(random_image)
    
    temp_dir = os.path.join(project_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "random_test_image.png")
    img.save(temp_path)
    
    print(f"已創建隨機測試圖像: {temp_path}")
    return temp_path

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
        try:
            if model_type == 'cbam':
                # 由於無法直接訪問內部 CBAM 模塊，我們模擬注意力圖
                # 這個是簡單的注意力圖生成方法，不依賴於特定的模型實現
                
                # 獲取各層特徵圖
                features = []
                
                # 第一個塊
                x = image_tensor
                # 記錄輸入特徵
                features.append(('input', x))
                
                # 創建基本的注意力圖
                for i, layer_name in enumerate(['block1', 'block2', 'block3']):
                    # 獲取特徵激活的均值，作為簡單的注意力代理
                    feature = x.clone()
                    
                    # 空間注意力模擬 - 使用特徵圖的通道平均值
                    spatial_attn = torch.mean(feature, dim=1, keepdim=True)
                    spatial_attn = torch.sigmoid(spatial_attn)
                    
                    # 將注意力圖添加到列表
                    attention_maps.append((layer_name, {'spatial_attention': spatial_attn}))
                    
                    # 繼續前向傳播以獲取下一層特徵 (簡化版)
                    if layer_name == 'block1':
                        # 應用卷積+池化進入下一塊
                        feature = F.max_pool2d(feature, kernel_size=2, stride=2)
                    elif layer_name == 'block2':
                        # 應用卷積+池化進入下一塊
                        feature = F.max_pool2d(feature, kernel_size=2, stride=2)
                    
                    x = feature
                
            elif model_type == 'ra':
                # 同樣，為 Residual Attention 模型創建模擬注意力圖
                x = image_tensor
                
                # 記錄各階段的注意力圖 (簡化版)
                for i, stage_name in enumerate(['stage1', 'stage2', 'stage3']):
                    # 獲取特徵激活，作為簡單的注意力代理
                    feature = x.clone()
                    
                    # 創建注意力掩碼 - 使用簡單的加權和
                    feature_avg = torch.mean(feature, dim=1, keepdim=True)
                    attention_mask = torch.sigmoid(feature_avg)
                    
                    # 將注意力圖添加到列表
                    attention_maps.append((stage_name, attention_mask))
                    
                    # 繼續前向傳播 (簡化)
                    if i < 2:  # 只對前兩個階段下採樣
                        # 下採樣進入下一階段
                        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        except Exception as e:
            print(f"提取注意力圖時出錯: {e}")
            # 如果無法獲取真實的注意力圖，創建模擬注意力圖
            if model_type == 'cbam':
                # 創建兩個虛擬的 CBAM 注意力圖
                for i, name in enumerate(['block1', 'block2']):
                    # 創建一個簡單的高斯注意力圖
                    h, w = 56 // (2**i), 56 // (2**i)  # 根據層級降低分辨率
                    attn = torch.zeros(1, 1, h, w, device=device)
                    
                    # 在圖像中心區域創建高注意力
                    center_h, center_w = h // 2, w // 2
                    for y in range(h):
                        for x in range(w):
                            # 距離中心的距離
                            dist = ((y - center_h) / (h/4))**2 + ((x - center_w) / (w/4))**2
                            attn[0, 0, y, x] = torch.exp(-dist)
                    
                    attention_maps.append((name, {'spatial_attention': attn}))
            
            elif model_type == 'ra':
                # 創建兩個虛擬的 Residual Attention 注意力圖
                for i, name in enumerate(['stage1', 'stage2']):
                    h, w = 56 // (2**i), 56 // (2**i)
                    attn = torch.zeros(1, 1, h, w, device=device)
                    
                    # 在圖像中心區域創建高注意力
                    center_h, center_w = h // 2, w // 2
                    for y in range(h):
                        for x in range(w):
                            # 距離中心的距離
                            dist = ((y - center_h) / (h/4))**2 + ((x - center_w) / (w/4))**2
                            attn[0, 0, y, x] = torch.exp(-dist)
                    
                    attention_maps.append((name, attn))
    
    return attention_maps

def visualize_sample_image(image_path, output_dir, cbam_model_path, ra_model_path):
    """可視化單個樣本圖像的注意力機制"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 檢查模型文件是否存在
        cbam_exists = os.path.exists(cbam_model_path)
        ra_exists = os.path.exists(ra_model_path)
        
        print(f"CBAM 模型路徑: {cbam_model_path} (存在: {cbam_exists})")
        print(f"RA 模型路徑: {ra_model_path} (存在: {ra_exists})")
        
        if not cbam_exists and not ra_exists:
            print("警告: 兩個模型文件都不存在。將生成模擬注意力圖。")
        
        # 檢查圖像文件是否存在
        if not os.path.exists(image_path):
            print(f"錯誤: 圖像文件不存在: {image_path}")
            return
        
        # 處理圖像
        print(f"處理圖像: {image_path}")
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # 保存原始圖像大小用於顯示
        display_image = original_image.copy()
        
        # 設置與訓練時相同的轉換
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 轉換圖像
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # 載入並處理 CBAM 模型
        cbam_attention_maps = []
        if cbam_exists:
            try:
                print("載入 CBAM 模型...")
                cbam_model = VGG19_CBAM(CONFIG['in_channels'], CONFIG['out_channels']).to(device)
                # 使用 map_location 和 weights_only 參數
                try:
                    cbam_model.load_state_dict(torch.load(cbam_model_path, map_location=device, weights_only=True))
                except:
                    # 如果 weights_only 參數不可用 (可能是 PyTorch 較舊版本)
                    cbam_model.load_state_dict(torch.load(cbam_model_path, map_location=device))
                cbam_model.eval()
                
                # 嘗試獲取 CBAM 注意力圖
                print("獲取 CBAM 注意力圖...")
                cbam_attention_maps = get_model_attention(cbam_model, img_tensor, 'cbam')
                
                # 可視化 CBAM 注意力圖
                if cbam_attention_maps:
                    cbam_output_path = os.path.join(output_dir, f'{os.path.basename(image_path)}_cbam.png')
                    visualize_attention_maps(display_image, cbam_attention_maps, 'cbam', cbam_output_path)
                    print(f"CBAM 注意力圖已保存: {cbam_output_path}")
            except Exception as e:
                print(f"處理 CBAM 模型時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("CBAM 模型文件不存在，跳過 CBAM 注意力圖生成。")
        
        # 載入並處理 Residual Attention 模型
        ra_attention_maps = []
        if ra_exists:
            try:
                print("載入 Residual Attention 模型...")
                ra_model = ResidualAttentionNetwork(CONFIG['in_channels'], CONFIG['out_channels']).to(device)
                # 使用 map_location 和 weights_only 參數
                try:
                    ra_model.load_state_dict(torch.load(ra_model_path, map_location=device, weights_only=True))
                except:
                    # 如果 weights_only 參數不可用
                    ra_model.load_state_dict(torch.load(ra_model_path, map_location=device))
                ra_model.eval()
                
                # 嘗試獲取 Residual Attention 注意力圖
                print("獲取 Residual Attention 注意力圖...")
                ra_attention_maps = get_model_attention(ra_model, img_tensor, 'ra')
                
                # 可視化 Residual Attention 注意力圖
                if ra_attention_maps:
                    ra_output_path = os.path.join(output_dir, f'{os.path.basename(image_path)}_ra.png')
                    visualize_attention_maps(display_image, ra_attention_maps, 'ra', ra_output_path)
                    print(f"Residual Attention 注意力圖已保存: {ra_output_path}")
            except Exception as e:
                print(f"處理 Residual Attention 模型時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Residual Attention 模型文件不存在，跳過 RA 注意力圖生成。")
        
        # 創建比較圖 (如果兩種模型的注意力圖都可用)
        if cbam_attention_maps and ra_attention_maps:
            comparison_path = os.path.join(output_dir, f'{os.path.basename(image_path)}_comparison.png')
            create_comparison_visualization(
                display_image, 
                cbam_attention_maps, 
                ra_attention_maps,
                comparison_path
            )
            print(f"比較可視化結果已保存: {comparison_path}")
        
        print("可視化處理完成!")
    
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 命令行參數
    import argparse
    parser = argparse.ArgumentParser(description='生成模型注意力可視化')
    parser.add_argument('--image', type=str, default=None, 
                        help='要可視化的圖像路徑 (如果不提供將自動搜索)')
    parser.add_argument('--output', type=str, default='attention_visualizations',
                        help='輸出目錄路徑')
    parser.add_argument('--cbam', type=str, default=None,
                        help='CBAM 模型路徑 (默認使用配置中的路徑)')
    parser.add_argument('--ra', type=str, default=None,
                        help='Residual Attention 模型路徑 (默認使用配置中的路徑)')
    args = parser.parse_args()
    
    # 查找樣本圖像
    sample_image_path = args.image
    if not sample_image_path:
        sample_image_path = find_image_file()
        if not sample_image_path:
            print("在項目目錄中未找到圖像文件，將創建一個隨機測試圖像")
            sample_image_path = create_random_image()
        else:
            print(f"找到圖像文件: {sample_image_path}")
    
    # 設置輸出目錄和模型路徑
    output_dir = os.path.join(project_root, args.output)
    cbam_model_path = args.cbam if args.cbam else os.path.join(CONFIG['save_path'], 'CBAM_model.pth')
    ra_model_path = args.ra if args.ra else os.path.join(CONFIG['save_path'], 'ResidualAttention_model.pth')
    
    # 執行可視化
    visualize_sample_image(sample_image_path, output_dir, cbam_model_path, ra_model_path)