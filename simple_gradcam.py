import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
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

# Grad-CAM 實現
class GradCAM:
    """使用 Grad-CAM 技術生成注意力圖"""
    def __init__(self, model, target_layer):
        """
        初始化 Grad-CAM
        
        參數:
            model: 要分析的模型
            target_layer: 目標層，用於提取注意力圖
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # 註冊鉤子
        self._register_hooks()
        
        # 設置模型為評估模式
        self.model.eval()
        
    def _register_hooks(self):
        """註冊前向和後向鉤子"""
        # 前向鉤子
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # 後向鉤子
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 註冊鉤子
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        
        # 保存鉤子以便稍後移除
        self.hooks.append(forward_handle)
        self.hooks.append(backward_handle)
    
    def remove_hooks(self):
        """移除所有鉤子"""
        for hook in self.hooks:
            hook.remove()
    
    def __call__(self, input_image, class_idx=None):
        """
        生成 Grad-CAM 注意力圖
        
        參數:
            input_image: 輸入圖像張量 (1, C, H, W)
            class_idx: 目標類別索引，如果為 None，則使用最大可能性的類別
        
        返回:
            cam: 注意力圖
        """
        # 確保梯度計算
        input_image.requires_grad = True
        
        # 前向傳播
        output = self.model(input_image)
        
        # 如果未指定類別，使用最大可能性的類別
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # 清零梯度
        self.model.zero_grad()
        
        # 反向傳播
        output[0, class_idx].backward(retain_graph=True)
        
        # 計算 Grad-CAM
        with torch.no_grad():
            # 獲取梯度的全局平均池化
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            
            # 加權和
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            
            # ReLU
            cam = torch.relu(cam)
            
        # 轉換為 numpy
        cam = cam.squeeze().cpu().numpy()
        
        # 歸一化
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / cam.max()
        
        return cam

def find_target_layer(model, model_type):
    """
    查找適合 Grad-CAM 的目標層
    
    參數:
        model: 模型
        model_type: 'cbam' 或 'ra'
    
    返回:
        目標層
    """
    if model_type == 'cbam':
        # 對於 CBAM 模型，使用最後一個卷積層
        target_layer = model.conv_block5[3]  # 假設這是最後一個卷積層
    else:
        # 對於 Residual Attention 模型，使用最後一個注意力模塊
        target_layer = model.stage4  # 假設這是最後一個注意力模塊
    
    return target_layer

def preprocess_image(image_path, transform=None):
    """
    預處理圖像
    
    參數:
        image_path: 圖像路徑
        transform: 圖像轉換
    
    返回:
        processed_image: 處理後的圖像張量
        original_image: 原始圖像
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 讀取圖像
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # 應用轉換
    processed_image = transform(image).unsqueeze(0)
    
    return processed_image, original_image

def generate_cam_visualization(image, cam, output_path=None):
    """
    生成 CAM 可視化
    
    參數:
        image: 原始圖像
        cam: CAM 注意力圖
        output_path: 輸出路徑
    
    返回:
        visualization: 可視化結果
    """
    # 調整 CAM 大小以匹配原始圖像
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # 創建熱力圖
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 疊加原始圖像和熱力圖
    image_float = image.astype(np.float32) / 255.0
    heatmap_float = heatmap.astype(np.float32) / 255.0
    visualization = 0.6 * image_float + 0.4 * heatmap_float
    visualization = np.clip(visualization, 0, 1)
    visualization = (visualization * 255).astype(np.uint8)
    
    # 如果提供了輸出路徑，保存可視化結果
    if output_path:
        plt.figure(figsize=(12, 4))
        
        # 顯示原始圖像
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')
        
        # 顯示 CAM
        plt.subplot(1, 3, 2)
        plt.title('Grad-CAM')
        plt.imshow(cam_resized, cmap='jet')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        # 顯示疊加圖
        plt.subplot(1, 3, 3)
        plt.title('Overlay')
        plt.imshow(visualization)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    return visualization

def compare_models_with_gradcam(image_path, cbam_model_path, ra_model_path, output_dir):
    """
    使用 Grad-CAM 比較 CBAM 和 Residual Attention 模型
    
    參數:
        image_path: 圖像路徑
        cbam_model_path: CBAM 模型路徑
        ra_model_path: Residual Attention 模型路徑
        output_dir: 輸出目錄
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 檢查模型文件是否存在
    cbam_exists = os.path.exists(cbam_model_path)
    ra_exists = os.path.exists(ra_model_path)
    
    print(f"CBAM 模型路徑: {cbam_model_path} (存在: {cbam_exists})")
    print(f"RA 模型路徑: {ra_model_path} (存在: {ra_exists})")
    
    # 檢查圖像文件是否存在
    if not os.path.exists(image_path):
        print(f"錯誤: 圖像文件不存在: {image_path}")
        return
    
    # 預處理圖像
    processed_image, original_image = preprocess_image(image_path)
    processed_image = processed_image.to(device)
    
    # 保存原始圖像作為參考
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "original_image.png"))
    plt.close()
    
    # 處理 CBAM 模型
    if cbam_exists:
        try:
            print("載入 CBAM 模型...")
            cbam_model = VGG19_CBAM(CONFIG['in_channels'], CONFIG['out_channels']).to(device)
            # 載入模型權重
            try:
                cbam_model.load_state_dict(torch.load(cbam_model_path, map_location=device, weights_only=True))
            except:
                cbam_model.load_state_dict(torch.load(cbam_model_path, map_location=device))
            
            # 找到目標層
            target_layer = find_target_layer(cbam_model, 'cbam')
            
            # 初始化 Grad-CAM
            grad_cam = GradCAM(cbam_model, target_layer)
            
            # 生成 CAM
            print("生成 CBAM 的 Grad-CAM...")
            cam = grad_cam(processed_image)
            
            # 生成可視化
            cbam_output_path = os.path.join(output_dir, f'{os.path.basename(image_path)}_cbam_gradcam.png')
            generate_cam_visualization(original_image, cam, cbam_output_path)
            print(f"CBAM Grad-CAM 已保存: {cbam_output_path}")
            
            # 清理
            grad_cam.remove_hooks()
            
        except Exception as e:
            print(f"處理 CBAM 模型時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 處理 Residual Attention 模型
    if ra_exists:
        try:
            print("載入 Residual Attention 模型...")
            ra_model = ResidualAttentionNetwork(CONFIG['in_channels'], CONFIG['out_channels']).to(device)
            # 載入模型權重
            try:
                ra_model.load_state_dict(torch.load(ra_model_path, map_location=device, weights_only=True))
            except:
                ra_model.load_state_dict(torch.load(ra_model_path, map_location=device))
            
            # 找到目標層
            target_layer = find_target_layer(ra_model, 'ra')
            
            # 初始化 Grad-CAM
            grad_cam = GradCAM(ra_model, target_layer)
            
            # 生成 CAM
            print("生成 Residual Attention 的 Grad-CAM...")
            cam = grad_cam(processed_image)
            
            # 生成可視化
            ra_output_path = os.path.join(output_dir, f'{os.path.basename(image_path)}_ra_gradcam.png')
            generate_cam_visualization(original_image, cam, ra_output_path)
            print(f"Residual Attention Grad-CAM 已保存: {ra_output_path}")
            
            # 清理
            grad_cam.remove_hooks()
            
        except Exception as e:
            print(f"處理 Residual Attention 模型時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 生成比較可視化
    if cbam_exists and ra_exists:
        try:
            print("生成比較可視化...")
            
            # 重新生成 CAM
            cbam_target_layer = find_target_layer(cbam_model, 'cbam')
            cbam_grad_cam = GradCAM(cbam_model, cbam_target_layer)
            cbam_cam = cbam_grad_cam(processed_image)
            
            ra_target_layer = find_target_layer(ra_model, 'ra')
            ra_grad_cam = GradCAM(ra_model, ra_target_layer)
            ra_cam = ra_grad_cam(processed_image)
            
            # 調整 CAM 大小
            cbam_cam_resized = cv2.resize(cbam_cam, (original_image.shape[1], original_image.shape[0]))
            ra_cam_resized = cv2.resize(ra_cam, (original_image.shape[1], original_image.shape[0]))
            
            # 生成熱力圖
            cbam_heatmap = cv2.applyColorMap(np.uint8(255 * cbam_cam_resized), cv2.COLORMAP_JET)
            cbam_heatmap = cv2.cvtColor(cbam_heatmap, cv2.COLOR_BGR2RGB)
            
            ra_heatmap = cv2.applyColorMap(np.uint8(255 * ra_cam_resized), cv2.COLORMAP_JET)
            ra_heatmap = cv2.cvtColor(ra_heatmap, cv2.COLOR_BGR2RGB)
            
            # 疊加
            image_float = original_image.astype(np.float32) / 255.0
            cbam_heatmap_float = cbam_heatmap.astype(np.float32) / 255.0
            ra_heatmap_float = ra_heatmap.astype(np.float32) / 255.0
            
            cbam_overlay = 0.6 * image_float + 0.4 * cbam_heatmap_float
            cbam_overlay = np.clip(cbam_overlay, 0, 1)
            cbam_overlay = (cbam_overlay * 255).astype(np.uint8)
            
            ra_overlay = 0.6 * image_float + 0.4 * ra_heatmap_float
            ra_overlay = np.clip(ra_overlay, 0, 1)
            ra_overlay = (ra_overlay * 255).astype(np.uint8)
            
            # 創建比較圖
            plt.figure(figsize=(15, 8))
            
            # 原始圖像
            plt.subplot(2, 3, 2)
            plt.title('Original Image', fontsize=14)
            plt.imshow(original_image)
            plt.axis('off')
            
            # CBAM Grad-CAM
            plt.subplot(2, 3, 1)
            plt.title('CBAM Grad-CAM', fontsize=14)
            plt.imshow(cbam_cam_resized, cmap='jet')
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            # CBAM 疊加
            plt.subplot(2, 3, 3)
            plt.title('CBAM Overlay', fontsize=14)
            plt.imshow(cbam_overlay)
            plt.axis('off')
            
            # RA Grad-CAM
            plt.subplot(2, 3, 4)
            plt.title('RA Grad-CAM', fontsize=14)
            plt.imshow(ra_cam_resized, cmap='jet')
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            # RA 疊加
            plt.subplot(2, 3, 6)
            plt.title('RA Overlay', fontsize=14)
            plt.imshow(ra_overlay)
            plt.axis('off')
            
            # 差異圖 (RA - CBAM)
            plt.subplot(2, 3, 5)
            plt.title('Difference (RA - CBAM)', fontsize=14)
            difference = ra_cam_resized - cbam_cam_resized
            plt.imshow(difference, cmap='seismic', vmin=-1, vmax=1)
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, f'{os.path.basename(image_path)}_gradcam_comparison.png')
            plt.savefig(comparison_path, dpi=300)
            plt.close()
            
            print(f"比較可視化已保存: {comparison_path}")
            
            # 清理
            cbam_grad_cam.remove_hooks()
            ra_grad_cam.remove_hooks()
            
        except Exception as e:
            print(f"生成比較可視化時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("Grad-CAM 分析完成!")

def find_images():
    """尋找項目中所有的圖像文件"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    image_files = []
    
    # 首先檢查是否有 sample_images 目錄
    sample_dir = os.path.join(project_root, "sample_images")
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(sample_dir, file))
    
    # 然後檢查 data 目錄
    data_dir = os.path.join(project_root, "data")
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
    
    # 最後檢查整個項目目錄
    for root, dirs, files in os.walk(project_root):
        # 跳過已經檢查過的目錄
        if root.startswith(sample_dir) or root.startswith(data_dir):
            continue
        
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def select_image():
    """讓用戶選擇圖像文件"""
    # 尋找所有圖像文件
    image_files = find_images()
    
    if not image_files:
        print("未找到任何圖像文件，將創建一個隨機測試圖像")
        return create_random_image()
    
    # 顯示找到的圖像文件供用戶選擇
    print("\n找到以下圖像文件:")
    for i, image_file in enumerate(image_files):
        print(f"{i+1}. {image_file}")
    
    # 讓用戶選擇
    while True:
        try:
            choice = input("\n請選擇一個圖像文件 (輸入對應的數字)，或直接輸入圖像路徑: ")
            
            # 檢查是否是數字
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(image_files):
                    return image_files[idx]
                else:
                    print(f"無效的選擇，請輸入 1 到 {len(image_files)} 之間的數字")
            # 檢查是否是文件路徑
            elif os.path.isfile(choice):
                return choice
            else:
                print("無效的路徑，請重新輸入")
        except KeyboardInterrupt:
            print("\n取消選擇，將使用第一個找到的圖像文件")
            return image_files[0]
        except Exception as e:
            print(f"發生錯誤: {e}")

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

if __name__ == "__main__":
    # 命令行參數
    import argparse
    parser = argparse.ArgumentParser(description='使用 Grad-CAM 比較 CBAM 和 Residual Attention 模型')
    parser.add_argument('--image', type=str, default=None, 
                        help='要分析的圖像路徑 (如果不提供將提示用戶選擇)')
    parser.add_argument('--output', type=str, default='gradcam_visualizations',
                        help='輸出目錄路徑')
    parser.add_argument('--cbam', type=str, default=None,
                        help='CBAM 模型路徑 (默認使用配置中的路徑)')
    parser.add_argument('--ra', type=str, default=None,
                        help='Residual Attention 模型路徑 (默認使用配置中的路徑)')
    parser.add_argument('--interactive', action='store_true',
                        help='啟用交互式模式，讓用戶選擇圖像')
    args = parser.parse_args()
    
    # 獲取圖像路徑
    sample_image_path = args.image
    if not sample_image_path or args.interactive:
        sample_image_path = select_image()
    
    if not sample_image_path:
        print("未能獲取有效的圖像路徑，程序退出")
        sys.exit(1)
    
    print(f"使用圖像: {sample_image_path}")
    
    # 設置輸出目錄和模型路徑
    output_dir = os.path.join(project_root, args.output)
    cbam_model_path = args.cbam if args.cbam else os.path.join(CONFIG['save_path'], 'CBAM_model.pth')
    ra_model_path = args.ra if args.ra else os.path.join(CONFIG['save_path'], 'ResidualAttention_model.pth')
    
    # 執行 Grad-CAM 分析
    compare_models_with_gradcam(sample_image_path, cbam_model_path, ra_model_path, output_dir)