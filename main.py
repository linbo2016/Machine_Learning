import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

from model.base_models import VGG19_CBAM, ResidualAttentionNetwork
from data.dataset import get_data_loaders, get_class_distribution
from utils.train_utils import train_model, evaluate_model
from utils.visualization import plot_training_history, compare_models_performance
from configs.config import CONFIG
from utils.visualization_utils import process_test_images

def set_seed(seed):
    """設置隨機種子以確保可重複性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def analyze_dataset(data_path, magnification):
    """分析數據集並顯示分佈情況"""
    dist = get_class_distribution(data_path, magnification)
    
    print("\n數據集分佈情況:")
    print(f"總圖像數: {dist['total']}")
    print(f"良性圖像: {dist['benign_total']} ({dist['benign_total']/dist['total']:.2%})")
    print(f"惡性圖像: {dist['malignant_total']} ({dist['malignant_total']/dist['total']:.2%})")
    
    # 顯示子類別分佈
    benign_types = [k for k in dist.keys() if k.startswith("benign/")]
    malignant_types = [k for k in dist.keys() if k.startswith("malignant/")]
    
    print("\n良性子類別分佈:")
    for b_type in benign_types:
        print(f"  {b_type.split('/')[1]}: {dist[b_type]} ({dist[b_type]/dist['benign_total']:.2%})")
    
    print("\n惡性子類別分佈:")
    for m_type in malignant_types:
        print(f"  {m_type.split('/')[1]}: {dist[m_type]} ({dist[m_type]/dist['malignant_total']:.2%})")
    
    # 繪製圖表
    plt.figure(figsize=(14, 6))
    
    # 繪製良性子類別
    plt.subplot(1, 2, 1)
    b_labels = [k.split('/')[1] for k in benign_types]
    b_values = [dist[k] for k in benign_types]
    plt.pie(b_values, labels=b_labels, autopct='%1.1f%%')
    plt.title('良性子類別分佈')
    
    # 繪製惡性子類別
    plt.subplot(1, 2, 2)
    m_labels = [k.split('/')[1] for k in malignant_types]
    m_values = [dist[k] for k in malignant_types]
    plt.pie(m_values, labels=m_labels, autopct='%1.1f%%')
    plt.title('惡性子類別分佈')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['save_path'], 'dataset_distribution.png'))
    plt.show()

def main(args):
    # 設置隨機種子
    set_seed(CONFIG['seed'])
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 分析數據集
    if args.analyze:
        analyze_dataset(CONFIG['data_path'], CONFIG['magnification'])
    
    # 獲取數據加載器
    train_loader, test_loader = get_data_loaders(
        data_path=CONFIG['data_path'],
        batch_size=CONFIG['batch_size'],
        magnification=CONFIG['magnification'],
        test_split=CONFIG['test_split'],
        seed=CONFIG['seed'],
        num_workers=CONFIG['num_workers']
    )
    
    # 初始化模型
    models = {}
    
    if args.model == 'all' or args.model == 'cbam':
        models['CBAM'] = VGG19_CBAM(
            in_channels=CONFIG['in_channels'], 
            out_channels=CONFIG['out_channels']
        ).to(device)
    
    if args.model == 'all' or args.model == 'ra':
        models['ResidualAttention'] = ResidualAttentionNetwork(
            in_channels=CONFIG['in_channels'], 
            out_channels=CONFIG['out_channels']
        ).to(device)
    
    # 設置損失函數
    weights = torch.tensor([1.0, 0.9], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # 訓練和評估每個模型
    histories = {}
    metrics = {}
    
    # 創建保存目錄
    if not os.path.exists(CONFIG['save_path']):
        os.makedirs(CONFIG['save_path'])
    
    for name, model in models.items():
        print(f"\n{'='*20} Training {name} {'='*20}")
        
        # 設置優化器和調度器
        optimizer = optim.Adam(
            model.parameters(), 
            lr=CONFIG['learning_rate'], 
            weight_decay=CONFIG['weight_decay']
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=CONFIG['lr_step_size'], 
            gamma=CONFIG['lr_gamma']
        )
        
        # 訓練模型
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=CONFIG['num_epochs'],
            device=device
        )
        
        # 記錄訓練歷史
        histories[name] = history
        
        # 評估模型
        print(f"\n{'='*20} Evaluating {name} {'='*20}")
        metrics[name] = evaluate_model(trained_model, test_loader, device)
        
        # 保存模型
        torch.save(
            trained_model.state_dict(),
            os.path.join(CONFIG['save_path'], f"{name}_model.pth")
        )
    
    # 繪製訓練歷史
    for name, history in histories.items():
        plot_training_history(history, title=f"{name} Training History")
        plt.savefig(os.path.join(CONFIG['save_path'], f"{name}_training_history.png"))
    
    # 比較模型性能
    if len(models) > 1:
        compare_models_performance(
            [metrics[name] for name in models.keys()],
            list(models.keys())
        )
        plt.savefig(os.path.join(CONFIG['save_path'], "models_comparison.png"))

    if args.visualize_attention:
        print("\n===== 可視化模型的注意力機制 =====")
        
        # 載入訓練好的模型
        cbam_model = VGG19_CBAM(CONFIG['in_channels'], CONFIG['out_channels']).to(device)
        cbam_model.load_state_dict(torch.load(os.path.join(CONFIG['save_path'], 'CBAM_model.pth')))
        
        ra_model = ResidualAttentionNetwork(CONFIG['in_channels'], CONFIG['out_channels']).to(device)
        ra_model.load_state_dict(torch.load(os.path.join(CONFIG['save_path'], 'ResidualAttention_model.pth')))
        
        # 設置可視化輸出目錄
        vis_output_dir = os.path.join(CONFIG['save_path'], 'attention_visualizations')
        os.makedirs(vis_output_dir, exist_ok=True)
        
        # 處理測試集中的圖像
        process_test_images(
            cbam_model=cbam_model,
            ra_model=ra_model,
            test_loader=test_loader,
            output_dir=vis_output_dir,
            num_samples=args.num_samples
        )
        
        print(f"注意力可視化結果已保存至：{vis_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate breast cancer classification models')
    parser.add_argument('--model', type=str, default='all', choices=['all', 'cbam', 'ra'],
                        help='Model to train: "cbam" (CBAM), "ra" (Residual Attention), or "all"')
    parser.add_argument('--analyze', action='store_true',
                        help='只分析數據集而不訓練模型')
    parser.add_argument('--visualize-attention', action='store_true',
                        help='可視化模型的注意力機制')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='要可視化的樣本數量')

    args = parser.parse_args()
    
    main(args)