import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_training_history(history, title=None):
    """
    繪製訓練歷史
    
    參數:
        history: 包含訓練和驗證損失/準確率的字典
        title: 圖表標題
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 繪製損失曲線
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # 繪製準確率曲線
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()

def compare_models_performance(models_metrics, model_names):
    """
    比較多個模型的性能
    
    參數:
        models_metrics: 模型度量字典的列表
        model_names: 模型名稱列表
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # 準備數據
    values = []
    for metric in metrics:
        values.append([m[metric] for m in models_metrics])
    
    # 繪製條形圖
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.2
    
    # 繪製每個度量的條形
    for i, (metric, vals) in enumerate(zip(metrics, values)):
        ax.bar(x + i*width, vals, width, label=metric.capitalize())
    
    # 設置圖表
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    plt.tight_layout()
    plt.show()