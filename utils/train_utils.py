import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    """
    訓練模型並返回最佳模型
    
    參數:
        model: 要訓練的模型
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        criterion: 損失函數
        optimizer: 優化器
        scheduler: 學習率調度器
        num_epochs: 訓練輪數
        device: 訓練設備 ('cuda' or 'cpu')
    """
    best_acc = 0.0
    best_model_weights = model.state_dict()
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 訓練階段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # 迭代訓練數據
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度歸零
            optimizer.zero_grad()
            
            # 前向傳播
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # 反向傳播與優化
                loss.backward()
                optimizer.step()
            
            # 統計
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # 計算訓練集的損失和準確率
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 驗證階段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        # 迭代驗證數據
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向傳播
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # 統計
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # 計算驗證集的損失和準確率
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 如果是最佳模型，保存權重
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = model.state_dict()
            
        # 更新學習率
        scheduler.step()
        
        print()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    # 載入最佳模型權重
    model.load_state_dict(best_model_weights)
    return model, history

def evaluate_model(model, test_loader, device='cuda'):
    """
    評估模型性能
    
    參數:
        model: 要評估的模型
        test_loader: 測試數據加載器
        device: 使用的設備
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 計算各種指標
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # 繪製混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}