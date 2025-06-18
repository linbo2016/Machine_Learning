# 配置參數
CONFIG = {
    # 數據路徑 (需要使用絕對路徑)
    'data_path': r'C:\linbo\Structural_Machine_Learning_Models_and_Their_Applications\cmba\BreaKHis_v1\BreaKHis_v1\histology_slides',  # 改為您的BreakHis數據集絕對路徑
    
    # 訓練參數
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'lr_step_size': 10,
    'lr_gamma': 0.1,
    
    # 數據參數
    'magnification': 'all',  # '40X', '100X', '200X', '400X'
    'test_split': 0.2,      # 測試集比例 (20%)
    'seed': 42,             # 隨機種子 (用於確保可重現性)
    'num_workers': 4,
    
    # 模型參數
    'in_channels': 3,
    'out_channels': 2,  # 二分類: 良性/惡性
    
    # 儲存路徑 (最好也使用絕對路徑)
    'save_path': r'D:\cheng\CBAM vs. Residual Attention Network'
}