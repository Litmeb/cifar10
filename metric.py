import torch

def get_metrics(model, load_testset, beta=1, fault_tolerance=0):
    """
    计算模型的各项指标
    Args:
        model: 训练好的模型
        load_testset: 测试数据加载器
        beta: F1分数的beta参数，默认为1（即F1）
        fault_tolerance: 容错度，如果>0，则允许在top-k预测中
    Returns:
        f1: 各类别的F1分数 (tensor)
        accuracy: 总体准确率 (tensor)
        precision: 各类别的精确率 (tensor)
        recall: 各类别的召回率 (tensor)
    """
    device = next(model.parameters()).device
    acc = torch.zeros(10, device=device)
    predict = torch.zeros(10, device=device)
    total = torch.zeros(10, device=device)
    
    if fault_tolerance:
        for data in load_testset:
            imgs, label = data
            imgs = imgs.to(device)
            label = label.to(device)
            ans = model(imgs)
            for i in range(10):
                acc[i] += torch.sum(torch.sum(torch.topk(ans, 1+fault_tolerance, dim=1).indices==label.unsqueeze(1), dim=-1)*(label==i)).item()
                total[i] += torch.sum(label==i).item()
                predict[i] += torch.sum(torch.topk(ans, 1+fault_tolerance, dim=1).indices==i).item()
    else:
        for data in load_testset:
            imgs, label = data
            imgs = imgs.to(device)
            label = label.to(device)
            ans = model(imgs)
            for i in range(10):
                acc[i] += torch.sum((torch.argmax(ans, axis=1)==label)*(label==i)).item()
                total[i] += torch.sum(label==i).item()
                predict[i] += torch.sum(torch.argmax(ans, axis=1)==i).item()
    
    precision = acc / predict
    recall = acc / total
    f1 = precision * recall / (beta**0.5 * precision + recall) * (1 + beta**0.5)
    accuracy = torch.sum(acc) / torch.sum(total)
    
    return f1, accuracy, precision, recall

