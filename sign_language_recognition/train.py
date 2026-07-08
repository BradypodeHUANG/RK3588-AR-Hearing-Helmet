import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Windows GBK console can't encode emoji printed by PyTorch's ONNX exporter logs.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import SignLanguageDataset
from model import SignLanguageModel


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
BATCH_SIZE = 16
EPOCHS = 100
WARMUP_EPOCHS = 5
PATIENCE = 20  # early stopping: val_acc 连续这么多 epoch 无提升则停止
LR = 0.01
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Speed: autotune conv algorithms (fixed input shape) + allow TF32 matmul on Ada
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, labels in loader:
        data = data.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, labels in loader:
        data = data.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        output = model(data)
        loss = criterion(output, labels)

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_and_report(model, loader, writer, class_names, num_classes):
    """在 val 集上输出每类准确率 + 混淆矩阵（打印 + 写入 TensorBoard，无额外依赖）。"""
    import numpy as np
    model.eval()
    preds_all, labels_all = [], []
    for data, labels in loader:
        data = data.to(DEVICE, non_blocking=True)
        out = model(data)
        preds_all.append(out.argmax(dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    # 每类准确率
    print('\n=== 每类准确率 (val) ===')
    md = ['| idx | class | correct/total | acc |', '|---|---|---|---|']
    for i, name in enumerate(class_names):
        n = int(cm[i].sum())
        acc = cm[i, i] / n if n > 0 else 0.0
        print(f'  [{i:2d}] {name:10s} {cm[i, i]:3d}/{n:<3d} = {acc:.3f}')
        writer.add_scalar(f'PerClassAcc/{name}', acc, 0)
        md.append(f'| {i} | {name} | {cm[i, i]}/{n} | {acc:.3f} |')
    writer.add_text('per_class_accuracy', '\n'.join(md), 0)

    # 错分明细
    print('=== 错分明细 (true -> pred) ===')
    errs = []
    for t in range(num_classes):
        for p in range(num_classes):
            if t != p and cm[t, p] > 0:
                line = f'{class_names[t]} -> {class_names[p]}: {cm[t, p]}'
                print('  ' + line)
                errs.append(line)
    if not errs:
        print('  （无错分）')

    # 文本混淆矩阵写入 TensorBoard（行=true，列=pred）
    header = 'T\\P ' + ''.join(f'{j:>4d}' for j in range(num_classes))
    rows = [header]
    for i in range(num_classes):
        rows.append(f'{i:>3d} ' + ''.join(f'{cm[i, j]:>4d}' for j in range(num_classes)))
    legend = '  '.join(f'{i}={n}' for i, n in enumerate(class_names))
    cm_text = '```\n' + '\n'.join(rows) + '\n\n' + legend + '\n```'
    writer.add_text('confusion_matrix', cm_text, 0)
    print('每类准确率/混淆矩阵已写入 TensorBoard '
          '(PerClassAcc/*, per_class_accuracy, confusion_matrix)')


def export_onnx(model, save_path='best_model.onnx'):
    model.eval()
    model.to('cpu')
    dummy_input = torch.randn(1, 6, 60, 26, 2)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,
    )
    print(f'ONNX model exported to {save_path}')


def main():
    train_dataset = SignLanguageDataset(
        DATA_DIR, split='train', train_ratio=TRAIN_RATIO, augment=True
    )
    val_dataset = SignLanguageDataset(
        DATA_DIR, split='val', train_ratio=TRAIN_RATIO, augment=False
    )

    num_classes = train_dataset.num_classes
    print(f'Classes ({num_classes}): {train_dataset.pinyin_labels}')
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # Data is cached in memory -> num_workers=0 avoids Windows spawn overhead.
    # Larger batch + pin_memory keeps the GPU fed.
    pin = (DEVICE.type == 'cuda')
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin
    )

    model = SignLanguageModel(num_classes=num_classes).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {param_count:,}')
    print(f'Device: {DEVICE}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # warmup（线性升温）后接 cosine decay
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_val_acc = 0.0
    best_model_path = 'best_model.pth'
    epochs_no_improve = 0

    log_dir = os.path.join('runs', time.strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir=log_dir)
    print(f'TensorBoard logs -> {log_dir}  (查看: tensorboard --logdir runs)')

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', lr, epoch)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f'Epoch {epoch:3d}/{EPOCHS} | '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
                  f'{elapsed/epoch*1000:.0f}ms/ep')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f'Early stopping at epoch {epoch} '
                      f'(val_acc 连续 {PATIENCE} epoch 无提升, best={best_val_acc:.4f})')
                break

    total_time = time.time() - t_start
    print(f'\nBest Val Accuracy: {best_val_acc:.4f}')
    print(f'Total training time: {total_time:.1f}s ({total_time/epoch*1000:.0f}ms/epoch)')

    writer.add_scalar('Best/val_acc', best_val_acc, EPOCHS)

    # 加载最佳权重，输出每类准确率 + 混淆矩阵
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    evaluate_and_report(model, val_loader, writer,
                        train_dataset.pinyin_labels, num_classes)
    writer.close()

    export_onnx(model)


if __name__ == '__main__':
    main()
