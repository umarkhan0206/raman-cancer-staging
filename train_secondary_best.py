import os, numpy as np, torch, torch.nn as nn, random
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import json

DATA_ROOT   = '/users/vvgk0135/raman-project/data/extracted_cells_border75px'
RESULTS_DIR = '/users/vvgk0135/raman-project/results/2025_75px_final'
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

CLASS_MAP = {
    'FHC':0,'HCT116':1,'SW480':2,'CaCo2':2,'HT29':3,'SW620':3
}
CLASS_NAMES = ["Healthy\n(FHC)","Duke's A\n(HCT116)",
               "Duke's B\n(SW480+CaCo2)","Duke's C\n(HT29+SW620)"]

def acquisition_based_split(root_dir, class_map):
    area_groups = defaultdict(list)
    for cell_line, label in class_map.items():
        folder = os.path.join(root_dir, cell_line)
        if not os.path.exists(folder): continue
        for f in sorted(os.listdir(folder)):
            if not f.endswith('.npy'): continue
            path  = os.path.join(folder, f)
            fname = f.lower()
            if   'area 1' in fname or 'area1' in fname: area = 'area1'
            elif 'area 2' in fname or 'area2' in fname: area = 'area2'
            elif 'area 3' in fname or 'area3' in fname: area = 'area3'
            elif 'area 4' in fname or 'area4' in fname: area = 'area4'
            else:                                        area = 'area1'
            area_groups[(cell_line, area)].append((path, label))

    cell_line_areas = defaultdict(list)
    for (cell_line, area) in sorted(area_groups.keys()):
        cell_line_areas[cell_line].append(area)

    train, val, test = [], [], []
    for cell_line, areas in cell_line_areas.items():
        areas  = sorted(areas)
        groups = [area_groups[(cell_line, a)] for a in areas]
        if len(areas) == 1:
            train.extend(groups[0])
        elif len(areas) == 2:
            train.extend(groups[0])
            mid = len(groups[1]) // 2
            val.extend(groups[1][:mid])
            test.extend(groups[1][mid:])
        elif len(areas) >= 3:
            for g in groups[:-1]:
                train.extend(g)
            mid = len(groups[-1]) // 2
            val.extend(groups[-1][:mid])
            test.extend(groups[-1][mid:])
    return train, val, test

train_s, val_s, test_s = acquisition_based_split(DATA_ROOT, CLASS_MAP)
print(f'Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}')

# Check test distribution
test_labels_check = [l for _,l in test_s]
counts = np.bincount(test_labels_check, minlength=4)
print(f'Test distribution: FHC={counts[0]} HCT116={counts[1]} DukesB={counts[2]} DukesC={counts[3]}')

# Save split
split_info = {
    'train': [p for p,_ in train_s],
    'val':   [p for p,_ in val_s],
    'test':  [p for p,_ in test_s],
}
with open(os.path.join(RESULTS_DIR, 'split.json'), 'w') as f:
    json.dump(split_info, f, indent=2)

class RamanDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = [s[0] for s in samples]
        self.labels  = [s[1] for s in samples]
        self.augment = augment

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        cell = np.load(self.samples[idx]).astype(np.float32)
        # Per-channel min-max normalisation
        for ch in range(cell.shape[0]):
            mn, mx = cell[ch].min(), cell[ch].max()
            if mx > mn:
                cell[ch] = (cell[ch] - mn) / (mx - mn)

        if self.augment:
            if random.random() > 0.5: cell = cell[:, ::-1, :].copy()
            if random.random() > 0.5: cell = cell[:, :, ::-1].copy()
            if random.random() > 0.5:
                k = random.choice([1,2,3])
                cell = np.rot90(cell, k=k, axes=(1,2)).copy()
            if random.random() > 0.5:
                # Channel noise stds for [1450, 1660, 2850, 2935]
                noise_stds = [0.0562, 0.0458, 0.0486, 0.0783]
                noise = np.zeros_like(cell)
                for ch in range(4):
                    noise[ch] = np.random.normal(0, noise_stds[ch],
                                                  cell[ch].shape).astype(np.float32)
                cell = np.clip(cell + noise, 0, 1)

        return torch.tensor(cell), self.labels[idx]

class RamanResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        original_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7,
                                      stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.model.conv1.weight[:, :3] = original_conv.weight
            self.model.conv1.weight[:, 3:4] = original_conv.weight.mean(dim=1, keepdim=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Class weights
all_labels = [l for _,l in train_s] + [l for _,l in val_s]
counts_all = [all_labels.count(i) for i in range(4)]
total = sum(counts_all)
weights = torch.tensor([total/(4*c) if c>0 else 0 for c in counts_all],
                        dtype=torch.float32).to(DEVICE)
print(f'Class weights: {weights}')

train_loader = DataLoader(RamanDataset(train_s, augment=True),
                          batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(RamanDataset(val_s,   augment=False),
                          batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(RamanDataset(test_s,  augment=False),
                          batch_size=32, shuffle=False, num_workers=4)

# Train 3 runs, save best
best_overall_acc = 0.0
best_wts = None

for run in range(3):
    torch.manual_seed(42 + run)
    model = RamanResNet50(num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_acc = 0.0
    best_run_wts = None
    patience = 0
    PATIENCE = 10

    for epoch in range(100):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                val_preds.extend(model(images.to(DEVICE)).argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.numpy())
        val_acc = (np.array(val_preds)==np.array(val_labels)).mean()*100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_run_wts = {k:v.clone() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0:
            print(f'  Run {run+1} Epoch {epoch:3d} | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%')

        if patience >= PATIENCE:
            print(f'  Run {run+1} early stop at epoch {epoch}')
            break

    print(f'  Run {run+1}: {best_val_acc:.2f}%')

    if best_val_acc > best_overall_acc:
        best_overall_acc = best_val_acc
        best_wts = best_run_wts

print(f'\nBest val accuracy: {best_overall_acc:.2f}%')
torch.save(best_wts, os.path.join(RESULTS_DIR, 'best_model.pth'))

# Test
model = RamanResNet50(num_classes=4).to(DEVICE)
model.load_state_dict(best_wts)
model.eval()

test_preds, test_labels_list = [], []
with torch.no_grad():
    for images, labels in test_loader:
        test_preds.extend(model(images.to(DEVICE)).argmax(dim=1).cpu().numpy())
        test_labels_list.extend(labels.numpy())

test_preds  = np.array(test_preds)
test_labels = np.array(test_labels_list)
acc = (test_preds==test_labels).mean()*100
f1  = f1_score(test_labels, test_preds, average='macro', zero_division=0)
print(f'\nTest Accuracy: {acc:.2f}% | Macro F1: {f1:.4f}')
print(classification_report(test_labels, test_preds,
                             target_names=CLASS_NAMES, zero_division=0))

# Save results
results = {'accuracy':float(acc), 'macro_f1':float(f1),
           'best_val_acc':float(best_overall_acc),
           'train':len(train_s), 'val':len(val_s), 'test':len(test_s)}
with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.5)
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
axes[0].set_title('Raw Counts')

sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
axes[1].set_title('Normalised')

fig.suptitle(f'Secondary Dataset — ResNet50 pretrained, 75px border\n'
             f'Test Accuracy: {acc:.2f}%  |  Macro F1: {f1:.4f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cm_2025_75px.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
print('Confusion matrix saved.')
print('Done')
