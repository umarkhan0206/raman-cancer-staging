
import os, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix, classification_report

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

class_map = {"FHC":0,"HCT116":1,"SW480":2,"CaCo2":2,"HT29":3,"SW620":3}
class_names = ["Healthy (FHC)","Dukes A (HCT116)",
               "Dukes B (SW480+CaCo2)","Dukes C (HT29+SW620)"]

class RamanCellDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = [s[0] for s in samples]
        self.labels  = [s[1] for s in samples]
        self.augment = augment
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        cell = np.load(self.samples[idx]).astype(np.float32)
        gmin, gmax = cell.min(), cell.max()
        if gmax > gmin:
            cell = (cell - gmin) / (gmax - gmin)
        if self.augment:
            if random.random() > 0.5: cell = cell[:,::-1,:].copy()
            if random.random() > 0.5: cell = cell[:,:,::-1].copy()
            if random.random() > 0.5:
                k = random.choice([1,2,3])
                cell = np.rot90(cell, k=k, axes=(1,2)).copy()
            if random.random() > 0.5:
                noise_stds = [0.0562,0.0458,0.0486,0.0783]
                noise = np.zeros_like(cell)
                for ch in range(4):
                    noise[ch] = np.random.normal(
                        0, noise_stds[ch], cell[ch].shape).astype(np.float32)
                cell = np.clip(cell+noise, 0, 1)
        return torch.tensor(cell), self.labels[idx]

class BalancedSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, num_classes=4):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, features, labels):
        device = features.device
        batch  = features.shape[0]
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        mask_self = torch.eye(batch, dtype=torch.bool, device=device)
        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        mask_pos   = (labels_col == labels_row) & ~mask_self
        class_counts = torch.zeros(self.num_classes, device=device)
        for c in range(self.num_classes):
            class_counts[c] = (labels == c).sum().float()
        class_counts = class_counts.clamp(min=1)
        sample_weights = 1.0 / class_counts[labels]
        sample_weights = sample_weights / sample_weights.sum() * batch
        sim_masked = sim.masked_fill(mask_self, -1e9)
        log_prob   = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)
        num_pos    = mask_pos.sum(1).clamp(min=1)
        loss_per   = -(log_prob * mask_pos).sum(1) / num_pos
        loss = (loss_per * sample_weights).sum() / sample_weights.sum()
        return loss

class RamanResNet50_SCL(nn.Module):
    def __init__(self, num_classes=4, proj_dim=128):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        original_conv = base.conv1
        base.conv1 = nn.Conv2d(4,64,kernel_size=7,stride=2,padding=3,bias=False)
        with torch.no_grad():
            base.conv1.weight[:,:3] = original_conv.weight
            base.conv1.weight[:,3:4] = original_conv.weight.mean(dim=1,keepdim=True)
        self.backbone   = nn.Sequential(*list(base.children())[:-1])
        feat_dim        = base.fc.in_features
        self.projector  = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat  = self.backbone(x).flatten(1)
        proj  = self.projector(feat)
        logit = self.classifier(feat)
        return logit, proj

# Get 2026 75px samples
root_dir = "/users/vvgk0135/raman-project/data/extracted_2026_25px"
all_samples = []
for cell_line, lbl in class_map.items():
    folder = os.path.join(root_dir, cell_line)
    if not os.path.exists(folder): continue
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".npy"): continue
        all_samples.append((os.path.join(folder, f), lbl))

print(f"Total 2026 samples: {len(all_samples)}", flush=True)

sample_groups = defaultdict(list)
for path, lbl in all_samples:
    fname = os.path.basename(path).lower()
    parts = fname.split("_cell")[0].split("_")
    cl     = parts[0]
    sample = next((p for p in parts if p.startswith("s")
                  and p[1:].isdigit()), "s1")
    sample_groups[(cl, sample)].append((path, lbl))

cell_line_samples = defaultdict(list)
for (cl, s) in sorted(sample_groups.keys()):
    cell_line_samples[cl].append(s)

train_s, val_s, test_s = [], [], []
for cl, samples in cell_line_samples.items():
    samples = sorted(samples)
    n = len(samples)
    if n == 1:
        train_s.extend(sample_groups[(cl, samples[0])])
    elif n == 2:
        train_s.extend(sample_groups[(cl, samples[0])])
        test_s.extend(sample_groups[(cl, samples[1])])
    elif n == 3:
        train_s.extend(sample_groups[(cl, samples[0])])
        val_s.extend(sample_groups[(cl, samples[1])])
        test_s.extend(sample_groups[(cl, samples[2])])
    else:
        for s in samples[:-2]: train_s.extend(sample_groups[(cl, s)])
        val_s.extend(sample_groups[(cl, samples[-2])])
        test_s.extend(sample_groups[(cl, samples[-1])])

print(f"Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}", flush=True)
print("Test class distribution:", flush=True)
for i, name in enumerate(class_names):
    print(f"  {name}: {len([s for s in test_s if s[1]==i])}", flush=True)

train_ds = RamanCellDataset(train_s, augment=True)
val_ds   = RamanCellDataset(val_s,   augment=False)
test_ds  = RamanCellDataset(test_s,  augment=False)

counts  = [len([s for s in all_samples if s[1]==i]) for i in range(4)]
total   = sum(counts)
ce_weights = torch.tensor([total/(4*c) if c>0 else 0 for c in counts],
                           dtype=torch.float32).to(device)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=1)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=1)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=1)

scl_criterion = BalancedSupConLoss(temperature=0.07, num_classes=4)
ce_criterion  = nn.CrossEntropyLoss(weight=ce_weights)
alpha_scl     = 0.5   # weight for contrastive loss
alpha_ce      = 0.5   # weight for cross-entropy loss

val_accs = []
best_overall_acc = 0
best_wts = None

for run in range(3):
    print(f"\n--- Run {run+1}/3 ---", flush=True)
    model     = RamanResNet50_SCL(num_classes=4, proj_dim=128).to(device)
    optimizer = Adam(model.parameters(), lr=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    best_val_acc   = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience       = 0

    for epoch in range(100):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) \
                     else torch.tensor(labels).to(device)
            optimizer.zero_grad()
            logits, proj = model(inputs)
            loss_ce  = ce_criterion(logits, labels)
            loss_scl = scl_criterion(proj, labels)
            loss     = alpha_ce * loss_ce + alpha_scl * loss_scl
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device) if isinstance(labels, torch.Tensor) \
                         else torch.tensor(labels).to(device)
                logits, _ = model(inputs)
                correct  += logits.argmax(1).eq(labels).sum().item()
        val_acc = 100. * correct / len(val_ds)
        print(f"  Epoch {epoch+1}: val={val_acc:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience       = 0
        else:
            patience += 1
        if patience >= 15:
            print(f"  Early stop at epoch {epoch+1}", flush=True)
            break
        scheduler.step()

    val_accs.append(best_val_acc)
    if best_val_acc > best_overall_acc:
        best_overall_acc = best_val_acc
        best_wts = copy.deepcopy(best_model_wts)
    print(f"  Run {run+1} best val: {best_val_acc:.2f}%", flush=True)

torch.save(best_wts,
           "/users/vvgk0135/raman-project/results/best_model_scl.pth")

model.load_state_dict(best_wts)
model.eval()
all_preds, all_labels_list = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs   = inputs.to(device)
        labels_t = labels.to(device) if isinstance(labels, torch.Tensor) \
                   else torch.tensor(labels).to(device)
        logits, _ = model(inputs)
        preds     = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels_list.extend(labels_t.cpu().numpy())

print("\nClassification Report:", flush=True)
print(classification_report(all_labels_list, all_preds,
                             target_names=class_names, zero_division=0))
print("\nConfusion Matrix:", flush=True)
print(confusion_matrix(all_labels_list, all_preds))

test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels_list))
macro_f1 = f1_score(all_labels_list, all_preds, average="macro")

print("\n" + "="*60, flush=True)
print("FINAL RESULT -- Balanced SCL (2026 data)", flush=True)
print("="*60, flush=True)
print(f"Val:      {np.mean(val_accs):.2f}% +/- {np.std(val_accs):.2f}%", flush=True)
print(f"Test:     {test_acc:.2f}%", flush=True)
print(f"Macro F1: {macro_f1:.3f}", flush=True)
