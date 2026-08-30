import os, numpy as np, torch, torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import warnings; warnings.filterwarnings('ignore')

SCRATCH     = '/mnt/scratch/vvgk0135'
DATA_ROOT   = f'{SCRATCH}/extracted_2026_hs'
RESULTS_DIR = '/users/vvgk0135/raman-project/results/2026_12ch_pretrained_random'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED=42; NUM_EPOCHS=200; BATCH_SIZE=32; LR=5e-4; PATIENCE=50; NUM_RUNS=3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED)
print(f'Device: {DEVICE}')
print(f'RANDOM SPLIT BASELINE — 70/15/15 stratified by class')

CH_12 = [1, 2, 3, 13, 19, 23, 27, 30, 31, 35, 43, 47]
N_CH = len(CH_12)

CLASS_MAP  = {'FHC':0,'HCT116':1,'CaCo2':2,'SW480':2,'HT29':3,'SW620':3}
CLASS_NAMES = ['Healthy (FHC)',"Duke's A (HCT116)","Duke's B (SW480+CaCo2)","Duke's C (HT29+SW620)"]

TEST_SESSIONS = {
    'FHC':'s3','HCT116':'s3','CaCo2':'s3',
    'SW480':'s2','HT29':'s3','SW620':'s3',
}

SESSION_MEANS = {
    ('FHC','s1'):1221.8,('FHC','s2'):1783.6,('FHC','s3'):1818.8,
    ('HCT116','s1'):5574.8,('HCT116','s2'):6812.1,('HCT116','s3'):6019.0,
    ('CaCo2','s1'):6535.9,('CaCo2','s2'):6341.4,('CaCo2','s3'):5027.4,
    ('SW480','s1'):5885.8,('SW480','s2'):6387.0,('SW480','s3'):6153.6,
    ('HT29','s1'):7999.5,('HT29','s2'):9185.9,('HT29','s3'):6761.0,
    ('SW620','s1'):5112.8,('SW620','s2'):6179.5,('SW620','s3'):4594.2,
}
GLOBAL_REF = np.mean(list(SESSION_MEANS.values()))
SESSION_SCALE = {k: GLOBAL_REF/v for k,v in SESSION_MEANS.items()}

print("Computing protein max from all sessions (random split — no test holdout)...")
protein_max = 0.0
for cl in CLASS_MAP:
    folder = os.path.join(DATA_ROOT, cl)
    if not os.path.exists(folder): continue
    for fname in os.listdir(folder):
        if not fname.endswith('.npy'): continue
        cell = np.load(os.path.join(folder, fname)).astype(np.float32)
        val = float(cell[30].max())
        if val > protein_max: protein_max = val
print(f"Protein max: {protein_max:.2f}")

def get_random_split_files():
    """Random 70/15/15 split stratified by class."""
    all_files = {0:[], 1:[], 2:[], 3:[]}
    for cell_line, label in CLASS_MAP.items():
        folder = os.path.join(DATA_ROOT, cell_line)
        if not os.path.exists(folder): continue
        files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
        for f in files:
            parts = f.split('_')
            s = next((p for p in parts if p.startswith('s') and p[1:].isdigit()), 's1')
            all_files[label].append((os.path.join(folder, f), label, cell_line, s))

    train_files, val_files, test_files = [], [], []
    for label, files in all_files.items():
        np.random.shuffle(files)
        n = len(files)
        n_test = max(1, int(n * 0.15))
        n_val = max(1, int(n * 0.15))
        test_files.extend(files[:n_test])
        val_files.extend(files[n_test:n_test+n_val])
        train_files.extend(files[n_test+n_val:])
    return train_files, val_files, test_files

np.random.seed(SEED)
train_files, val_files, test_files = get_random_split_files()
print(f'Split: train={len(train_files)} val={len(val_files)} test={len(test_files)}')

def augment_cell(cell):
    cell = cell.copy()
    for c in range(cell.shape[0]):
        cell[c] = cell[c] * np.random.uniform(0.7,1.3) + np.random.uniform(-0.03,0.03)
    cell = np.clip(cell, 0, 1)
    if np.random.random()>0.5: cell=cell[:,:,::-1].copy()
    if np.random.random()>0.5: cell=cell[:,::-1,:].copy()
    k=np.random.randint(0,4)
    if k>0: cell=np.rot90(cell,k,axes=(1,2)).copy()
    return np.clip(cell+np.random.normal(0,0.01,cell.shape).astype(np.float32),0,1)

class RamanDataset(Dataset):
    def __init__(self,files,protein_max,session_scale,augment=False):
        self.files=files; self.protein_max=protein_max
        self.session_scale=session_scale; self.augment=augment
    def __len__(self): return len(self.files)
    def __getitem__(self,idx):
        path,label,cl,s=self.files[idx]
        cell=np.load(path).astype(np.float32)[CH_12]
        scale=self.session_scale.get((cl,s),1.0)
        cell=np.clip(cell*scale/self.protein_max,0,1)
        if self.augment: cell=augment_cell(cell)
        return torch.FloatTensor(cell), label

labels_train=[l for _,l,_,_ in train_files]
class_counts=np.bincount(labels_train,minlength=4)
class_weights=torch.FloatTensor([1.0/c if c>0 else 0.0 for c in class_counts]).to(DEVICE)
class_weights=class_weights/class_weights.sum()*4
print(f'Class counts (train): {class_counts}')
print(f'Class weights: {class_weights.cpu().numpy().round(3)}')
sample_weights=torch.DoubleTensor([1.0/class_counts[l] for _,l,_,_ in train_files])
balanced_sampler=WeightedRandomSampler(sample_weights,len(sample_weights))

def build_model():
    model = models.resnet50(weights='IMAGENET1K_V1')
    old_conv = model.conv1
    new_conv = nn.Conv2d(N_CH, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight = nn.Parameter(avg_weight.repeat(1, N_CH, 1, 1) / N_CH)
    model.conv1 = new_conv
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, 4))
    print(f'  {N_CH}ch ResNet50 pretrained | params: {sum(p.numel() for p in model.parameters()):,}')
    return model.to(DEVICE)

def evaluate_with_tta(model, loader, n_tta=5):
    model.eval()
    all_labels=[]; paths_list=[]
    for images, labels in loader:
        all_labels.extend(labels.numpy()); paths_list.append(images)
    accumulated_probs = None
    for tta_idx in range(n_tta):
        run_probs = []
        for images in paths_list:
            with torch.no_grad():
                if tta_idx > 0:
                    images = torch.FloatTensor(np.stack([augment_cell(img) for img in images.numpy()]))
                run_probs.extend(torch.softmax(model(images.to(DEVICE)), dim=1).cpu().numpy())
        run_probs = np.array(run_probs)
        accumulated_probs = run_probs if accumulated_probs is None else accumulated_probs + run_probs
    avg_probs = accumulated_probs / n_tta
    return avg_probs.argmax(axis=1), np.array(all_labels), avg_probs

def train_experiment(run_num):
    print(f'\n{"="*60}\n12ch Pretrained Random Split | Run {run_num}\n{"="*60}')
    train_ds = RamanDataset(train_files, protein_max, SESSION_SCALE, augment=True)
    val_ds   = RamanDataset(val_files,   protein_max, SESSION_SCALE, augment=False)
    test_ds  = RamanDataset(test_files,  protein_max, SESSION_SCALE, augment=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=balanced_sampler, num_workers=4)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=4)
    model = build_model()
    backbone_params = [p for n,p in model.named_parameters() if 'fc' not in n]
    head_params = [p for n,p in model.named_parameters() if 'fc' in n]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR * 0.1},
        {'params': head_params, 'lr': LR}
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
    best_val_f1=0.0; patience_count=0; best_state=None
    for epoch in range(NUM_EPOCHS):
        model.train(); train_loss=0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        model.eval(); val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                val_preds.extend(model(images.to(DEVICE)).argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.numpy())
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f} | Best: {best_val_f1:.4f} | Pat: {patience_count}/{PATIENCE}')
        if patience_count >= PATIENCE:
            print(f'  Early stopping at epoch {epoch}'); break
    model.load_state_dict(best_state)
    test_preds, test_labels, test_probs = evaluate_with_tta(model, test_loader)
    acc = (test_preds == test_labels).mean() * 100
    macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    macro_auc = roc_auc_score(label_binarize(test_labels, classes=[0,1,2,3]), test_probs, average='macro', multi_class='ovr')
    cm = confusion_matrix(test_labels, test_preds)
    print(f'\n  Test Accuracy {acc:.2f}% | Macro F1 {macro_f1:.4f} | Macro AUC {macro_auc:.4f}')
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, zero_division=0))
    print(f'Confusion Matrix:\n{cm}')
    torch.save(best_state, os.path.join(RESULTS_DIR, f'best_model_run{run_num}.pth'))
    return {'run':run_num,'accuracy':acc,'macro_f1':macro_f1,'macro_auc':macro_auc}

all_results = []
for run in range(1, NUM_RUNS+1):
    torch.manual_seed(SEED+run)
    all_results.append(train_experiment(run))

accs=[r['accuracy'] for r in all_results]
f1s=[r['macro_f1'] for r in all_results]
aucs=[r['macro_auc'] for r in all_results]
print(f'\n{"="*60}')
print(f'FINAL SUMMARY — 12ch Pretrained RANDOM SPLIT (upper bound)')
print(f'{"="*60}')
print(f'Accuracy:  {np.mean(accs):.2f}% +/-{np.std(accs):.2f}')
print(f'Macro F1:  {np.mean(f1s):.4f} +/-{np.std(f1s):.4f}')
print(f'Macro AUC: {np.mean(aucs):.4f} +/-{np.std(aucs):.4f}')
print('Done')
