import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Function
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import warnings; warnings.filterwarnings('ignore')

SCRATCH     = '/mnt/scratch/vvgk0135'
DATA_ROOT   = f'{SCRATCH}/extracted_2026_hs'
RESULTS_DIR = '/users/vvgk0135/raman-project/results/2026_hs_dann'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED=42; NUM_EPOCHS=300; BATCH_SIZE=32; LR=5e-4; PATIENCE=75; NUM_RUNS=3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CH_REF = 0; N_MODEL_CH = 52; N_SESSIONS = 3

CLASS_MAP = {'FHC':0,'HCT116':1,'CaCo2':2,'SW480':2,'HT29':3,'SW620':3}
CLASS_NAMES = ['Healthy (FHC)','Dukes A (HCT116)',
               'Dukes B (SW480+CaCo2)','Dukes C (HT29+SW620)']
N_CLASSES = 4
SESSION_MAP = {'s1':0,'s2':1,'s3':2}

TEST_SESSIONS = {
    'FHC':'s3','HCT116':'s3','CaCo2':'s3',
    'SW480':'s2','HT29':'s3','SW620':'s3'
}

SESSION_MEANS = {
    ('FHC','s1'):1468.7,('FHC','s2'):1552.6,('FHC','s3'):1869.8,
    ('HCT116','s1'):5985.2,('HCT116','s2'):7224.2,('HCT116','s3'):6671.3,
    ('CaCo2','s1'):7266.2,('CaCo2','s2'):7359.4,('CaCo2','s3'):5129.5,
    ('SW480','s1'):6047.0,('SW480','s2'):6357.2,('SW480','s3'):5993.5,
    ('HT29','s1'):8453.4,('HT29','s2'):9415.0,('HT29','s3'):7057.0,
    ('SW620','s1'):5509.3,('SW620','s2'):7410.7,('SW620','s3'):5724.7,
}
GLOBAL_REF = sum(SESSION_MEANS.values()) / len(SESSION_MEANS)
SESSION_SCALE = {k: GLOBAL_REF/v for k,v in SESSION_MEANS.items()}

print('Device:', DEVICE)
print('Solution 2: Domain Adversarial Training (DANN)')

# Gradient Reversal Layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

import torchvision.models as models

class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)
        base.conv1 = nn.Conv2d(N_MODEL_CH, 64, 7, 2, 3, bias=False)
        self.features = nn.Sequential(*list(base.children())[:-1])  # remove FC
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, N_CLASSES)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, N_SESSIONS)
        )
    def forward(self, x, alpha=1.0):
        feat = self.features(x).squeeze(-1).squeeze(-1)
        class_out = self.classifier(feat)
        domain_out = self.discriminator(grad_reverse(feat, alpha))
        return class_out, domain_out

def get_split():
    tr, va, te = [], [], []
    np.random.seed(SEED)
    for cl, lab in CLASS_MAP.items():
        fo = os.path.join(DATA_ROOT, cl)
        if not os.path.exists(fo): continue
        regular = []
        for f in sorted(os.listdir(fo)):
            if not f.endswith('.npy'): continue
            s = next((p for p in f.split('_') if p.startswith('s') and p[1:].isdigit()), 's1')
            rec = (os.path.join(fo, f), lab, cl, s)
            if s == TEST_SESSIONS[cl]: te.append(rec)
            else: regular.append(rec)
        nv = max(1, int(len(regular)*0.2))
        np.random.shuffle(regular)
        va.extend(regular[:nv]); tr.extend(regular[nv:])
    return tr, va, te

train_files, val_files, test_files = get_split()
print('Split: train=%d val=%d test=%d' % (len(train_files), len(val_files), len(test_files)))
counts = np.bincount([l for _,l,_,_ in train_files], minlength=N_CLASSES)
print('Train class counts:', counts)

def load_and_normalise(path, cl, s, eps=1e-6):
    c = np.load(path).astype(np.float32)
    ref = np.where(c[CH_REF] < eps, np.nan, c[CH_REF])
    c_norm = c / ref[None, :, :]
    c_norm = c_norm[1:] * SESSION_SCALE.get((cl, s), 1.0)
    c_norm = np.nan_to_num(c_norm, nan=0.0)
    p99 = np.percentile(c_norm, 99)
    if p99 > 0: c_norm = np.clip(c_norm, 0, p99) / p99
    return c_norm.astype(np.float32)

def augment(cell):
    cell = cell.copy() * np.random.uniform(0.85, 1.15)
    if np.random.random() > 0.5: cell = cell[:, :, ::-1].copy()
    if np.random.random() > 0.5: cell = cell[:, ::-1, :].copy()
    k = np.random.randint(0, 4)
    if k > 0: cell = np.rot90(cell, k, axes=(1,2)).copy()
    return np.clip(cell + np.random.normal(0, 0.005, cell.shape).astype(np.float32), 0, 1)

class DS(Dataset):
    def __init__(self, files, aug=False): self.files=files; self.aug=aug
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        p,l,cl,s = self.files[i]
        c = load_and_normalise(p, cl, s)
        if self.aug: c = augment(c)
        s_label = SESSION_MAP.get(s, 0)
        return torch.FloatTensor(c), l, s_label

inv = np.array([1.0/c if c > 0 else 0.0 for c in counts])
inv[0] *= 0.5; inv[1] *= 2.0
w = torch.FloatTensor(inv); w = (w/w.sum()*N_CLASSES).to(DEVICE)
sampler = WeightedRandomSampler(
    torch.DoubleTensor([1.0/counts[l] for _,l,_,_ in train_files]), len(train_files))

def run_once(r):
    print('='*60)
    print('DANN | Run', r)
    print('='*60)
    tr = DataLoader(DS(train_files, True), BATCH_SIZE, sampler=sampler, num_workers=4)
    va = DataLoader(DS(val_files), BATCH_SIZE, shuffle=False, num_workers=4)
    m = DANN().to(DEVICE)
    cls_crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    dom_crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-7)
    best=0.0; pat=0; st=None

    for ep in range(NUM_EPOCHS):
        m.train(); tot=0.0
        # Alpha increases gradually — start with no domain confusion, increase over time
        alpha = 2.0 / (1.0 + np.exp(-10 * ep / NUM_EPOCHS)) - 1.0
        for x,y,s_lab in tr:
            x,y,s_lab = x.to(DEVICE),y.to(DEVICE),s_lab.to(DEVICE)
            opt.zero_grad()
            cls_out, dom_out = m(x, alpha)
            loss = cls_crit(cls_out, y) + 0.1 * dom_crit(dom_out, s_lab)
            loss.backward(); opt.step(); tot+=loss.item()
        sch.step()
        m.eval(); vp,vl=[],[]
        with torch.no_grad():
            for x,y,_ in va:
                cls_out,_ = m(x.to(DEVICE), 0.0)
                vp.extend(cls_out.argmax(1).cpu().numpy()); vl.extend(y.numpy())
        vf = f1_score(vl, vp, average='macro', zero_division=0)
        if vf > best: best=vf; st={k:v.clone() for k,v in m.state_dict().items()}; pat=0
        else: pat+=1
        if ep%20==0: print('  Epoch %3d | alpha=%.2f | Loss %.4f | Val F1 %.4f | Best %.4f' % (ep, alpha, tot/len(tr), vf, best))
        if pat>=PATIENCE: print('  Early stop epoch', ep); break

    m.load_state_dict(st)
    te_ld = DataLoader(DS(test_files), BATCH_SIZE, shuffle=False, num_workers=4)
    m.eval(); probs,true=[],[]
    with torch.no_grad():
        for x,y,_ in te_ld:
            cls_out,_ = m(x.to(DEVICE), 0.0)
            probs.extend(torch.softmax(cls_out,dim=1).cpu().numpy())
            true.extend(y.numpy())
    probs=np.array(probs); true=np.array(true); pred=probs.argmax(1)
    acc=(pred==true).mean()*100
    f1=f1_score(true,pred,average='macro',zero_division=0)
    auc=roc_auc_score(label_binarize(true,classes=list(range(N_CLASSES))),
                      probs,average='macro',multi_class='ovr')
    print('Test Accuracy %.2f%% | Macro F1 %.4f | Macro AUC %.4f' % (acc, f1, auc))
    print(classification_report(true,pred,target_names=CLASS_NAMES,
                                labels=list(range(N_CLASSES)),zero_division=0))
    print(confusion_matrix(true,pred))
    torch.save(st, os.path.join(RESULTS_DIR, 'model_run%d.pth' % r))
    return {'run':r,'acc':acc,'f1':f1,'auc':auc}

results=[]
for r in range(1, NUM_RUNS+1):
    torch.manual_seed(SEED+r); np.random.seed(SEED+r)
    results.append(run_once(r))

a=[x['acc'] for x in results]; f=[x['f1'] for x in results]; u=[x['auc'] for x in results]
print('='*60)
print('FINAL SUMMARY — DANN, 52ch, session split')
print('='*60)
print('Accuracy:  %.2f%% +/-%.2f' % (np.mean(a), np.std(a)))
print('Macro F1:  %.4f +/-%.4f' % (np.mean(f), np.std(f)))
print('Macro AUC: %.4f +/-%.4f' % (np.mean(u), np.std(u)))
np.save(os.path.join(RESULTS_DIR, 'results.npy'), results)
print('Done')
