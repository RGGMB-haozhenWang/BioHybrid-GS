import os
import copy
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings

# ==========================================
# 0. Configuration & Args
# ==========================================
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="BioHybrid Transformer for Rice Genomic Prediction")
    
    # Path Arguments (Defaults to current directory structure)
    parser.add_argument('--base_dir', type=str, default='./', help='Root directory of the project')
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory containing genotype/phenotype data')
    parser.add_argument('--out_dir', type=str, default='results/', help='Directory to save outputs')
    
    # Hyperparameters
    parser.add_argument('--runs', type=int, default=5, help='Number of ensemble runs')
    parser.add_argument('--epochs', type=int, default=120, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr_bg', type=float, default=5e-4, help='Learning rate for background stream')
    parser.add_argument('--lr_trans', type=float, default=1e-4, help='Learning rate for transformer stream')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed base')
    
    return parser.parse_args()

# ==========================================
# 1. Data Processing Utils
# ==========================================

def clean_column_names(df):
    new_cols = {}
    for col in df.columns:
        if col in ['FID','IID','PAT','MAT','SEX','PHENOTYPE']: continue
        new_name = re.sub(r'_[A-Z]$', '', col)
        new_cols[col] = new_name
    df.rename(columns=new_cols, inplace=True)
    return df

def load_genotype_data(data_path):
    print("[Data] Loading Genotypes...")
    geno_path = os.path.join(data_path, "raw_data_for_gs.raw")
    if not os.path.exists(geno_path): 
        raise FileNotFoundError(f"Genotype file not found: {geno_path}")
        
    geno_df = pd.read_csv(geno_path, sep='\s+')
    geno_df = clean_column_names(geno_df)
    geno_df.index = geno_df['IID']
    snp_cols = [c for c in geno_df.columns if ':' in c]
    snp_data = geno_df[snp_cols]
    
    # Simple Mode Imputation
    modes = snp_data.mode().iloc[0]
    snp_data = snp_data.fillna(modes)
    print(f"[Data] Loaded {snp_data.shape[1]} SNPs for {snp_data.shape[0]} samples.")
    return snp_data

def split_snps_features(snp_df, data_path, prior_files):
    print("[Data] Mapping Bio-Priors...")
    all_snps = set(snp_df.columns)
    prior_sets = {}
    all_prior_snps = set()
    
    for key, filename in prior_files.items():
        # Look for priors in the 'priors' subdirectory or main data dir
        path = os.path.join(data_path, "priors", filename)
        if not os.path.exists(path):
            path = os.path.join(data_path, filename) # Try root data dir
            
        if os.path.exists(path):
            with open(path) as f:
                raw_list = [x.strip() for x in f if x.strip()]
            valid_list = [s for s in raw_list if s in all_snps]
            valid_list.sort()
            prior_sets[key] = valid_list
            all_prior_snps.update(valid_list)
            print(f"  -> [{key}] Found {len(valid_list)} SNPs")
        else:
            print(f"  [Warning] Prior file {filename} not found.")
            prior_sets[key] = []
            
    background_snps = list(all_snps - all_prior_snps)
    print(f"  -> [Background] {len(background_snps)} SNPs.")
    return prior_sets, background_snps

def load_phenotypes_raw(data_path):
    def read_file(fname, env_name):
        path = os.path.join(data_path, fname)
        if not os.path.exists(path): 
            # print(f"  [Info] Phenotype file {fname} not found, skipping.")
            return None
        df = pd.read_csv(path)
        # Compatible column renaming
        df = df.rename(columns={df.columns[0]: 'Taxa', df.columns[1]: 'Raw_Value'})
        df['Env'] = env_name
        return df

    # Specific file names consistent with the manuscript
    mcc_14bj = read_file('pheno_BLUP_RSSR_2014BJ_cleaned.csv', '14BJ')
    mcc_15yn = read_file('pheno_BLUP_SSR_2015YN_cleaned.csv', '15YN')
    mod_22km = read_file('pheno_SSR_under_cold_tolerance_2022_Kunming_cleaned.csv', '22KM')
    mod_23gz = read_file('pheno_SSR_under_cold_tolerance_2023_Gongzhuling_cleaned.csv', '23GZ')
    mod_23km = read_file('pheno_SSR_under_cold_tolerance_2023_Kunming_cleaned.csv', '23KM')
    
    return {'14BJ': mcc_14bj, '15YN': mcc_15yn}, {'22KM': mod_22km, '23GZ': mod_23gz, '23KM': mod_23km}

# ==========================================
# 2. Optimized BioHybrid Transformer
# ==========================================

class BioHybridTransformer(nn.Module):
    def __init__(self, prior_dims, background_dim, d_model=16, nhead=2): 
        super(BioHybridTransformer, self).__init__()
        
        # --- Part 1: Background (Linear) ---
        self.background_encoder = nn.Linear(background_dim, 16)
        self.bg_norm = nn.LayerNorm(16)
        self.bg_head = nn.Linear(16, 1)
        
        # --- Part 2: Bio-Prior (Transformer) ---
        self.branch_keys = sorted(prior_dims.keys())
        self.embeddings = nn.ModuleDict()
        
        for key in self.branch_keys:
            # Handle empty prior sets gracefully
            input_dim = prior_dims[key] if prior_dims[key] > 0 else 1
            self.embeddings[key] = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.3) 
            )
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=32, 
                                                 batch_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.deep_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * len(self.branch_keys), 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        self.gate = nn.Parameter(torch.tensor(3.0)) 

    def forward(self, x_prior_dict, x_background):
        bg_feat = F.gelu(self.bg_norm(self.background_encoder(x_background)))
        bg_pred = self.bg_head(bg_feat)
        
        tokens = []
        for key in self.branch_keys:
            tok = self.embeddings[key](x_prior_dict[key]).unsqueeze(1)
            tokens.append(tok)
        
        seq_input = torch.cat(tokens, dim=1)
        trans_out = self.transformer(seq_input)
        deep_pred = self.deep_head(trans_out)
        
        alpha = torch.sigmoid(self.gate)
        final_pred = alpha * bg_pred + (1 - alpha) * deep_pred
        return final_pred

# ==========================================
# 3. Training & Helper Functions
# ==========================================

def get_tensors(df, snp_df, prior_sets, bg_snps, device, scaler=None, fit_scaler=False):
    valid_taxa = [t for t in df['Taxa'] if t in snp_df.index]
    if not valid_taxa: return None, None, None, None, None
    
    raw_y = df.set_index('Taxa').loc[valid_taxa]['Raw_Value'].values.reshape(-1, 1)
    
    if fit_scaler:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(raw_y)
    else:
        if scaler is None: raise ValueError("Scaler needed for test data")
        y_scaled = scaler.transform(raw_y)
        
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(device)
    
    snp_sub = snp_df.loc[valid_taxa]
    x_bg = torch.tensor(snp_sub[bg_snps].values, dtype=torch.float32).to(device)
    
    x_prior = {}
    for k, cols in prior_sets.items():
        if len(cols) > 0:
            x_prior[k] = torch.tensor(snp_sub[cols].values, dtype=torch.float32).to(device)
        else:
            # Placeholder for empty priors to prevent crash in demo mode
            x_prior[k] = torch.zeros((len(valid_taxa), 1), dtype=torch.float32).to(device)
        
    return x_prior, x_bg, y_tensor, scaler, valid_taxa

def train_single_model(X_p, X_b, y, p_dims, b_dim, args, device, seed):
    torch.manual_seed(seed)
    model = BioHybridTransformer(p_dims, b_dim).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.background_encoder.parameters(), 'lr': args.lr_bg, 'weight_decay': 0.1}, 
        {'params': model.bg_head.parameters(), 'lr': args.lr_bg, 'weight_decay': 0.1},
        {'params': model.embeddings.parameters(), 'lr': args.lr_trans, 'weight_decay': 0.01},
        {'params': model.transformer.parameters(), 'lr': args.lr_trans, 'weight_decay': 0.01},
        {'params': model.deep_head.parameters(), 'lr': args.lr_trans, 'weight_decay': 0.01},
        {'params': model.gate, 'lr': 0.01}
    ])
    
    loss_fn = nn.MSELoss()
    dataset_size = len(y)
    indices = np.arange(dataset_size)
    best_loss = float('inf')
    best_weights = None
    
    model.train()
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, dataset_size, args.batch_size):
            batch_idx = indices[start_idx : min(start_idx+args.batch_size, dataset_size)]
            
            xp_batch = {k: v[batch_idx] for k,v in X_p.items()}
            xb_batch = X_b[batch_idx]
            y_batch = y[batch_idx]
            
            optimizer.zero_grad()
            pred = model(xp_batch, xb_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            full_pred = model(X_p, X_b)
            val_loss = loss_fn(full_pred, y).item()
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_weights)
    return model

def analyze_snp_importance(models, X_p, X_b, prior_sets, device, out_path):
    print("\n=== Calculating SNP Importance (Saliency Map) ===")
    for k in X_p: X_p[k].requires_grad = True
    
    total_importance = {}
    for k in prior_sets:
        if len(prior_sets[k]) == 0: continue
        total_importance[k] = torch.zeros(len(prior_sets[k])).to(device)
        
    for model in models:
        model.eval()
        model.zero_grad()
        pred = model(X_p, X_b)
        pred.sum().backward()
        with torch.no_grad():
            for k in prior_sets:
                if len(prior_sets[k]) > 0 and X_p[k].grad is not None:
                    grad = X_p[k].grad.abs().mean(dim=0)
                    total_importance[k] += grad
        for k in X_p:
            if X_p[k].grad is not None: X_p[k].grad.zero_()
            
    importance_list = []
    for k in prior_sets:
        if k not in total_importance: continue
        avg_imp = (total_importance[k] / len(models)).cpu().numpy()
        if avg_imp.max() > 0: avg_imp = avg_imp / avg_imp.max()
        for snp, score in zip(prior_sets[k], avg_imp):
            importance_list.append({'Source': k, 'SNP': snp, 'Importance_Score': score})
            
    df_imp = pd.DataFrame(importance_list).sort_values('Importance_Score', ascending=False)
    
    # Plotting
    if not df_imp.empty:
        plt.figure(figsize=(10, 5))
        colors = {'GWAS': '#E64B35', 'DAS': '#4DBBD5', 'Hub': '#00A087'} 
        for source, group in df_imp.groupby('Source'):
            plt.scatter(group.index, group['Importance_Score'], label=source, color=colors.get(source, 'grey'), alpha=0.6, s=15)
        plt.title('Bio-Prior SNP Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, "Importance_Manhattan.png"))
        print(f"[Plot] Saved Manhattan plot.")
        
    return df_imp

# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    args = get_args()
    
    # Setup Paths
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
    device = torch.device("cpu") # Keep CPU for stability as requested
    print(f"[System] Device: {device} | Output: {args.out_dir}")

    try:
        # Load Data
        snp_df = load_genotype_data(args.data_dir)
        
        # Prior Files (Try to load these specific names)
        prior_files = {'GWAS': 'gwas_snps_793.txt', 'DAS': 'das_364.txt', 'Hub': 'hub_pro_genebody_1694.txt'}
        prior_sets, bg_snps = split_snps_features(snp_df, args.data_dir, prior_files)
        
        p_dims = {k: len(v) for k,v in prior_sets.items()}
        b_dim = len(bg_snps)
        
        train_sets, test_sets = load_phenotypes_raw(args.data_dir)
        
        print(f"\n=== Starting BioHybrid Ensemble ({args.runs} runs) ===")
        final_results = []
        
        for tr_name, tr_df in train_sets.items():
            if tr_df is None: continue
            print(f"\n>>> Training Environment: {tr_name} ({len(tr_df)} samples)")
            
            X_p_tr, X_b_tr, y_tr, scaler, _ = get_tensors(tr_df, snp_df, prior_sets, bg_snps, device, fit_scaler=True)
            if y_tr is None: continue
            
            models = [train_single_model(X_p_tr, X_b_tr, y_tr, p_dims, b_dim, args, device, seed=args.seed+i) for i in range(args.runs)]
            
            # Predict
            for te_name, te_df in test_sets.items():
                if te_df is None: continue
                X_p_te, X_b_te, _, _, valid_taxa_te = get_tensors(te_df, snp_df, prior_sets, bg_snps, device, scaler=scaler)
                if X_p_te is None: continue
                
                obs_raw = te_df.set_index('Taxa').loc[valid_taxa_te]['Raw_Value'].values
                preds_scaled = []
                for m in models:
                    m.eval()
                    with torch.no_grad():
                        preds_scaled.append(m(X_p_te, X_b_te).cpu().numpy().flatten())
                
                avg_pred_raw = scaler.inverse_transform(np.mean(preds_scaled, axis=0).reshape(-1, 1)).flatten()
                r, _ = pearsonr(obs_raw, avg_pred_raw)
                rmse = np.sqrt(np.mean((obs_raw - avg_pred_raw)**2))
                print(f"    -> Test on {te_name}: r = {r:.4f} | RMSE = {rmse:.4f}")
                
                final_results.append({'Train_Env': tr_name, 'Test_Env': te_name, 'Pearson_r': r, 'RMSE': rmse})

            # Feature Importance (Example: 15YN -> 23KM)
            if tr_name == '15YN' and '23KM' in test_sets and test_sets['23KM'] is not None:
                 X_p_int, X_b_int, _, _, _ = get_tensors(test_sets['23KM'], snp_df, prior_sets, bg_snps, device, scaler=scaler)
                 imp_df = analyze_snp_importance(models, X_p_int, X_b_int, prior_sets, device, args.out_dir)
                 imp_df.to_csv(os.path.join(args.out_dir, "Top_Functional_SNPs.csv"), index=False)

        pd.DataFrame(final_results).to_csv(os.path.join(args.out_dir, "Final_Results.csv"), index=False)
        print("\n[Done] Pipeline finished.")
        
    except Exception as e:
        print(f"\n[Error] {str(e)}")
        print("Tip: Ensure data path is correct or run 'python create_demo_data.py' to generate sample data.")