import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from nilearn import datasets, maskers
import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 0. CONFIG
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

CACHE_DIR = os.environ.get("HF_HOME", "./hf_cache")
DATA_DIR = os.environ.get("DATA_DIR", "./abide_data")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- SETTINGS ---
MAX_SUBJECTS = 100
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EPOCHS = 15
LR = 2e-4
N_ROIS = 39
MAX_SEQ_LEN = 100

# ==========================================
# 1. SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """You are an expert neuroscientist. Analyze the provided brain connectivity embedding.
Output valid JSON only.
Format: {"Diagnosis": "Autism Spectrum Disorder (ASD)"} or {"Diagnosis": "Typically Developing Control (TC)"}.
"""

# ==========================================
# 2. ARCHITECTURE
# ==========================================
class TemporalGraphEncoder(nn.Module):
    def __init__(self, n_rois, latent_dim=64):
        super().__init__()
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(n_rois, n_rois * 2, kernel_size=3, padding=1, groups=n_rois),
            nn.BatchNorm1d(n_rois * 2), nn.ReLU(),
            nn.Conv1d(n_rois * 2, n_rois, kernel_size=3, padding=1, groups=n_rois),
            nn.BatchNorm1d(n_rois), nn.ReLU()
        )
        self.time_proj = nn.Linear(MAX_SEQ_LEN, latent_dim)
        self.W_q = nn.Linear(latent_dim, latent_dim)
        self.W_k = nn.Linear(latent_dim, latent_dim)
        self.scale = latent_dim ** -0.5

    def forward(self, x):
        x = x.transpose(1, 2)
        x_temp = self.temporal_enc(x)
        h = self.time_proj(x_temp)
        Q = self.W_q(h)
        K = self.W_k(h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        adj = F.softmax(scores, dim=-1)
        return h, adj


class ClinicalBrainLLM(nn.Module):
    def __init__(self, n_rois):
        super().__init__()
        self.llm_id = os.environ.get(
            "MODEL_ID",
            "meta-llama/Meta-Llama-3-8B-Instruct"
        )

        print(f"Loading {self.llm_id}...")

        use_4bit = DEVICE == "cuda"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_id, token=HF_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_id,
            quantization_config=bnb_config if use_4bit else None,
            device_map="auto" if DEVICE == "cuda" else None,
            torch_dtype=DTYPE,
            token=HF_TOKEN,
        )

        if DEVICE != "cuda":
            self.llm.to(DEVICE)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)

        d_latent = 128
        self.graph_module = TemporalGraphEncoder(n_rois, latent_dim=128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_latent, nhead=4, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.projector = nn.Linear(d_latent, self.llm.config.hidden_size)
        self.ln_llm = nn.LayerNorm(self.llm.config.hidden_size)
        self.query_tokens = nn.Parameter(torch.randn(1, 8, d_latent))

        # SAFE DEVICE MOVE
        self.graph_module.to(DEVICE, dtype=DTYPE)
        self.encoder.to(DEVICE, dtype=DTYPE)
        self.projector.to(DEVICE, dtype=DTYPE)
        self.ln_llm.to(DEVICE, dtype=DTYPE)
        self.query_tokens.data = self.query_tokens.data.to(DEVICE, dtype=DTYPE)

    def forward(self, bold, input_ids, attention_mask=None, labels=None):
        with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE=="cuda")):
            bold = torch.nan_to_num(bold, nan=0.0).to(DEVICE)

            h, adj = self.graph_module(bold)
            h_graph = torch.matmul(adj, h)
            feat = self.encoder(h_graph)

            q = self.query_tokens.repeat(bold.shape[0], 1, 1)
            context = F.scaled_dot_product_attention(q, feat, feat)

            brain_embeds = self.projector(context)
            brain_embeds = self.ln_llm(brain_embeds)

            text_embeds = self.llm.get_input_embeddings()(input_ids.to(DEVICE))
            inputs_embeds = torch.cat([brain_embeds, text_embeds], dim=1)

            if attention_mask is not None:
                brain_mask = torch.ones((bold.shape[0], 8), device=DEVICE)
                attention_mask = torch.cat([brain_mask, attention_mask.to(DEVICE)], dim=1)

            if labels is not None:
                brain_labels = torch.full((bold.shape[0], 8), -100, device=DEVICE)
                labels = torch.cat([brain_labels, labels.to(DEVICE)], dim=1)

            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )

    def generate_report(self, bold):
        with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE=="cuda")):
            bold = torch.nan_to_num(bold, nan=0.0).to(DEVICE)

            h, adj = self.graph_module(bold)
            h_graph = torch.matmul(adj, h)
            feat = self.encoder(h_graph)

            q = self.query_tokens.repeat(bold.shape[0], 1, 1)
            context = F.scaled_dot_product_attention(q, feat, feat)

            brain_embeds = self.projector(context)
            brain_embeds = self.ln_llm(brain_embeds)

            prompt = f"<|begin_of_text|>Analyze fMRI<|end_of_text|>"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

            text_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
            inputs_embeds = torch.cat([brain_embeds, text_embeds], dim=1)

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=50,
                do_sample=False
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ==========================================
# 3. DATASET
# ==========================================
class InstructionABIDEDataset(Dataset):
    def __init__(self, tokenizer, data_dir=DATA_DIR):
        self.tokenizer = tokenizer

        print("Downloading/loading ABIDE dataset...")

        self.atlas = datasets.fetch_atlas_msdl()
        self.masker = maskers.NiftiMapsMasker(
            maps_img=self.atlas.maps,
            standardize="zscore_sample",
            memory=os.path.join(data_dir, 'nilearn_cache'),
            verbose=0
        )

        self.abide = datasets.fetch_abide_pcp(
            data_dir=data_dir,
            pipeline="cpac",
            quality_checked=True
        )

        self.data = []
        self._process_data()

    def _process_data(self):
        pheno = pd.DataFrame(self.abide.phenotypic)

        for i, func_file in enumerate(self.abide.func_preproc):
            if MAX_SUBJECTS and len(self.data) >= MAX_SUBJECTS:
                break

            try:
                if not os.path.exists(func_file):
                    continue

                ts = self.masker.fit_transform(func_file)
                ts = np.nan_to_num(ts)

                T, N = ts.shape
                if T > MAX_SEQ_LEN:
                    ts = ts[:MAX_SEQ_LEN]
                elif T < MAX_SEQ_LEN:
                    ts = np.vstack([ts, np.zeros((MAX_SEQ_LEN - T, N))])

                dx = pheno.iloc[i]['DX_GROUP']

                self.data.append({
                    'bold': torch.tensor(ts, dtype=torch.float32),
                    'label': dx
                })

            except:
                continue

        print(f"Loaded {len(self.data)} subjects")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]

        text = json.dumps({
            "Diagnosis": "Autism Spectrum Disorder (ASD)" if item['label']==1 else "Typically Developing Control (TC)"
        })

        enc = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=128)

        return {
            'bold': item['bold'],
            'input_ids': enc.input_ids.squeeze(),
            'attention_mask': enc.attention_mask.squeeze(),
            'labels': enc.input_ids.squeeze(),
            'raw_label': item['label']
        }


# ==========================================
# 4. TRAIN + EVAL
# ==========================================
def evaluate_model(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            text_out = model.generate_report(batch['bold'])

            pred = 1 if "Autism" in text_out else 2
            y_pred.append(pred)
            y_true.append(batch['raw_label'].item())

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def run_pipeline():
    print(f"Running on {DEVICE}")

    model = ClinicalBrainLLM(N_ROIS)
    dataset = InstructionABIDEDataset(model.tokenizer)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                batch['bold'],
                batch['input_ids'],
                batch['attention_mask'],
                batch['labels']
            )

            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    evaluate_model(model, test_loader)


if __name__ == "__main__":
    run_pipeline()
