import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import os
import zipfile
import urllib.request
from tqdm import tqdm

# --- 1. MODEL ARCHITECTURE ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, num_classes=5):
        super().__init__()
        self.lstm    = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                               bidirectional=True, num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        h = self.dropout(h)
        return self.fc(h)


# Helper class to display a download progress bar in the terminal
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def load_glove():
    zip_path = "glove.6B.zip"
    txt_path = "glove/glove.6B.300d.txt"
    
    os.makedirs("glove", exist_ok=True)
    
    # 1. DOWNLOAD WITH PROGRESS BAR
    if not os.path.exists(txt_path):
        if not os.path.exists(zip_path):
            print("Downloading GloVe embeddings (~822 MB)...")
            url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading GloVe") as t:
                urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
        
        # 2. EXTRACT WITH FEEDBACK
        print("Extracting GloVe zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("glove")
        print("Extraction complete!")

    # 3. LOAD TEXT FILE WITH LINE-BY-LINE PROGRESS BAR
    embeddings = {}
    print("Loading vectors into memory...")
    with open(txt_path, encoding="utf-8") as f:
        # GloVe 6B 300d has exactly 400,000 lines/words
        for line in tqdm(f, total=400000, desc="Parsing GloVe Vectors"):
            values = line.split()
            embeddings[values[0]] = np.array(values[1:], dtype=np.float32)
            
    print(f"Successfully loaded {len(embeddings):,} GloVe vectors!")
    return embeddings

def text_to_seq(text, glove_dict, max_len=50, dim=300):
    tokens = str(text).lower().split()[:max_len]
    vecs   = [glove_dict.get(t, np.zeros(dim, dtype=np.float32)) for t in tokens]
    while len(vecs) < max_len:
        vecs.append(np.zeros(dim, dtype=np.float32))
    return np.array(vecs, dtype=np.float32)

device = torch.device("cpu")
model = BiLSTMClassifier()
if os.path.exists("bilstm_weights.pth"):
    model.load_state_dict(torch.load("bilstm_weights.pth", map_location=device))
model.to(device)
model.eval()

glove = load_glove()
OPTIONS = ["A", "B", "C", "D", "E"]

# --- 3. PREDICTION FUNCTION ---
def predict_mcq(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    if not os.path.exists("bilstm_weights.pth"):
        return "Error: Model weights not found.", {}
        
    input_options = [opt_a, opt_b, opt_c, opt_d, opt_e]
    if not prompt or not all(input_options):
        return "Please fill in the prompt and all 5 options.", {}

    seqs = []
    for opt in input_options:
        combined = str(prompt) + " " + str(opt)
        seqs.append(text_to_seq(combined, glove))
        
    seqs_tensor = torch.tensor(np.array(seqs, dtype=np.float32))
    
    with torch.no_grad():
        scores = []
        for i in range(5):
            s = model(seqs_tensor[i].unsqueeze(0))[:, i].item()
            scores.append(s)

    # Format outputs
    top_3_idx = np.argsort(scores)[::-1][:3]
    top_3_preds = " ".join([OPTIONS[i] for i in top_3_idx])
    
    # Create a dictionary of scores for the Gradio Label component
    confidence_dict = {OPTIONS[i]: float(scores[i]) for i in range(5)}
    
    return f"Top 3 Predictions: {top_3_preds}", confidence_dict

# --- 4. GRADIO UI LAYOUT ---
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Smart MCQ Solver (BiLSTM)")
    gr.Markdown("Enter a prompt and 5 options to get the top 3 predicted answers.")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(lines=3, label="Question / Prompt")
            opt_a_input = gr.Textbox(label="Option A")
            opt_b_input = gr.Textbox(label="Option B")
            opt_c_input = gr.Textbox(label="Option C")
            opt_d_input = gr.Textbox(label="Option D")
            opt_e_input = gr.Textbox(label="Option E")
            submit_btn = gr.Button("Predict Answer", variant="primary")
            
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="Result", lines=2)
            label_output = gr.Label(label="Raw Model Scores (Logits)")

    submit_btn.click(
        fn=predict_mcq,
        inputs=[prompt_input, opt_a_input, opt_b_input, opt_c_input, opt_d_input, opt_e_input],
        outputs=[text_output, label_output]
    )

if __name__ == "__main__":
    # Launch locally or on Spaces
    demo.launch(theme=gr.themes.Soft(), ssr_mode=False)