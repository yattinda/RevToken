import pandas as pd
import torch
import torch.nn as nn
from transformers import AdamW
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import sys
import itertools
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
sys.path.append(os.pardir)
from train import codebert_design, dataloader, path_storage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Select CUDA')

BATCH_SIZE = 64
MODEL_TYPE = codebert_design.MODEL_TYPE
TOKENIZER = codebert_design.TOKENIZER
LEAENING_RATE = codebert_design.LEAENING_RATE
N_EPOCHS = 20
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([51.95528804815134]))
datasets_path = path_storage.datasets_path
output_path = '/work/yasuhito-m/workspace/review_for_codebert/QtBase/eval_for_linelevel'

def fmt_x(x):
    x_replase = x.replace('<NUMBER>', 'NUM')
    output_x = '<s> ' + x_replase + ' </s>'
    return output_x


df_one_test = pd.read_csv(os.path.join(datasets_path, 'one_test.csv'), usecols = ["shaped_code"])
df_zero_test = pd.read_csv(os.path.join(datasets_path, 'zero_test.csv'), usecols = ["shaped_code"])

df_Y_one_test = pd.read_csv(os.path.join(datasets_path, 'one_test.csv'), usecols = ["label"])
df_Y_zero_test = pd.read_csv(os.path.join(datasets_path, 'zero_test.csv'), usecols = ["label"])

df_test = pd.concat([df_one_test, df_zero_test], axis=0, ignore_index=True)
df_Y_test = pd.concat([df_Y_one_test, df_Y_zero_test], axis=0, ignore_index=True)

x_test = df_test.to_numpy().reshape(-1)
fmt_x_all = np.vectorize(fmt_x)
x_test = fmt_x_all(x_test)
y_test = list(itertools.chain.from_iterable(df_Y_test.values.tolist()))
print('Loaded')

test_toks = []
for sent in x_test:
    tok = TOKENIZER.encode_plus(sent,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True)
    test_toks.append(tok)

test_dataset = dataloader.InlineChangeDataset(test_toks, y_test)
test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        shuffle=False,
)

scores_df = pd.DataFrame(columns=['Epoch', 'F1 Score', 'Recall', 'FAR', 'D2H'])

model = codebert_design.CodeBert(MODEL_TYPE, TOKENIZER)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEAENING_RATE)

for i in range(N_EPOCHS):
    checkpoint = torch.load(f'/work/yasuhito-m/workspace/review_for_codebert/QtBase/model/checkpoint/checkpoint_{i}.pth', map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    y_pred, _ = dataloader.test_loop(test_dataloader, model, device, tqdm, loss_fn)
    _y_pred = (np.array(y_pred) > 0.5).astype(int)
    cm = confusion_matrix(y_test, _y_pred)
    cm_df = pd.DataFrame(cm, columns=['Predicted False', 'Predicted True'], index=['Actual False', 'Actual True'])

    tn, fp, fn, tp = confusion_matrix(y_test, _y_pred).ravel()
    far = fp / (fp + tn)
    recall = recall_score(y_test, _y_pred)
    d2h = (((((1 - recall) ** 2) + ((0 - far) ** 2))) / 2 ) ** 0.5
    f1 = f1_score(y_test, _y_pred)

    temp_df = pd.DataFrame({
        'Epoch': [i],
        'F1 Score': [f1],
        'Recall': [recall],
        'FAR': [far],
        'D2H': [d2h]
    })
    scores_df = pd.concat([scores_df, temp_df], ignore_index=True)

scores_df.to_csv(os.path.join(output_path, 'result.csv'), index=False)