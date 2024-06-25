import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import itertools
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import codebert_design
import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Select CUDA')
SEED = 0

MODEL_TYPE = codebert_design.MODEL_TYPE
TOKENIZER = codebert_design.TOKENIZER
LEAENING_RATE = codebert_design.LEAENING_RATE
BATCH_SIZE = codebert_design.BATCH_SIZE
N_EPOCHS = codebert_design.N_EPOCHS
output_model_dir = '../model'
datasets_path = '../datasets/train_and_test/'
fig_fontsize = 24

patience = 5
patience_counter = 0

def fmt_x(x):
    x_replase = x.replace('<NUMBER>', 'NUM')
    output_x = '<s> ' + x_replase + ' </s>'
    return output_x

def calc_weights(labels):
    counts = [labels.count(i) for i in range(2)]
    weights = [1. / c for c in counts]
    total = sum(weights)
    return [w / total for w in weights]


os.mkdir(output_model_dir)
os.mkdir(checkpoint_path := os.path.join(output_model_dir, 'checkpoint'))
os.mkdir(roc_path := os.path.join(output_model_dir, 'roc'))
os.mkdir(pr_path := os.path.join(output_model_dir, 'pr'))
os.mkdir(cm_path := os.path.join(output_model_dir, 'confusion_matrix'))
df_one_train = pd.read_csv(os.path.join(datasets_path, 'one_train.csv'), usecols = ["shaped_code"])
df_one_test = pd.read_csv(os.path.join(datasets_path, 'one_test.csv'), usecols = ["shaped_code"])
df_zero_train = pd.read_csv(os.path.join(datasets_path, 'zero_train.csv'), usecols = ["shaped_code"])
df_zero_test = pd.read_csv(os.path.join(datasets_path, 'zero_test.csv'), usecols = ["shaped_code"])

df_Y_one_train = pd.read_csv(os.path.join(datasets_path, 'one_train.csv'), usecols = ["label"])
df_Y_one_test = pd.read_csv(os.path.join(datasets_path, 'one_test.csv'), usecols = ["label"])
df_Y_zero_train = pd.read_csv(os.path.join(datasets_path, 'zero_train.csv'), usecols = ["label"])
df_Y_zero_test = pd.read_csv(os.path.join(datasets_path, 'zero_test.csv'), usecols = ["label"])

df_train = pd.concat([df_one_train, df_zero_train], axis=0, ignore_index=True)
df_test = pd.concat([df_one_test, df_zero_test], axis=0, ignore_index=True)
df_Y_train = pd.concat([df_Y_one_train, df_Y_zero_train], axis=0, ignore_index=True)
df_Y_test = pd.concat([df_Y_one_test, df_Y_zero_test], axis=0, ignore_index=True)

x_train = df_train.to_numpy().reshape(-1)
x_test = df_test.to_numpy().reshape(-1)
fmt_x_all = np.vectorize(fmt_x)
x_train = fmt_x_all(x_train)
x_test = fmt_x_all(x_test)
y_train = list(itertools.chain.from_iterable(df_Y_train.values.tolist()))
y_test = list(itertools.chain.from_iterable(df_Y_test.values.tolist()))
print('Loaded')


#損失関数を用いて不均衡データに対処
weights_all = calc_weights(y_train)
weights = [weights_all[1] / weights_all[0]]
print(weights)
class_weights = torch.Tensor(weights).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)

train_toks = []
for sent in x_train:
    tok = TOKENIZER.encode_plus(sent,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True)
    train_toks.append(tok)

test_toks = []
for sent in x_test:
    tok = TOKENIZER.encode_plus(sent,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True)
    test_toks.append(tok)

train_dataset = dataloader.InlineChangeDataset(train_toks, y_train)
test_dataset = dataloader.InlineChangeDataset(test_toks, y_test)
train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        shuffle=False,
)

model = codebert_design.CodeBert(MODEL_TYPE, TOKENIZER)
model.to(device)

num_training_steps = ((sample_size := weights_all[0] + weights_all[1]) // BATCH_SIZE) * N_EPOCHS
optimizer = AdamW(model.parameters(), lr=LEAENING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

train_losses, test_losses = [], []
print('Running')

best_val_loss = float('inf')

for epoch in range(N_EPOCHS):
    print(f"Epoch-{epoch}")
    train_losses += dataloader.train_loop(train_dataloader, model, optimizer, device, tqdm, loss_fn)
    y_pred, test_loss = dataloader.test_loop(test_dataloader, model, device, tqdm, loss_fn)
    scheduler.step(test_loss)
    test_losses.append(test_loss)

    # 各epochでのConfusion Matrixを描画
    _y_pred = (np.array(y_pred) > 0.5).astype(int)
    cm = confusion_matrix(y_test, _y_pred)
    cm_df = pd.DataFrame(cm,columns=['Predicted False', 'Predicted True'], index=['Actual False', 'Actual True'])
    plt.figure(figsize=(10,7))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(cm_path, f'confusion_matrix_{epoch}.png'))

    # ROC curveとAUCの計算
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # PR curveとAUCの計算
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    # 各epochでのROC curveとPR curveを描画
    plt.figure(figsize=(10,10))

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, linewidth = 5)
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth = 5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fig_fontsize)
    plt.ylabel('True Positive Rate', fontsize=fig_fontsize)
    plt.title('Receiver Operating Characteristic', fontsize=fig_fontsize)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(roc_path, f'roc_curve_{epoch}.pdf'))
    plt.clf()

    plt.figure(figsize=(10,10))
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc, linewidth = 5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=fig_fontsize)
    plt.ylabel('Precision', fontsize=fig_fontsize)
    plt.title('Precision-Recall curve', fontsize=fig_fontsize)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(pr_path, f'pr_curve_{epoch}.pdf'))
    plt.clf()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_path, f'checkpoint_{epoch}.pth'))

    if test_loss < best_val_loss:
        best_val_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

