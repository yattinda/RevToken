import pandas as pd
import re
import torch
from transformers import  AdamW
import numpy as np
from matplotlib import pyplot as plt
import string
import os
import sys
import itertools
import seaborn as sns
import csv
import diff_match_patch as dmp_module
import statistics
from matplotlib_venn import venn2
sys.path.append(os.path.join(os.pardir, os.pardir))
from train import codebert_design, dataloader
from lime.lime_text import LimeTextExplainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Select CUDA')
epoch = 18
MODEL_TYPE = codebert_design.MODEL_TYPE
TOKENIZER = codebert_design.TOKENIZER
LEAENING_RATE = codebert_design.LEAENING_RATE
datasets_path = '../../datasets/train_and_test/'
more_than_n = 0
os.mkdir(RQ2_path := f"../lime/result_more_{more_than_n}")
checkpoint = torch.load(f'../../model/checkpoint/checkpoint_{epoch}.pth', map_location=device)


explainer = LimeTextExplainer(class_names=["False", "True"])

def predictor(texts):
    tok = TOKENIZER.batch_encode_plus(texts, padding=True)
    input_ids = torch.tensor(tok['input_ids']).to(device)
    attention_mask = torch.tensor(tok['attention_mask']).to(device)

    with torch.no_grad():
        output, _ = model(input_ids, attention_mask,)
    probas = output.sigmoid().cpu().numpy()
    probas = probas.flatten()
    return np.vstack([1 - probas, probas]).T

def dict_search(d, key):
    results = []
    def _search(d, key):
        if isinstance(d, dict):
            if key in d:
                results.append(d.get(key))
            for k, v in d.items():
                _search(v, key)
        elif isinstance(d, list):
            for item in d:
                _search(item, key)

    _search(d, key)
    return results

def remove_key(d, key):
    if isinstance(d, dict):
        return {k: remove_key(v, key) for k, v in d.items() if k != key}
    elif isinstance(d, list):
        return [remove_key(v, key) for v in d]
    else:
        return d

def diff_token(before, after):
    def _divide_panc(input_list):
        output_list = []
        add_prefix = False
        input_list = [item for item in input_list if item != 'Ġ']
        for element in input_list:
            if set(element).issubset(string.punctuation):
                add_prefix = True
            else:
                if add_prefix:
                    element = 'Ġ' + element
                    add_prefix = False
                output_list.append(element)
        return output_list

    def _transform_list(tokens):
        token_key = []
        omit_char = string.punctuation + 'Ġ'
        for i, item in enumerate(tokens):
            if any([item.startswith('Ġ'), item == '<s>', item == '</s>', i == 0]):
                token_key.append(item)
            else:
                token_key[-1] += item
        omitted_list = [item for item in token_key if not all(char in omit_char for char in item)]
        return [item.replace('Ġ', ' ').replace("\"", '').replace('\'', "") for item in omitted_list]

    dmp = dmp_module.diff_match_patch()
    words_before = _transform_list(_divide_panc(TOKENIZER.tokenize(before)))
    words_after = _transform_list(_divide_panc(TOKENIZER.tokenize(after)))
    # Convert the words to unique characters
    char_to_word = {}
    before_list = []
    after_list = []
    i = 0
    for word in words_before:
        if word not in char_to_word:
            char_to_word[word] = chr(i)
            i += 1
        before_list.append(char_to_word[word])
    for word in words_after:
        if word not in char_to_word:
            char_to_word[word] = chr(i)
            i += 1
        after_list.append(char_to_word[word])

    # Compute the diff
    diffs = dmp.diff_main(''.join(before_list), ''.join(after_list), False)

    # Replace the unique characters with the original words
    word_to_char = {v: k for k, v in char_to_word.items()}
    for i, (op, data) in enumerate(diffs):
        words = [word_to_char[c] for c in data]
        diffs[i] = (op, ''.join(words))
    # Only pick -1
    pick_changed_words = [t[1] for t in diffs if t[0] == -1]
    output = [word for item in pick_changed_words for word in item.split()]
    return output

def replace_with_number(s):
    try:
        if isinstance(s, str) and re.search(r'\d', s):
            return re.sub(r'\d+', 'NUM', s)
        else:
            return s
    except ValueError:
        return s
    except TypeError:
        pass

def get_top_n_keys(dictionary, n):
    modify_dict = {}
    if "<s>" in dictionary:
        del dictionary["<s>"]
    if "</s>" in dictionary:
        del dictionary["</s>"]
    if "Ġ" in dictionary:
        del dictionary["Ġ"]
    if "s" in dictionary:
        del dictionary["s"]
    for key, value in dictionary.items():
        new_key = key.replace('Ġ', '')
        modify_dict[new_key] = value
    sorted_items = sorted(modify_dict.items(), key=lambda x: x[1], reverse=True)
    top_n_keys = [item[0] for item in sorted_items[:n]]
    return top_n_keys


def find_partial_matches(list1, list2):
    matches = []
    for item1 in list1:
        for item2 in list2:
            if item1 in item2:
                matches.append(item1)
    return matches

def F1_score(recall, precision):
    return (2 * recall * precision) / (recall + precision)

def calc_weights(labels):
    counts = [labels.count(i) for i in range(2)]
    weights = [1. / c for c in counts]
    total = sum(weights)
    return [w / total for w in weights]

def logging_result(before_code, after_code, acttually_changed_token_lst, top_n_lst):
    logs.write(before_code)
    logs.write("\n#################################\n")
    logs.write(after_code)
    logs.write("\n")
    logs.write('changed_token:' + str(acttually_changed_token_lst))
    logs.write("\n")
    logs.write('suggestion:' + str(top_n_lst))


def fmt_x(x):
    x_replace = str(x).replace('<NUMBER>', 'NUM')
    output_x = '<s> ' + x_replace + '</s>'
    return output_x

df_one_train = pd.read_csv(os.path.join(datasets_path, 'one_train.csv'), usecols = ["shaped_code"])
df_one_test = pd.read_csv(os.path.join(datasets_path, 'one_test.csv'), usecols = ["shaped_code"])
df_zero_train = pd.read_csv(os.path.join(datasets_path, 'zero_train.csv'), usecols = ["shaped_code"])
df_zero_test = pd.read_csv(os.path.join(datasets_path, 'zero_test.csv'), usecols = ["shaped_code"])

df_Y_one_train = pd.read_csv(os.path.join(datasets_path, 'one_train.csv'), usecols = ["label"])
df_Y_one_test = pd.read_csv(os.path.join(datasets_path, 'one_test.csv'), usecols = ["label"])
df_Y_zero_train = pd.read_csv(os.path.join(datasets_path, 'zero_train.csv'), usecols = ["label"])
df_Y_zero_test = pd.read_csv(os.path.join(datasets_path, 'zero_test.csv'), usecols = ["label"])

df_one_test_rq2 = pd.read_csv(os.path.join(datasets_path, 'one_test.csv'))
df_zero_test_rq2 = pd.read_csv(os.path.join(datasets_path, 'zero_test.csv'))

df_train = pd.concat([df_one_train, df_zero_train], axis=0, ignore_index=True)
df_test = pd.concat([df_one_test, df_zero_test], axis=0, ignore_index=True)
df_Y_train = pd.concat([df_Y_one_train, df_Y_zero_train], axis=0, ignore_index=True)
df_Y_test = pd.concat([df_Y_one_test, df_Y_zero_test], axis=0, ignore_index=True)

# df_test_rq2 =  pd.concat([df_one_test_rq2, df_zero_test_rq2], axis=0, ignore_index=True)

x_train = df_train.to_numpy().reshape(-1)
# 推論用に変更
x_test = df_one_test.to_numpy().reshape(-1)
fmt_x_all = np.vectorize(fmt_x)
x_train = fmt_x_all(x_train)
x_test = fmt_x_all(x_test)
y_train = list(itertools.chain.from_iterable(df_Y_train.values.tolist()))
y_test = list(itertools.chain.from_iterable(df_Y_test.values.tolist()))

print('Loaded')

headers = ["label", "status", "file_dir", "change_id", "change_num", "patch_set", "shaped_code", "before_code", "after_code", "line", "ref"]
os.mkdir(os.path.join(RQ2_path, "positive_1"))
os.mkdir(os.path.join(RQ2_path, "negative_1"))
os.mkdir(os.path.join(RQ2_path, "positive_3"))
os.mkdir(os.path.join(RQ2_path, "negative_3"))
os.mkdir(os.path.join(RQ2_path, "positive_5"))
os.mkdir(os.path.join(RQ2_path, "negative_5"))
os.mkdir(os.path.join(RQ2_path, "positive_30per"))
os.mkdir(os.path.join(RQ2_path, "negative_30per"))
os.mkdir(os.path.join(RQ2_path, "unknown"))
with open(os.path.join(RQ2_path, "positive_1.csv"), "w") as p_tmp:
    p_tmp = csv.writer(p_tmp)
    p_tmp.writerow(headers)
with open(os.path.join(RQ2_path, "negative_1.csv"), "w") as n_tmp:
    n_tmp = csv.writer(n_tmp)
    n_tmp.writerow(headers)
with open(os.path.join(RQ2_path, "positive_3.csv"), "w") as p_tmp:
    p_tmp = csv.writer(p_tmp)
    p_tmp.writerow(headers)
with open(os.path.join(RQ2_path, "negative_3.csv"), "w") as n_tmp:
    n_tmp = csv.writer(n_tmp)
    n_tmp.writerow(headers)
with open(os.path.join(RQ2_path, "positive_5.csv"), "w") as p_tmp:
    p_tmp = csv.writer(p_tmp)
    p_tmp.writerow(headers)
with open(os.path.join(RQ2_path, "negative_5.csv"), "w") as n_tmp:
    n_tmp = csv.writer(n_tmp)
    n_tmp.writerow(headers)

test_toks = []
for sent in x_test:
    tok = TOKENIZER.encode_plus(sent,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True)
    test_toks.append(tok)

test_dataset = dataloader.InlineChangeDataset_for_token(test_toks, x_test, y_test)

model = codebert_design.CodeBert(MODEL_TYPE, TOKENIZER)
model.to(device)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer = AdamW(model.parameters(), lr=LEAENING_RATE)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

pos_num_1 = 1
neg_num_1 = 1
pos_num_3 = 1
neg_num_3 = 1
pos_num_5 = 1
neg_num_5 = 1
pos_num_30per = 1
neg_num_30per = 1
unk_num = 1

changed_token_lst_counter = []
codes_token_counter = []
top_1_token_counter = []
top_3_token_counter = []
top_5_token_counter = []
top_30per_token_counter = []
top_1_common_counter = []
top_3_common_counter = []
top_5_common_counter = []
top_30per_common_counter = []

ahi_counter = 0
for i, d in enumerate(test_dataset):
    input_ids = d["input_ids"].to(device).unsqueeze(0)
    attention_mask = d["attention_mask"].to(device).unsqueeze(0)
    target = d["target"].to(device)
    shaped_code = d['shaped_code']
    codes_token_counter.append(len(shaped_code.split(' '))-2)
    with torch.no_grad():
        output, _ = model(input_ids, attention_mask,)
    codes_token_counter.append(len(shaped_code.split(' '))-2)


    y_pred_row = (output.sigmoid().cpu()).item()
    y_pred = 1 if y_pred_row > 0.5 else 0
    print(f"y_test : {y_test[i]}\ny_pred : {y_pred}")
    ahi_counter += 1
    target_row = df_one_test_rq2.loc[i, :]


    if int(y_test[i]) == 1:
        exp = explainer.explain_instance(
                shaped_code,
                predictor,
                num_features=20,
                num_samples=10000)

        token_coefficients = exp.as_list()
        token_coefficients_dict = dict(token_coefficients)


        before_code = target_row['before_code'].replace('\\n', '\n')
        after_code = target_row['after_code'].replace('\\n', '\n')

        try:
            diff_list = diff_token(before_code, after_code)
            omit_special_char = [replace_with_number(i) for i in diff_list]
            changed_token_lst = list(filter(None, omit_special_char))
            acttually_changed_token_lst = list(set(changed_token_lst))

            top_1_lst = get_top_n_keys(token_coefficients_dict, 1)
            top_3_lst = get_top_n_keys(token_coefficients_dict, 3)
            top_5_lst = get_top_n_keys(token_coefficients_dict, 5)

            top_1_token_counter.append(len(top_1_lst))
            top_3_token_counter.append(len(top_3_lst))
            top_5_token_counter.append(len(top_5_lst))

            if y_pred == 1 and len(shaped_code.split(' '))-2 >= more_than_n:
                top_1_common_counter.append(len(find_partial_matches(top_1_lst, acttually_changed_token_lst)))
                top_3_common_counter.append(len(find_partial_matches(top_3_lst, acttually_changed_token_lst)))
                top_5_common_counter.append(len(find_partial_matches(top_5_lst, acttually_changed_token_lst)))

                changed_token_lst_counter.append(len(acttually_changed_token_lst))
                if (common_parts := find_partial_matches(top_1_lst, acttually_changed_token_lst)):
                    with open(os.path.join(RQ2_path, "positive_1.csv"), "a") as changed:
                        writer = csv.writer(changed)
                        writer.writerow(target_row)
                    with open(os.path.join(RQ2_path, "positive_1", f"logs_{pos_num_1}.txt"), "w")as logs:
                        logging_result(before_code, after_code, acttually_changed_token_lst, top_1_lst)
                    pos_num_1 += 1
                elif not (common_parts := find_partial_matches(top_1_lst, acttually_changed_token_lst)):
                    with open(os.path.join(RQ2_path, "negative_1.csv"), "a") as no_changed:
                        writer = csv.writer(no_changed)
                        writer.writerow(target_row)
                    with open(os.path.join(RQ2_path, "negative_1", f"logs_{neg_num_1}.txt"), "w")as logs:
                        logging_result(before_code, after_code, acttually_changed_token_lst, top_1_lst)
                    neg_num_1 += 1
                if (common_parts := find_partial_matches(top_3_lst, acttually_changed_token_lst)):
                    with open(os.path.join(RQ2_path, "positive_3.csv"), "a") as changed:
                        writer = csv.writer(changed)
                        writer.writerow(target_row)
                    with open(os.path.join(RQ2_path, "positive_3", f"logs_{pos_num_3}.txt"), "w")as logs:
                        logging_result(before_code, after_code, acttually_changed_token_lst, top_3_lst)
                    pos_num_3 += 1
                elif not (common_parts := find_partial_matches(top_3_lst, acttually_changed_token_lst)):
                    with open(os.path.join(RQ2_path, "negative_3.csv"), "a") as no_changed:
                        writer = csv.writer(no_changed)
                        writer.writerow(target_row)
                    with open(os.path.join(RQ2_path, "negative_3", f"logs_{neg_num_3}.txt"), "w")as logs:
                        logging_result(before_code, after_code, acttually_changed_token_lst, top_3_lst)
                    neg_num_3 += 1
                if (common_parts := find_partial_matches(top_5_lst, acttually_changed_token_lst)):
                    with open(os.path.join(RQ2_path, "positive_5.csv"), "a") as changed:
                        writer = csv.writer(changed)
                        writer.writerow(target_row)
                    with open(os.path.join(RQ2_path, "positive_5", f"logs_{pos_num_5}.txt"), "w")as logs:
                        logging_result(before_code, after_code, acttually_changed_token_lst, top_5_lst)
                    pos_num_5 += 1
                elif not (common_parts := find_partial_matches(top_5_lst, acttually_changed_token_lst)):
                    with open(os.path.join(RQ2_path, "negative_5.csv"), "a") as no_changed:
                        writer = csv.writer(no_changed)
                        writer.writerow(target_row)
                    with open(os.path.join(RQ2_path, "negative_5", f"logs_{neg_num_5}.txt"), "w")as logs:
                        logging_result(before_code, after_code, acttually_changed_token_lst, top_5_lst)
                    neg_num_5 += 1
            else:
                with open(os.path.join(RQ2_path, "FN.csv"), "a") as ahi:
                    writer = csv.writer(ahi)
                    writer.writerow(target_row)
        except Exception as e:
            print(e)
            with open(os.path.join(RQ2_path, "unknown.csv"), "a") as unknown:
                writer = csv.writer(unknown)
                writer.writerow(target_row)
            with open(os.path.join(RQ2_path, "unknown", f"logs_{unk_num}.txt"), "w")as logs:
                    logs.write(before_code)
                    logs.write("\n")
                    logs.write(after_code)
            unk_num += 1
    else:
        pass

def venn_make(changed_token_lst_counter, top_n_common_counter, top_n_token_counter, n):
    top_n_venn_left = sum(changed_token_lst_counter) - sum(top_n_common_counter)
    top_n_venn_right = sum(top_n_token_counter) - sum(top_n_common_counter)
    top_n_venn_common = sum(top_n_common_counter)
    plt.rcParams['figure.subplot.right'] = 0.8
    venn = venn2(subsets = (top_n_venn_left, top_n_venn_right, top_n_venn_common), set_labels = ('Token actually changed', 'Token expected to change'))
    recall_n = sum(top_n_common_counter) / sum(changed_token_lst_counter)
    precision_n = sum(top_n_common_counter) / sum(top_n_token_counter)
    with open(os.path.join(RQ2_path, f'top_{n}_score.txt'), 'w') as f:
        print(f'top_{n}_recall : {recall_n:.3f}', file=f)
        print(f'top_{n}_precision : {precision_n:.3f}', file=f)
        print(f'top_{n}_F1_Score : {F1_score(recall_n, precision_n):.3f}', file=f)
    plt.savefig(os.path.join(RQ2_path, f'top_{n}_venn.pdf'))
    plt.close()

venn_make(changed_token_lst_counter, top_1_common_counter, top_1_token_counter, 1)
venn_make(changed_token_lst_counter, top_3_common_counter, top_3_token_counter, 3)
venn_make(changed_token_lst_counter, top_5_common_counter, top_5_token_counter, 5)

print("Token per line")
print(f"mean: {statistics.mean(codes_token_counter)}")
print(f"median: {statistics.median(codes_token_counter)}")
print(f"mode: {statistics.mode(codes_token_counter)}")
print(f"pvariance: {statistics.pvariance(codes_token_counter)}")

print("changed_token per line")
print(f"mean: {statistics.mean(changed_token_lst_counter)}")
print(f"median: {statistics.median(changed_token_lst_counter)}")
print(f"mode: {statistics.mode(changed_token_lst_counter)}")
print(f"pvariance: {statistics.pvariance(changed_token_lst_counter)}")
sns.violinplot(data=[codes_token_counter, changed_token_lst_counter])
plt.savefig(os.path.join(RQ2_path, 'changed_token_perline.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.violinplot(data=[codes_token_counter, changed_token_lst_counter])
plt.ylabel('Token', fontsize=24)  # y軸のラベル
plt.xticks([0, 1], ['Token per line', 'Changed Token per line'], fontsize=24)  # x軸のティックラベル

plt.savefig(os.path.join(RQ2_path, 'violin_plot.pdf'))

count = sum(1 for i in codes_token_counter if i >= 5)
