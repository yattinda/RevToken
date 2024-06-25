import pandas as pd
import re
import torch
from transformers import AdamW
import numpy as np
from matplotlib import pyplot as plt
import string
import os
import sys
import itertools
import seaborn as sns
import xmltodict
import csv
import statistics
from matplotlib_venn import venn2
sys.path.append(os.path.join(os.pardir, os.pardir))
from train import codebert_design, dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Select CUDA')
epoch = 19
MODEL_TYPE = codebert_design.MODEL_TYPE
TOKENIZER = codebert_design.TOKENIZER
LEAENING_RATE = codebert_design.LEAENING_RATE
datasets_path = '../../datasets/train_and_test/'
more_than_n = 7
os.mkdir(RQ2_path := f"../attention/result_more_{more_than_n}")
checkpoint = torch.load(f'../../model/checkpoint/checkpoint_{epoch}.pth', map_location=device)

def top_10_percent_value(lst):
    lst.sort(reverse=True)
    index = int(len(lst) * 0.1)
    return lst[index]

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
    for key, value in dictionary.items():
        new_key = key.replace('Ġ', '')
        modify_dict[new_key] = value
    sorted_items = sorted(modify_dict.items(), key=lambda x: x[1], reverse=True)
    top_n_keys = [item[0] for item in sorted_items[:n]]
    return top_n_keys

def transform_list(tokens, attention):
    token_key = []
    attention_values = []
    for i, item in enumerate(tokens):
        if any([item.startswith('Ġ'), item == '<s>', item == '</s>']):
            token_key.append(item)
            attention_values.append(attention[i])
        else:
            token_key[-1] += item
            attention_values[-1] = max(attention_values[-1], attention[i])
    return token_key, attention_values


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
    x_replace = x.replace('<NUMBER>', 'NUM')
    output_x = '<s> ' + x_replace + '</s>'
    return output_x

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def is_contains_special_characters(s):
    try:
        if any(char in s for char in string.punctuation):
            pass
        else:
            return s
    except TypeError:
        return s

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
    # token_type_ids = d["token_type_ids"].to(device).unsqueeze(0)
    target = d["target"].to(device)
    shaped_code = d['shaped_code']
    codes_token_counter.append(len(shaped_code.split(' '))-2)
    with torch.no_grad():
        output, attention = model(input_ids, attention_mask,)

    attention = attention.cpu()[0].numpy()
    attention_mask = attention_mask.cpu()[0].numpy()
    attention = attention[attention_mask == 1][1:-1]

    ids = input_ids.cpu()[0][attention_mask == 1][1:-1].tolist()
    tokens = TOKENIZER.convert_ids_to_tokens(ids)
    attention = attention.tolist()
    tokens, attention = transform_list(tokens, attention)
    token_with_attention = dict(zip(tokens, attention))
    y_pred_row = (output.sigmoid().cpu()).item()
    y_pred = 1 if y_pred_row > 0.5 else 0
    print(f"y_test : {y_test[i]}\ny_pred : {y_pred}")
    ahi_counter += 1
    target_row = df_one_test_rq2.loc[i, :]


    if int(y_test[i]) == 1:
        before_code = target_row['before_code'].replace('\\n', '\n')
        after_code = target_row['after_code'].replace('\\n', '\n')
        tmp_xml = "output.xml"
        with open("../srcdiff-ubuntu/before_code.cpp", 'w') as file:
            file.write(before_code)
        with open("../srcdiff-ubuntu/after_code.cpp", 'w') as file:
            file.write(after_code)
        os.chmod("../srcdiff-ubuntu/before_code.cpp", 0o777)
        os.chmod("../srcdiff-ubuntu/after_code.cpp", 0o777)
        os.system(f'../srcdiff-ubuntu/srcdiff ../srcdiff-ubuntu/before_code.cpp ../srcdiff-ubuntu/after_code.cpp -o {tmp_xml}')
        diff_contents = []

        try:
            with open('output.xml', encoding='utf-8') as fp:
            # xml読み込み
                xml_data = fp.read()
            # xml → dict
                dict_data = xmltodict.parse(xml_data)

            delete_name = dict_search(remove_key(dict_search(dict_data, "diff:delete"), "diff:common"), "name")
            delete_text = dict_search(remove_key(dict_search(dict_data, "diff:delete"), "diff:common"), "#text")
            delete_name.append(delete_text)
            lst = [item for item in delete_name if not isinstance(item, dict)]
            flat_lst = flatten_list(lst)
            omit_special_char = [replace_with_number(is_contains_special_characters(i)) for i in flat_lst]
            changed_token_lst = list(filter(None, omit_special_char))
            acttually_changed_token_lst = list(set(changed_token_lst))

            top_1_lst = get_top_n_keys(token_with_attention, 1)
            top_3_lst = get_top_n_keys(token_with_attention, 3)
            top_5_lst = get_top_n_keys(token_with_attention, 5)

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
print(codes_token_counter)
print(f'Token 10%: {top_10_percent_value(codes_token_counter)}')

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
plt.ylabel('Token', fontsize=24)
plt.xticks([0, 1], ['Token per line', 'Changed Token per line'], fontsize=24)

plt.savefig(os.path.join(RQ2_path, 'violin_plot.pdf'))

count = sum(1 for i in codes_token_counter if i >= 5)
