import pandas as pd
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv('./track_diff_linelevel/OpenStack_ex.csv')

df_zero = df[df['label'] == 0]

df_one = df[df['label'] == 1]

dir = '../train_and_test/'
os.mkdir(dir)
df_zero_train, df_zero_test = train_test_split(df_zero, test_size=0.1, random_state=1000)
df_one_train, df_one_test = train_test_split(df_one, test_size=0.1, random_state=1000)

headers = ["label", "status", "file_dir", "change_id", "change_num", "patch_set", "shaped_code", "before_code", "after_code", "line", "ref"]
file_list = ['zero_test.csv', 'zero_train.csv', 'one_train.csv', 'one_test.csv', ]

df = pd.DataFrame(columns=headers)
for file_name in file_list:
    df.to_csv(os.path.join(dir, file_name), index=False)

df_zero_train.to_csv(os.path.join(dir, 'zero_train.csv'), index=False)
df_zero_test.to_csv(os.path.join(dir, 'zero_test.csv'), index=False)
df_one_train.to_csv(os.path.join(dir, 'one_train.csv'), index=False)
df_one_test.to_csv(os.path.join(dir, 'one_test.csv'), index=False)
