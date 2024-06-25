import csv
from dotenv import load_dotenv
load_dotenv()
import os
import time
import requests
from urllib.parse import quote
from enum import Enum
from pygerrit2 import GerritRestAPI, HTTPBasicAuth

output_path = "track_diff_linelevel666"
linelevel_path = 'linelevel_raw/openstack_nova.csv'

class CSVCol(Enum):
    LABEL = 0
    FILEDIR = 1
    CHANGEID = 2
    CODE = 3
    RAWCODE = 4
    PROJECT = 5
    CREATE = 6
    DELETIONS = 7
    ADDITIONS = 8
    CHANGED_LINE = 9
    LINE = 10
    REF = 11

# status:
#     0:インラインコメントなし
#     1:インラインコメントがついて変更された
#     2:インラインコメントがついたが変更されない


def find_index(lst, str):
    indices = [i for i, s in enumerate(lst) if str in s]
    return indices

start = 1
end = 700000
big_step = 10000
small_step = 200

# 範囲を作成
big_ranges = [(i, i+big_step-1) for i in range(start, end, big_step)]
small_ranges = [(i, i+small_step-1) for i in range(start, end, small_step)]


gerrit_url = 'https://review.opendev.org/'
#Loading from .env file
auth = HTTPBasicAuth(os.getenv('USER_NAME'), os.getenv('PASSWORD'))
rest = GerritRestAPI(url=gerrit_url, auth=auth)

os.mkdir(output_path)
caution_tmp = open(os.path.join(output_path, 'caution.csv'), 'w')
caution_tmp.close()
notfound_tmp = open(os.path.join(output_path, 'not_found.csv'), 'w')
notfound_tmp.close()

headers = ["label", "status", "file_dir", "change_id", "change_num", "patch_set", "shaped_code", "before_code", "after_code", "line", "ref"]
with open(linelevel_path, 'r') as tmp:
    with open(os.path.join(output_path, "OpenStack_ex.csv"), 'a', newline='') as tmp2:
        reader = csv.reader(tmp)
        next(reader)
        writer = csv.writer(tmp2)
        writer.writerow(headers)
        for row in reader:
            parts = row[CSVCol.REF.value].split("/")
            change_num = int(parts[3])
            patch_num = int(parts[4])
            if row[CSVCol.LABEL.value] == "0":
                row_tmp = [0, 0, row[CSVCol.FILEDIR.value], row[CSVCol.CHANGEID.value], change_num, patch_num, row[CSVCol.CODE.value], row[CSVCol.RAWCODE.value], None, row[CSVCol.LINE.value], row[CSVCol.REF.value]]
                writer.writerow(row_tmp)
            else:
                encoded = quote(row[CSVCol.FILEDIR.value], safe='')
                try:
                    diff = rest.get("/changes/{}/revisions/{}/files/{}/diff?base={}".format(row[CSVCol.CHANGEID.value], patch_num + 1, encoded, patch_num))
                    time.sleep(2)
                except requests.exceptions.HTTPError as err:
                    print("retry 1")
                    try:
                        time.sleep(2)
                        diff = rest.get("/changes/{}/revisions/{}/files/{}/diff?base={}".format(row[CSVCol.CHANGEID.value], patch_num + 2, encoded, patch_num))
                        time.sleep(2)
                    except requests.exceptions.HTTPError as err:
                        print("retry 2")
                        try:
                            time.sleep(2)
                            diff = rest.get("/changes/{}/revisions/{}/files/{}/diff?base={}".format(row[CSVCol.CHANGEID.value], patch_num + 3, encoded, patch_num))
                            time.sleep(2)
                        except:
                            print("retry 3")
                            with open(os.path.join(output_path,"log.txt"), "a") as log:
                                log.write("/changes/{}/revisions/{}/files/{}/diff?base={}\n".format(row[CSVCol.CHANGEID.value], patch_num + 1, encoded, patch_num))


                element_counter = 0
                ab_flag = False
                a_flag = False
                target_raw_code = str(row[CSVCol.RAWCODE.value].lstrip('+'))
                for data in diff["content"]:
                    for key, val in data.items():
                        if not isinstance(val, list):
                            pass
                        elif target_raw_code in (val := ''.join(val)):
                            if key == "ab" or key == "b":
                                label_tmp = 0
                                status = 2
                                b_code = None
                                row_tmp = [label_tmp, status, row[CSVCol.FILEDIR.value], row[CSVCol.CHANGEID.value], change_num, patch_num, row[CSVCol.CODE.value], row[CSVCol.RAWCODE.value], b_code, row[CSVCol.LINE.value], row[CSVCol.REF.value]]
                                writer.writerow(row_tmp)
                                ab_flag = True
                                with open(os.path.join(output_path,"status.txt"), "a") as status_log:
                                        print(("Status == 2"))
                                        status_log.write("Status == 2\n")
                            elif key == "a":
                                label_tmp = 1
                                status = 1
                                try:
                                    num = find_index(diff["content"][element_counter]["a"], target_raw_code)
                                    a_code = diff["content"][element_counter]["a"][num[0]]

                                    try:
                                        b_code_before = diff["content"][element_counter]["b"][num[0]-1]
                                    except:
                                        b_code_before = ""
                                    try:
                                        b_code_target = diff["content"][element_counter]["b"][num[0]]
                                    except KeyError:
                                        b_code_targets = ""
                                    try:
                                        b_code_after = diff["content"][element_counter]["b"][num[0]+1]
                                    except:
                                        b_code_after = ""
                                    b_code = b_code_before + r"\n" + b_code_target + r"\n" + b_code_after

                                    with open(os.path.join(output_path,"status.txt"), "a") as status_log:
                                        print("Status == 1")
                                        status_log.write("Status == 1\n")
                                except IndexError:
                                    with open(os.path.join(output_path, 'not_found.csv'), 'a') as tmp3:
                                        caution = csv.writer(tmp3)
                                        err_tmp = [row[CSVCol.FILEDIR.value], row[CSVCol.CHANGEID.value], change_num, patch_num, row[CSVCol.CODE.value], row[CSVCol.RAWCODE.value], row[CSVCol.LINE.value], row[CSVCol.REF.value]]
                                        caution.writerow(err_tmp)
                                        with open(os.path.join(output_path,"status.txt"), "a") as status_log:
                                            print("Status == 1 but err")
                                            status_log.write("Status == 1 but err\n")
                                    break
                                row_tmp = [label_tmp, status, row[CSVCol.FILEDIR.value], row[CSVCol.CHANGEID.value], change_num, patch_num, row[CSVCol.CODE.value], a_code, b_code, row[CSVCol.LINE.value], row[CSVCol.REF.value]]
                                writer.writerow(row_tmp)
                                a_flag = True
                            else:
                                with open(os.path.join(output_path, 'caution.csv'), 'a') as tmp3:
                                    caution = csv.writer(tmp3)
                                    err_tmp = [row[CSVCol.FILEDIR.value], row[CSVCol.CHANGEID.value], change_num, patch_num, row[CSVCol.CODE.value], row[CSVCol.RAWCODE.value], b_code, row[CSVCol.LINE.value], row[CSVCol.REF.value]]
                                    caution.writerow(err_tmp)
                                    with open(os.path.join(output_path,"status.txt"), "a") as status_log:
                                        print("err")
                                        status_log.write("err\n")
                                pass
                    element_counter += 1
                if(ab_flag & a_flag):
                    with open(os.path.join(output_path, 'double.csv'), 'a') as tmp3:
                        caution = csv.writer(tmp3)
                        err_tmp = [row[CSVCol.FILEDIR.value], row[CSVCol.CHANGEID.value], change_num, patch_num, row[CSVCol.CODE.value], row[CSVCol.RAWCODE.value], b_code, row[CSVCol.LINE.value], row[CSVCol.REF.value]]
                        caution.writerow(err_tmp)
