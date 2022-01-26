import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
import pyperclip
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import plotly.express as px
import re

parser = argparse.ArgumentParser()
parser.add_argument("--last_ep", type=int, default=10,
                    help="")
# parser.add_argument("--inputopt",  action='store_true',
#                     help="choose whether or not you want to input the name of the option. Default:False, place --inputopt if you want it to be True.")
parser.add_argument("--allepoch",  action='store_true',
                    help="choose whether or not you want to evaluate all epoch's data. Default:False, place --allepoch if you want it to be True.")
parser.add_argument("--norealdata",  action='store_false',
                    help="choose whether or not you want to. Default:True, place --norealdata if you want it to be False.")
opt = parser.parse_args()


driver = webdriver.Chrome()
driver.get('http://codes.bio/meta-iavp/')
search_box = driver.find_element_by_id("Sequence")
predave_file = "preds_ave.txt"
pred_file = "preds.txt"
seqpred_file = "seq_preds.txt"
sorted_seqpred_file = "sorted_seq_preds.txt"
cred_file = "credibility.txt"
cred_threshold = 0.5
pred_img = "preds.png"
val_pos_file = 'val_positive'
val_neg_file = 'val_negative'
query = "ep(.*)_ut"
options = ["ep", "ut", "fe"]
# options = ["ep", "ba", "lr", "pc", "opt", "itr"]
file_list = []
check_realdata = opt.norealdata


def get_preds_from_web(path):
    with open(path) as f:
        search_box.clear()
        js_cmd = "document.getElementById('contents').innerHTML=''"
        driver.execute_script(js_cmd)

        all_seqs = f.read()
        time.sleep(1)
        pyperclip.copy(all_seqs)

        search_box.send_keys(Keys.SHIFT, Keys.INSERT)
        driver.find_element_by_id('submitbutton').click()

        try:
            element = WebDriverWait(driver, 3).until(
                EC.text_to_be_present_in_element((By.ID, "contents"), 'Prediction'))
            print(path, ' :successfull!!!')
        except:
            print(path, ' :unsuccessfull...')
            return None
        alltext = driver.find_element_by_id("contents").text

    return alltext


def write_results(preds, fasta_path, run_dir, seqpred_file):
    seqpred_dic = {}
    i = 0
    header = 'Synthetic Sequence\tProbability\t\n'

    with open(fasta_path) as f:
        for line in f:
            if not ('>') in line:
                # print(seqpred_dic)
                seqpred_dic[line[:-1]] = preds[i]
                i += 1

    if not os.path.exists(eval_dir + run_dir):
        os.mkdir(eval_dir + run_dir)

    with open(eval_dir+run_dir+'/'+seqpred_file, 'w') as f:
        f.write(header)
        for s, p in seqpred_dic.items():
            f.write("\t".join([s, str(p), '\n']))

    with open(eval_dir+run_dir+'/'+sorted_seqpred_file, 'w') as f:
        f.write(header)
        sorted_seqpred_list = sorted(
            seqpred_dic.items(), key=lambda x: x[1], reverse=True)
        for s, p in sorted_seqpred_list:
            f.write("\t".join([s, str(p), '\n']))

    with open(eval_dir+run_dir+'/'+cred_file, 'w') as f:
        cred_seq_lis = [k for k, v in sorted_seqpred_list
                        if v >= cred_threshold]
        cred = len(cred_seq_lis)/len(sorted_seqpred_list)*100
        print('Credibility: ', cred)
        f.write(''.join(['Credibility: ', str(cred), '%']))


def webscraping(file_list, eval_dir, realdata=False):
    for dir_name, run_dir in file_list:
        preds_lis = []
        i = 1

        while True:
            if not realdata:
                if opt.allepoch:
                    fasta_path = dir_name+run_dir+str(i)+'.fasta'
                else:
                    gen_file = re.findall(query, run_dir)[0]
                    # print(gen_file)
                    fasta_path = dir_name+run_dir+gen_file+'.fasta'
            else:
                fasta_path = dir_name+run_dir+'.fasta'

            if os.path.exists(fasta_path):
                alltext = get_preds_from_web(fasta_path)

                if alltext:
                    preds_ave, preds = calc_preds_ave(alltext, i)
                    write_results(preds, fasta_path,
                                  run_dir, seqpred_file)
                    preds_lis.append(preds_ave)

                else:
                    i += 1
                    continue

                if realdata or not opt.allepoch:
                    makeplot_from_intlis(preds_lis, eval_dir, run_dir+'/')
                    break

                i += 1

            else:
                break

        if not realdata:
            makeplot_from_intlis(preds_lis, eval_dir, run_dir)


def calc_preds_ave(alltext, i):
    text = alltext.split()
    del text[0:3]
    preds = []

    for i, val in enumerate(text):
        if i % 4 == 2:
            preds.append(float(val))

    preds_ave = sum(preds)/len(preds)

    return preds_ave, preds


def makeplot_from_intlis(intlis, eval_dir, run_dir):
    """
    INPUT
    intlis: list(int)
    """

    if not os.path.exists(eval_dir + run_dir):
        os.mkdir(eval_dir + run_dir)

    with open(eval_dir + run_dir + predave_file, "w") as f:
        str_intlis = [str(num) for num in intlis]
        f.write("\n".join(str_intlis))

    plt.figure()
    plt.plot([i for i in range(1, len(intlis)+1)], intlis)
    plt.savefig(eval_dir + run_dir + pred_img)


def calc_last_pred(epochs, run_dirs, eval_dir, real_file_list):
    preds_dic = {}

    for run_dir in run_dirs:
        last_preds_list = []

        with open(eval_dir + run_dir + '/' + predave_file) as f:
            for line in f:
                last_preds_list.append(float(line[:-1]))
        tmp = last_preds_list[-epochs:]
        preds_dic[run_dir] = sum(tmp)/len(tmp)

    for sample_dir, realdata_dir in real_file_list:
        with open(eval_dir + realdata_dir + '/' + predave_file) as f:
            realdata_val = float(f.readline()[:-1])
        preds_dic[realdata_dir] = realdata_val
    sort_preds_dic = sorted(preds_dic.items(), key=lambda x: -x[1])

    with open(eval_dir + predave_file, "w") as f:
        header = [opt for opt in options] + ["Probability"]
        f.write("\t".join(header))
        for dir, pred in sort_preds_dic:
            if 'positive' in dir or 'negative_exp' in dir:
                f.write('\n')
                row = [dir.replace("_", "-")] + ["-" for _ in range(len(options)-1)] + \
                    [str(round(pred, 3))]
                f.write("\t".join(row))
            else:
                for op in options:
                    dir = dir.replace(op, "")
                dir = "\n" + dir
                row = dir.split("_") + [str(round(pred, 3))]
                f.write("\t".join(row))


def select_undone_dir(eval_dir, samples_dir):
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    pre_run_dirs = os.listdir(samples_dir)
    run_dirs = [f for f in pre_run_dirs if os.path.isdir(
        os.path.join(samples_dir, f))]
    existing_run_dirs = os.listdir(eval_dir)

    if not existing_run_dirs:
        new_run_dirs = run_dirs

    else:
        run_dirs_df = pd.DataFrame(run_dirs)
        existing_run_dirs_df = pd.DataFrame(existing_run_dirs)
        new_run_dirs_df = run_dirs_df[~run_dirs_df[0].isin(
            existing_run_dirs_df[0])]
        new_run_dirs = new_run_dirs_df[0].values.tolist()

    for run_dir in new_run_dirs:
        file_list.append([samples_dir, run_dir+"/"])

    return run_dirs, file_list


def run_eval(real_file_list, eval_dir, file_list, run_dirs):
    if check_realdata:
        webscraping(real_file_list, eval_dir, realdata=True)
    webscraping(file_list, eval_dir)
    calc_last_pred(opt.last_ep, run_dirs, eval_dir, real_file_list)


########################################## main ##########################################

samples_dir = "./samples_fasta/"
eval_dir = "./eval/"
real_fasta_dir = "./real_data_fasta/"
real_file_list = [[real_fasta_dir, val_pos_file],
                  [real_fasta_dir, val_neg_file]]
run_dirs, file_list = select_undone_dir(eval_dir, samples_dir)
run_eval(real_file_list, eval_dir, file_list, run_dirs)
