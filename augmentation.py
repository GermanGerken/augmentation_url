import json

import numpy as np
import pandas as pd
import glob
import csv
from predict.storage import create_container_not_exists

# Hyper parameters

'''
--| `time_ignorance` metric uses in two cases:
-first case  : We dont know how much time user spend on last page before quiting the site
-second case : User has visited only one page and quit, so we dont know how much time he spent on

--| `break_point` - if `time_on_page`(sec) is more than `break_point` means new session start
'''


# !!!
# "_>>_"
# "=>"

# --- args
# input_loc
# output_loc

# to UDF file

def TimeOnPage(x, time_ignorance=5):
    arr = np.array(x)
    on_page = arr[1:] - arr[0:-1]
    result = np.append(on_page, time_ignorance)
    return result


def SessMask(x, break_point=1800):
    return np.where(x > break_point, 1, 0)


def SessNum(x):
    # Not optimized
    acc = []
    sess_num = 0

    if x[0] == 1:
        acc.append(sess_num)
        for i in x[1:]:
            if i == 0:
                acc.append(sess_num)
            else:
                sess_num += 1
                acc.append(sess_num)
    else:
        for i in x:
            if i == 0:
                acc.append(sess_num)
            else:
                sess_num += 1
                acc.append(sess_num)
    return acc


def ChainMask(x):
    if len(x) == 1:
        return [False]

    else:
        arr = np.array(x)
        mask0 = arr[0:-1] == arr[1:]
        mask1 = np.insert(mask0, 0, False)
        return mask1


def ToCategorical(val, q_dict, prefix):
    keys = q_dict.keys()
    for i in keys:
        if val <= q_dict[i]:
            return prefix + i
        else:
            pass
    return prefix + "100%"


def augmentation(**kwargs):
    input_detail_path = glob.glob(kwargs["input_detail_path"])
    output_path = kwargs["output_path"]
    time_ignorance = kwargs["time_ignorance"]
    break_point = kwargs["break_point"]
    data = []

    for i, e in enumerate(input_detail_path):
        try:
            data.append(pd.read_csv(input_detail_path[i], dtype=str))
        except:
            pass
    data = pd.concat(data)
    data['time_seq'] = data['timeline'].apply(lambda x: [int(i) for i in x.split("=>")])
    data['time_on_page'] = data['time_seq'].apply(lambda x: TimeOnPage(x))
    data['session_mask'] = data['time_on_page'].apply(lambda x: SessMask(x))
    data['time_on_pageU'] = data['time_on_page'].apply(lambda x: list(np.where(x > 1800, time_ignorance, x)))
    data['sess_num'] = data['session_mask'].apply(lambda x: SessNum(x))
    data['path_seq'] = data['user_path'].apply(lambda x: [i.split("_>>_")[0] for i in x.split("=>")])
    data['path_seq_ed'] = data['user_path'].apply(lambda x: [i.split("_>>_")[0] for i in x.split("=>")])
    data = data.explode('path_seq')
    data_dict = data[['path_seq', 'sess_num', 'time_on_pageU']]

    data_dict = data_dict.explode('sess_num')
    data_dict = data_dict.explode('time_on_pageU')

    uniq_seq = list(data_dict["path_seq"].unique())
    appended_data = []
    quantile_data = []
    cols = ["url", 'sess_quantile', 'sess_dec', 'time_on_page_quantile', 'top_dec']

    for seq in uniq_seq:
        sess_dict = pd.to_numeric(data_dict.groupby('path_seq')["sess_num"].get_group(seq)).describe()[['25%', '50%', '75%']].to_dict()
        top_dict = pd.to_numeric(data_dict.groupby('path_seq')["time_on_pageU"].get_group(seq)).describe()[['25%', '50%', '75%']].to_dict()
        data_url = data.loc[data['path_seq'] == seq]
        sess_quant = pd.DataFrame(sess_dict.items(), columns=['sess_quantile', 'sess_dec'])
        top_quant = pd.DataFrame(top_dict.items(), columns=['time_on_page_quantile', 'top_dec'])
        quant = pd.merge(sess_quant, top_quant, how="left", left_on=['sess_quantile'], right_on=['time_on_page_quantile'])
        quant["url"] = seq
        data_url['sess_descrete'] = data_url['sess_num'].apply(lambda x: [ToCategorical(i, sess_dict, "sess") for i in x])
        data_url['ToP_descrete'] = data_url['time_on_pageU'].apply(lambda x: [ToCategorical(i, top_dict, "pageT") for i in x])
        quantile_data.append(quant[cols])
        appended_data.append(data_url)
    appended_data = pd.concat(appended_data)
    quantile_data = pd.concat(quantile_data)
    appended_data['generator'] = appended_data.apply(lambda x: zip(x['path_seq_ed'], x['sess_descrete'], x['ToP_descrete']), axis=1)
    appended_data['new_channel'] = appended_data['generator'].apply(lambda x: ["_>>_".join([i, str(k), str(l)]) for (i, k, l) in x])
    appended_data['graph_path'] = appended_data['new_channel'].apply(lambda x: "=>".join(x))

    data = appended_data[['ClientID', 'graph_path']].drop_duplicates()
    create_container_not_exists(output_path)
    create_container_not_exists(f'{output_path}/quant')
    with open(f'{output_path}output.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ClientID', 'graph_path'])
    quantile_data.to_csv(f'{output_path}quant/quantile_output_tutu.csv', index=False)
    data.to_csv(f'{output_path}output.csv', mode='a', index=False, header=False)
