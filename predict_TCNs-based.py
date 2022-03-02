##############################################################################
import numpy as np
import pickle
import urllib.parse
import urllib.request
import json
import logging
import argparse
import datetime
import schedule
import time
from typing import List, Dict, Any
import sys

import tensorflow as tf
import tensorflow_addons as tfa
import tcn
import keras
import socket
import datetime
import csv
##############################################################################
dt_format = '%Y-%m-%dT%H:%M:%SZ'
dt_format_srv = '%Y-%m-%dT%H:%M:%S.%fZ'

model_file = 'model.pickle'

username = 'Gc4TurKY@N9msHMRX'
password = 'AR33mDrfZyeME5uEWw299HVUS!'

auth_url = 'https://api.livingrobot-platform.com/v1/token'
get_url = 'https://api.livingrobot-platform.com/v1/sensors/data?{}'
post_url = 'https://api.livingrobot-platform.com/v1/sensors/results'
push_url = 'https://api.livingrobot-platform.com/v1/sensors/push'

log_format = '%(name)s : %(levelname)s : %(asctime)s : %(message)s'

is_invalid_token = True
token = ''
##############################################################################
def get_access_token(api_url: str, username: str, password: str) -> str:
    payload = {
        'grant_type': 'password',
        'username': username,
        'password': password,
    }
    data = urllib.parse.urlencode(payload).encode('utf-8')
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    req = urllib.request.Request(api_url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req) as res:
            auth_data = json.load(res)
            logger.debug('get auth token: {}'.format(json.dumps(auth_data, indent=2)))
            return auth_data['access_token']
    except urllib.error.URLError as err:
        logger.critical('auth failed: {}'.format(err.reason))
    except urllib.error.HTTPError as err:
        logger.error('auth failed: {} / {}'.format(err.code, err.reason))

    return ''
##############################################################################
def get_sensor_data(api_url: str, token: str, device_id: str, delay: int) -> Dict:
    now = datetime.datetime.utcnow()
#   from_date = datetime.datetime.strftime(now - datetime.timedelta(seconds=120 + delay), dt_format) #サーバレスポンス遅い対策 2021/11/11 maru
    from_date = datetime.datetime.strftime(now - datetime.timedelta(seconds=180 + delay), dt_format) #2分間分のデータとる 2021/11/11 maru
    to_date = datetime.datetime.strftime(now - datetime.timedelta(seconds=60 + delay), dt_format)

    query = {
        'from': from_date,
        'to': to_date,
        'sensor_device_id': device_id,
        'sensor_type': 'smell'
    }
    print(from_date)
    print(to_date)
    query_string = urllib.parse.urlencode(query)
    req = urllib.request.Request(api_url.format(query_string))
    req.add_header('Authorization', "Bearer {}".format(token))
    try:
        with urllib.request.urlopen(req,timeout=120) as res:
            sensor_data = json.load(res)
            logger.debug('get sensor data: {} / {} / {}'.format(from_date, to_date, json.dumps(sensor_data, indent=2)))
            return sensor_data
    except urllib.error.URLError as err:
        logger.critical('get failed: {}'.format(err.reason))
    except urllib.error.HTTPError as err:
        logger.error('get failed: {} / {}'.format(err.code, err.reason))
    except socket.timeout:
        logger.error('get failed:timeout')

    return {}
##############################################################################
def predict(sensor_data: Dict, model:Any) -> List[Dict]:
    results = []

    timeseries_num = 15
    sampling_rate = 0.8
    time_threshold = 15
    scale_factor = 100

    for sensor in sensor_data['sensors']:
        sensor_deviced_id = sensor['sensor_device_id']
        sensor_data = sensor['sensor_data']
        
        data_all = None
        value_threshold_switch = 0
        for i in range(len(sensor_data)): # convert the dict. into one numpy array
            if sensor_data[i]['sensor_type'] == 'smell': # determine whether it is smell data or not
                data_temp = sensor_data[i]['value']
                data_temp = list(map(int, data_temp.split(',')))
                
                if i < int(time_threshold/sampling_rate):
                    continue
                elif i >= int(time_threshold/sampling_rate):
                    pass

                if value_threshold_switch == 1:
                    pass
                elif value_threshold_switch == 0:
                    value_threshold_switch += 1
                    pass
                
                data_temp = np.array(data_temp)
                if data_all is None:
                    data_all = data_temp
                else:
                    data_all = np.vstack((data_all, data_temp))
            else:
                continue
        if data_all is None:
            logger.debug('empty smell data')
            continue
        
        ### fileへの書き出し ###################################################
        ### (学習データ用)   ###################################################

        if data_all.shape[0] >= timeseries_num:

            #標準フォーマットの配列に合わせる

            #列を追加(内容はすべて0)
            c1 = np.full(51,0)
            col_swap = np.insert(data_all, [10], c1, axis=1)

            #ヘッダ7行分と頭カット18行(15s)分を追加(内容はすべて0)
            c2 = np.full((25,61),0)
            col_swap = np.vstack((c2,col_swap))

            #列を入れ替えて標準センサーの素子位置に合わせる
            col_swap = col_swap[:,[43,10,22,28,4,55,40,52,16,19,1,11,12,13,14,15,8,17,18,9,20,21,2,23,24,25,26,27,3,29,30,31,32,33,34,35,36,37,38,39,6,41,42,0,44,45,46,47,48,49,50,51,7,53,54,5,56,57,58,59,60]]
            col_swap = col_swap[:,[0,4,2,3,1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]]

            #CSVファイルに出力
            now = datetime.datetime.now()
            filename = './output/log_' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
            np.savetxt( filename, col_swap, delimiter=',', fmt='%d')

        ########################################################################

        data_all = data_all/scale_factor # feature scaling for normalization
        data_all = data_all.reshape(-1,10)

        print('data_all.shape=',data_all.shape)
        if data_all.shape[0] < timeseries_num:
            logger.debug('The number of this data is less than that of the recognition network input.')
            continue

        x_timeseries = np.ones( (data_all.shape[0]-timeseries_num+1, timeseries_num, 10), dtype='int16')
        for j in range(data_all.shape[0]-timeseries_num+1):
            x_timeseries_temp = data_all[j:j+timeseries_num, :].reshape(1,timeseries_num,10)
            x_timeseries[j] = x_timeseries_temp
        data_all = x_timeseries
        
        #maru
        print('data_all.shape=',data_all.shape)
        print('data_all',data_all)

        data_pred = model.predict(data_all) # input to the mlp-based DL model

        #maru
        print(data_pred)

        data_pred = list(np.argmax(data_pred,axis=1))
        coffee_cnt = 0
        alcohol_cnt= 0
        for j in range(len(data_pred)): # ground truth : coffee -> 0, alcochol -> 1 / assume adding the classes in the future
            if data_pred[j] == 0:
                coffee_cnt += 1
            elif data_pred[j] == 1:
                alcohol_cnt += 1

        logger.debug('coffee  : {}'.format(coffee_cnt))
        logger.debug('alcohol : {}'.format(alcohol_cnt))

        if coffee_cnt >= alcohol_cnt:
            result_estimated = 0 #'coffee'
        else:
            result_estimated = 1 #'alcohol'
        
        report_time_last = sensor_data[-1]['report_time']
        time = datetime.datetime.strftime(datetime.datetime.strptime(report_time_last, 
                                                                     dt_format_srv), 
                                          dt_format)
        result = {
            'sensor_device_id': sensor_deviced_id,
            'excretion_date': time,
            'excretion': str(bool(result_estimated)).lower(),
            'next_excretion_forecast': time
            }
        logger.debug('predict result: {}'.format(json.dumps(result, indent=2)))
        results.append(result)

    return results
##############################################################################
def post_results(api_url: str, token: str, results: List) -> str:
    payload = {
        'results': results
    }
    print(results)
    data = json.dumps(payload).encode()
    headers = {
        'Content-Type': 'application/json'
    }

    req = urllib.request.Request(api_url, data=data, headers=headers)
    req.add_header('Authorization', "Bearer {}".format(token))
    try:
        with urllib.request.urlopen(req) as res:
            return res
    except urllib.error.URLError as err:
        logger.critical('post failed: {}'.format(err.reason))
    except urllib.error.HTTPError as err:
        logger.error('post failed: {} / {}'.format(err.code, err.reason))

    return ''
##############################################################################
def push_message(api_url: str, token: str, results: List) -> str:
    payload = {
        "message": "最新の解析結果が出力されました。",
        "push_sensors": [r['sensor_device_id'] for r in results]
    }
    data = json.dumps(payload).encode()
    headers = {
        'Content-Type': 'application/json'
    }
    req = urllib.request.Request(api_url, data=data, headers=headers)
    req.add_header('Authorization', "Bearer {}".format(token))
    try:
        with urllib.request.urlopen(req) as res:
            return res
    except urllib.error.URLError as err:
        logger.critical('push failed: {}'.format(err.reason))
    except urllib.error.HTTPError as err:
        logger.error('push failed: {} / {}'.format(err.code, err.reason))

    return ''
##############################################################################
def main(sensor_device_id: str, second: str, delay: int):
    #model = pickle.load(open(model_file, 'rb'))
    #model = keras.models.load_model('./model+weights_best.h5',
    #                                custom_objects={'LayerNormLSTMCell': tfa.rnn.LayerNormLSTMCell})
    model = keras.models.load_model('./model+weights_best.h5',
                                    custom_objects={'TCN': tcn.TCN})

    def job(model: Any, delay: int):
        global is_invalid_token
        global token

        if is_invalid_token:
            token = get_access_token(auth_url, username, password)
            if token:
                is_invalid_token = False

        sensor_data = get_sensor_data(get_url, token, sensor_device_id, delay)
        if not sensor_data:
            is_invalid_token = True
            return

        results = predict(sensor_data, model)
        if not results:
            logger.info('no sensor data')
            return

        res = post_results(post_url, token, results)
        if not res:
            is_invalid_token = True
            return
        if res.status == 201:
            logger.debug('post succeeded')
        else:
            logger.error('post result: {} / {}'.format(res.getcode(), json.loads(res.read())))
            is_invalid_token = True
            return

        #maru#
        logger.debug('excretion: {}'.format(json.dumps([r['excretion'] for r in results], indent=2)))
        if [r['excretion'] for r in results] == ['false']:
            return

        res = push_message(push_url, token, results)
        if not res:
            is_invalid_token = True
            return
        if res.status == 200:
            logger.debug('push succeeded')
        else:
            logger.error('push result: {} / {}'.format(res.getcode(), json.loads(res.read())))
            is_invalid_token = True
            return
#   schedule.every().minute.at(":{}".format(second)).do(job, model, delay)
    schedule.every(2).minutes.at(":{}".format(second)).do(job, model, delay) #サーバレスポンス遅い対策 2021/11/11 maru
    while True:
        schedule.run_pending()
        time.sleep(1)
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--second', help='各分の何秒に実行するか', type=str, default='00')
    parser.add_argument('--delay', help='取得時刻を何秒前にずらすか', type=int, default=0)
    parser.add_argument('--sdid', help='センサーデバイスID', type=str, default='200720CRB1007596FBD4C')
    parser.add_argument('--loglevel', help='ログレベル', type=str, default='INFO')

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.propagate = False
    stream_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt=log_format)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    args = parser.parse_args()
    ll = args.loglevel.lower()
    if ll == 'critical':
        logger.setLevel(logging.CRITICAL)
    elif ll == 'error':
        logger.setLevel(logging.ERROR)
    elif ll == 'warning':
        logger.setLevel(logging.WARNING)
    elif ll == 'info':
        logger.setLevel(logging.INFO)
    elif ll == 'debug':
        logger.setLevel(logging.DEBUG)

    main(args.sdid, args.second, args.delay)
##############################################################################