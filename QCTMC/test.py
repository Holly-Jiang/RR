import requests
import time
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('-u', type=str, default=None, help='username')
parser.add_argument('-p', type=str, default=None, help='password')
parser.add_argument('-op', action='store_false', default=True, help='operation')
args = parser.parse_args()


def isConnected():
    try:
        res = requests.get("http://www.baidu.com", timeout=2, verify=False)
        g = re.findall(r".+?baidu.+?", res.text)
        if len(g) > 0:
            return True
        else:
            return False
    except:
        return False

def auto_connect(uname, pwd):
    assert (uname is not None and pwd is not None), "username or password is None"
    url = 'https://login.ecnu.edu.cn/include/auth_action.php'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    data = {
        'action': 'login',
        'username': uname,
        'password': pwd,
        'ac_id': 1,
        'user_ip':'',
        'nas_ip':'',
        'user_mac':'',
        'save_me':0,
        'ajax':1
    }
    res = requests.post(url, data=data, headers=headers, verify=False)
    if 'login_ok' in res.text:
        print('Login successfully !')
    else:
        print('Login failure, please check your username and password !')
    # if res.status_code == 200:
    #     if isConnected():
    #         time.sleep(1)
    #         print('Login successfully !')
    #     else:
    #         print('Login failure, please check your username and password !')
    # else:
    #     print('Connection error occurred !')


def logout():
    url = 'https://login.ecnu.edu.cn/srun_portal_pc.php?ac_id=1&'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    html = requests.get(url, headers=headers, verify=False)
    action = re.findall(r"<input.+?type=\"hidden\".+?name=\"action\".+?value=\"(.+?)\">", html.text)[1]
    user_ip = re.findall(r"<input.+?type=\"hidden\".+?name=\"user_ip\".+?value=\"(.+?)\">", html.text)[0]
    username = re.findall(r"<input.+?type=\"hidden\".+?name=\"username\".+?value=\"(.+?)\">", html.text)[0]
    data = {
        'action': action,
        'user_ip': user_ip,
        'username': username
    }
    res = requests.post(url, data=data, headers=headers)
    if res.status_code == 200:
        if not isConnected():
            print('Logout successfully......')
        else:
            print('Logout failure......')
    else:
        print('Request error occurred !')


minute = 10
# print(args.u, args.p, args.op)
if args.op:
    print('Get ready to login network......')
    while True:
        print('Current connection status is', isConnected())
        if not isConnected():
            print('Try to login network......')
            auto_connect(args.u, args.p)
        time.sleep(60 * minute)
else:
    print('Get ready to logout network......')
    logout()