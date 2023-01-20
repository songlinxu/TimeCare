from pywebio import start_server, config
from pywebio.session import register_thread
from pywebio.input import input, FLOAT, NUMBER, TEXT
from pywebio.input import select
from pywebio.output import put_text, put_html, put_scope, style, put_link, put_markdown, use_scope
import argparse, time, datetime
from pywebio.platform import path_deploy, path_deploy_http
from os import path
import threading
import numpy as np
import json

from modular_math_new import modular_math


@config(theme="minty")
def timecare():
    with open('usercode_shuffled_new.json', 'r') as fjson:
        usercode_lib = json.load(fjson)
    put_text('TimeCare User Study')
    username = input("Your Invitation Code：", type=TEXT)
    print('username: ',username)
    
    
    if username not in usercode_lib.keys():
        put_text('You are not invited to join in this study.')
    else:
        studytype = select(label='Select Your Study Type', options = ['Task 1: Modular Math'])
        print('studytype: ',studytype)
        if studytype != 'Task 1: Modular Math':
            put_text('Please only choose math task.')
        else:
            strategy = usercode_lib[username]['group']
            order_id = usercode_lib[username]['order']
            modular_math_task = modular_math(username = username, strategy_name = strategy, order_id = order_id)
            modular_math_task.main_run()

if __name__ == '__main__':
    start_server(timecare, port=80, remote_access = True)



