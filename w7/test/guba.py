# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/5/19
Description:
"""
import json
import re

import requests


class GuBa(object):
    def __init__(self):
        self.base_url = 'http://guba.eastmoney.com/default,99_%s.html'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'
        }
        self.infos = []
        self.parse()

    def parse(self):
        for i in range(1,2):
            response = requests.get(self.base_url % i, headers=self.headers)

            '''阅读数,评论数,标题,作者,更新时间,详情页url'''
            ul_pattern = re.compile(r'<ul class="newlist" tracker-eventcode="gb_xgbsy_ lbqy_rmlbdj">(.*?)</ul>', re.S)
            ul_content = ul_pattern.search(response.text)
            if ul_content:
                ul_content = ul_content.group()

            print(ul_content)

gb=GuBa()
