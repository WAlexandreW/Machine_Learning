"""
The code below is stricly for educational purpose 
I am not responsible for the potential wrong use of the snippet below

"""

import requests
from bs4 import BeautifulSoup
from scrapy import Selector
import datetime
import argparse
import re
import urllib.request
import time
import requests
from random import choice

import time
timestr = time.strftime("%Y%m%d")

#Optional : make a backup of the file, just in case
import shutil
shutil.copy(r'C:\Users\Alexandre\Project\Webscrapping.py', r'C:\Users\Alexandre\Project\Backup_Scripts\Webscrapping_'+timestr+'.py')




