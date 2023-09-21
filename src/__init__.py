import logging
import os

try:
    loglevel = os.environ.get("LOGLEVEL")
except:
    loglevel = 10

LOG_FORMAT = ('%(levelname)s:%(asctime)s:%(message)s')
logging.basicConfig(level=loglevel, format=LOG_FORMAT)