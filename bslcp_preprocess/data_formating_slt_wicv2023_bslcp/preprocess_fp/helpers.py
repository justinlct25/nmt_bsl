import re
from datetime import datetime
import json
import gzip
import pickle


def load_config(file='config.json'):
    with open(file) as f:
        return json.load(f)