from streamlit.web import bootstrap
from streamlit import config
import os
import sys

if __name__ == '__main__':
    config.set_option("server.port", 8888)
    real_script = "demo-ui.py"
    bootstrap.run(real_script, f'run.py {real_script}', [], {})