## Virtualenv settings
activate_this='/var/www/deep_zoom_test/app/venv/bin/activate_this.py'
exec(open(activate_this).read(),dict(__file__=activate_this))


import sys
import site
from logging import Formatter, FileHandler
import os


ALLDIRS =['/var/www/deep_zoom_test/app/venv/lib/python3.5/site-packages']

#Remember original sys.path.
prev_sys_path= list(sys.path)
# Addeach new site-packages directory.
for directory in ALLDIRS:
    site.addsitedir(directory)
# Reordersys.path so new directories at the front.
new_sys_path= []
for item in list(sys.path):
    if item not in prev_sys_path:
        new_sys_path.append(item)
        sys.path.remove(item)
sys.path[:0]= new_sys_path

sys.path.insert(0,"/var/www/deep_zoom_test")
os.chdir("/var/www/deep_zoom_test")


from app import app as application
application.secret_key = 'test'
