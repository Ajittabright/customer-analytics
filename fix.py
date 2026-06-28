import shutil
import os

src = os.path.join(os.environ['USERPROFILE'], 'Downloads', 'streamlit_app_v3.py')

if os.path.exists(src):
    shutil.copy(src, 'app/streamlit_app.py')
    print("SUCCESS! V3 installed!")
else:
    print("File not found in Downloads!")
    print("Please download streamlit_app_v3.py first")