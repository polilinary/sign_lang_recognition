import os
import zipfile

os.makedirs("/root/.kaggle", exist_ok=True)
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

# Скачиваем датасет
!kaggle datasets download -d haqishen/russian-sign-language -p dataset/
!unzip dataset/russian-sign-language.zip -d dataset/
