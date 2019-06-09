from download_utils import download_data, download_models
import os

download_data('data')
os.system("unzip data/picsart_hack_online_data.zip -d data")
os.system("rm -r data/picsart_hack_online_data.zip")
os.system("rm -r data/__MACOSX")
os.system("rm -r data/sample_submission.csv")

download_models('models')
os.system("unzip models/models.zip -d models")
os.system("rm -r models/models.zip")
os.system("rm -r models/__MACOSX")
