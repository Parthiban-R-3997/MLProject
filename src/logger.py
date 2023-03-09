import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" ## Ensures how the log file strucure has to be
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) ## Creating folder and naming convetion of folder
os.makedirs(logs_path,exist_ok=True) ## exist_ok=True means eventhough there is a file and folder it will keep on appending the files inside

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE) ## Logs files path inside the Logs folder

logging.basicConfig(
    filename=LOG_FILE_PATH, ## Where you want to basically save it
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,


)


