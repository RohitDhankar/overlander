import os , sys
import logging
import logging.config
from datetime import date
from datetime import datetime

#from settings import LOGS_PATH, LOG_LEVEL
#from utilities.db_logger_handler import DBLoggingHandler
#"class": "utilities.db_logger_handler.DBLoggingHandler",

LOGS_DIR_PATH = "../logs_dir/"
logging_level = "DEBUG" #"INFO" #https://docs.python.org/3/library/logging.html#logging.Logger.setLevel
os.makedirs(LOGS_DIR_PATH, exist_ok=True)

def setup_logger_linux(module_name=None): #, folder_name=None):
    """ 
    """
    
    dt_time_now = datetime.now()
    hour_now = dt_time_now.strftime("_%m_%d_%Y_%H")  # Date Format == _%m_%d_%Y 
    if module_name:
        module_log_file_name = "ipwebcam_log_"+hour_now+"00h_" 
        module_log_file = os.path.join(LOGS_DIR_PATH, f'{module_log_file_name}.log')
   
    cnfg_dict = {
        'version': 1,
        "disable_existing_loggers": False,
        "formatters": {
            "ipwebcam_log_format": {
                "format": "%(asctime)s - %(levelname)s -_Name: %(filename)s -_Meth_Name: "
                          "%(funcName)s() -_Line: %(lineno)d -_Log_Message:  %(message)s",
                "datefmt": "_%m_%d_%Y_%H:%M:%S"
            }
        },
        'handlers': {
            'common_handler': {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": logging_level,
                "formatter": "ipwebcam_log_format",
                "filename": module_log_file,
                "when": "midnight",
                "interval": 1,
                "backupCount": 31,
                "encoding": "utf8"
            }
        },
        'loggers': {
            module_name: {
                        'level': logging_level,
                        'handlers': ['common_handler'],
                        "propagate": False
                        }
                    }
    }

    logging.config.dictConfig(cnfg_dict)
    linux_logger = logging.getLogger(module_name)
    #linux_logger.propagate = False # no repeat Log prints
    return linux_logger

def setup_logger_winOS(module_name=None): #, folder_name=None):
    """
    Desc:
        - 
    """

    dt_time_now = datetime.now()
    hour_now = dt_time_now.strftime("_%m_%d_%Y_%H")  # Date Format == _%m_%d_%Y 
    
    if not os.path.exists(LOGS_DIR_PATH):
        os.makedirs(LOGS_DIR_PATH)
    # hourly_log_path = os.path.join(LOGS_PATH, f'{hour_now}')
    # if not os.path.exists(hourly_log_path):
    #     os.makedirs(hourly_log_path)
    
    # if folder_name is None:
    #     module_log_file = os.path.join(hourly_log_path, f'{module_name}.log')
    # else:
        # if not os.path.exists(os.path.join(hourly_log_path, module_name)):
        #     os.makedirs(os.path.join(hourly_log_path, module_name))
    if module_name:
        module_log_file_name = "ipwebcam_log_"+hour_now+"00h_" 
        module_log_file = os.path.join(LOGS_DIR_PATH, f'{module_log_file_name}.log')

    cnfg_dict = {
        'version': 1,
        "disable_existing_loggers": True,
        "formatters": {
            "ipwebcam_log_format": {
                "format": "%(asctime)s - %(levelname)s -_Name: %(filename)s -_Meth_Name: "
                          "%(funcName)s() -_Line: %(lineno)d -_Log_Message:  %(message)s",
                "datefmt": "_%m_%d_%Y_%H:%M:%S"
            }
        },
        'handlers': {
            'common_handler': {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": logging_level,
                "formatter": "ipwebcam_log_format",
                "filename": module_log_file,
                "when": "midnight",
                "interval": 1,
                "backupCount": 31,
                "encoding": "utf8"
            }
        
        },
        'loggers': {
            'general': {
                        'level': logging_level,
                        'handlers': ['common_handler']
                        }
                    }
    }


    logging.config.dictConfig(cnfg_dict)
    return logging.getLogger(module_name)



"""  
if module_name != 'general':

module_handler = {
'class': 'logging.FileHandler',
'level': LOG_LEVEL,
'formatter': "ollama_log_format",
'filename': module_log_file
}
module_logger = {
'level': LOG_LEVEL,
'handlers': [f'console_for_{module_name}','common_handler'] #
}

if f'console_for_{module_name}' not in cnfg_dict['handlers'].keys():
cnfg_dict['handlers'][f'console_for_{module_name}'] = module_handler
if module_name not in cnfg_dict['loggers'].keys():
cnfg_dict['loggers'][module_name] = module_logger
"""



""" 
# linux_logger.setLevel(logging_level) # INFO to supress all DEBUG LOgs 
# if linux_logger.hasHandlers(): # clearall other Handlers -- remove Double Prints 
#     linux_logger.handlers.clear()
#stream_handler = logging.StreamHandler(sys.Stdout) # TODO -- call Handler if no Log File required to log on WB APP to STDOUT
# ipwebcam_log_format= {
#                     "format": "%(asctime)s - %(levelname)s -_Name: %(filename)s -_Meth_Name: "
#                                 "%(funcName)s() -_Line: %(lineno)d -_Log_Message:  %(message)s",
#                     "datefmt": "_%m_%d_%Y_%H:%M:%S"
#                     }
# formatter = logging.Formatter(fmt=ipwebcam_log_format["format"],
#                               datefmt=ipwebcam_log_format["datefmt"])
# logging_file_handler= {
#                         "class": "logging.handlers.TimedRotatingFileHandler",
#                         "level":logging_level,
#                         "formatter": "ipwebcam_log_format",
#                         "filename": module_log_file,
#                         "when": "midnight",
#                         "interval": 1,
#                         "backupCount": 31,
#                         "encoding": "utf8"
#                     }
# linux_logger.addHandler(logging_file_handler)


"""