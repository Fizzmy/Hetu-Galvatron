import os
import logging

def ensure_log_dir(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def get_thread_logger(bsz, chunk, log_dir='logs'):

    logger_name = f"galvatron_bsz{bsz}_chunk{chunk}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    log_file = os.path.join(log_dir, f"search_bsz{bsz}_chunk{chunk}.log")
    file_handler = logging.FileHandler(log_file, mode='w')

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    logger.propagate = False
    
    return logger