import os
from typing import Any, Dict, List
import numpy as np
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

def create_path(path: str, ignore_home: bool = False):
    full_path = ""

    if not ignore_home:
        base_path = os.getenv("EVAL_HOME")

        if not base_path:
            raise Exception("Evaluation home folder not set! Please set EVAL_HOME env variable!")

        full_path = f"{base_path}/{path}"
    else:
        full_path = path

    if not Path(full_path).exists():
        Path(full_path).mkdir(parents=True, exist_ok=True)

def get_path(file_path: str) -> str:
    base_path = os.getenv("EVAL_HOME")

    if not base_path:
        raise Exception("Evaluation home folder not set! Please set EVAL_HOME env variable!")

    full_path = f'{base_path}/{file_path}'

    # Check if directory exists
    path = full_path.split('/')[:-1]

    if not Path('/'.join(path)).exists():
        Path('/'.join(path)).mkdir(parents=True, exist_ok=True)

    return full_path

def file_exists(filename: str) -> bool:
    full_filepath = get_path(filename)
    return Path(full_filepath).is_file()

def save_json_file(filename: str, data: Dict[str, Any] | List[Dict[str, Any]] | Any):
    full_filepath = get_path(filename)

    with open(full_filepath, "w") as outfile: 
        serializable_data = json.dumps(data, default=handle_non_serializable)
        outfile.write(serializable_data)

def save_pickle(filename: str, data: Dict[str, Any] | List[Dict[str, Any]] | Any):
    full_filepath = get_path(filename)

    with open(full_filepath, 'wb') as outfile:
        pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
