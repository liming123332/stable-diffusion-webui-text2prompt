import os
import re
from modules import scripts

# to use intellisense on vscode
DEVELOP = False

def get_abspath(path: str):
    return os.path.abspath(os.path.join(scripts.basedir(), path))


TOKENIZER_NAMES = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2']

TOKENIZER_MODELS = {
    TOKENIZER_NAMES[0]: f'sentence-transformers/{TOKENIZER_NAMES[0]}',
    TOKENIZER_NAMES[1]: f'sentence-transformers/{TOKENIZER_NAMES[1]}'
}

DATABASE_PATH_DANBOORU = get_abspath('data/danbooru')

RE_TOKENFILE_DANBOORU = re.compile(r'(danbooru_[^_]+)_token_([^_]+)')

# Text2Prompt Fixed Prompts Settings
DEFAULT_FIXED_PREFIX = "masterpiece, best quality, highres"
DEFAULT_FIXED_SUFFIX = ""
DEFAULT_ENABLE_FIXED = True

# 用户可配置的固定提示词设置
FIXED_PROMPT_SETTINGS = {
    'prefix': DEFAULT_FIXED_PREFIX,
    'suffix': DEFAULT_FIXED_SUFFIX,
    'enabled': DEFAULT_ENABLE_FIXED
}