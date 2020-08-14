# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:40:41 2019

@author: tomson
"""
import os
#try:
from torch.hub import _get_torch_home
torch_cache_home = _get_torch_home()
#except ImportError:
#    torch_cache_home = os.path.expanduser(
#        os.getenv('TORCH_HOME', os.path.join(
#            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'pytorch_transformers')
print(default_cache_path)

#try:
#    from urllib.parse import urlparse
#except ImportError:
#    from urlparse import urlparse

print(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE', default_cache_path))
from pathlib import Path
PYTORCH_PRETRAINED_BERT_CACHE = Path(
    os.getenv('PYTORCH_TRANSFORMERS_CACHE', os.getenv('PYTORCH_PRETRAINED_BERT_CACHE', default_cache_path)))
#except (AttributeError, ImportError):
#    print('hahah')
#    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_TRANSFORMERS_CACHE',
#                                              os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
#                                                        default_cache_path))

PYTORCH_TRANSFORMERS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility
print(PYTORCH_TRANSFORMERS_CACHE)