import yaml
import argparse
import sys
import os

_PRETAINED_ROOT_PATH_KEYS = {
    'clamp3_path', 'codec_path',
}

_REPO_SOURCE_PATH_KEYS = {
    'config_root', 'data_config',
}

_CHECKPOINT_ROOT_PATH_KEYS = {
    'ckpt_dir', 'l_ckpt_fp', 'resume_ckpt_fp',
}

_ALL_RESOLVED_KEYS = _PRETAINED_ROOT_PATH_KEYS | _REPO_SOURCE_PATH_KEYS | _CHECKPOINT_ROOT_PATH_KEYS


def _get_code_root(config_file_path=None):
    env_root = os.environ.get('LADA_CODE_ROOT')
    if env_root:
        return env_root
    if config_file_path and os.path.exists(config_file_path):
        d = os.path.dirname(os.path.abspath(config_file_path))
        while d and d != '/':
            if os.path.isdir(os.path.join(d, 'lada_band')) and os.path.isdir(os.path.join(d, 'MuCodec')):
                return d
            d = os.path.dirname(d)
    return os.getcwd()


def resolve_relative_paths(config_dict, config_file_path=None):
    pretrained_root = config_dict.get('pretrained_root') or os.environ.get('LADA_PRETRAINED_ROOT', '')
    checkpoint_root = config_dict.get('checkpoint_root') or pretrained_root
    code_root = config_dict.get('code_root') or _get_code_root(config_file_path)

    def _resolve(key, value, root):
        if value is None or root is None or root == '':
            return value
        if os.path.isabs(str(value)):
            return value
        return os.path.join(root, str(value))

    def _walk(d, prefix=''):
        for key in list(d.keys()):
            full_key = key if not prefix else f'{prefix}.{key}'
            value = d[key]
            if isinstance(value, dict):
                _walk(value, full_key)
            elif isinstance(value, str):
                if full_key in _PRETAINED_ROOT_PATH_KEYS:
                    d[key] = _resolve(key, value, pretrained_root)
                elif full_key in _CHECKPOINT_ROOT_PATH_KEYS:
                    d[key] = _resolve(key, value, checkpoint_root)
                elif full_key in _REPO_SOURCE_PATH_KEYS:
                    d[key] = _resolve(key, value, code_root)
                elif key.endswith('_path') or key.endswith('_fp') or key.endswith('_dir'):
                    if not os.path.isabs(value):
                        d[key] = _resolve(key, value, code_root)

    for top_key in list(config_dict.keys()):
        if top_key in _ALL_RESOLVED_KEYS:
            value = config_dict[top_key]
            if isinstance(value, str):
                if top_key in _PRETAINED_ROOT_PATH_KEYS:
                    config_dict[top_key] = _resolve(top_key, value, pretrained_root)
                elif top_key in _CHECKPOINT_ROOT_PATH_KEYS:
                    config_dict[top_key] = _resolve(top_key, value, checkpoint_root)
                elif top_key in _REPO_SOURCE_PATH_KEYS:
                    config_dict[top_key] = _resolve(top_key, value, code_root)

    if 'model' in config_dict and isinstance(config_dict['model'], dict):
        for model_name, model_cfg in config_dict['model'].items():
            if isinstance(model_cfg, dict) and 'json_path' in model_cfg:
                model_cfg['json_path'] = _resolve('json_path', model_cfg['json_path'], pretrained_root)

    return config_dict


class Config:
    def __init__(self, data=None):
        self._data = {}
        if data:
            self.update(data)

    def update(self, data):
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    if k in self._data and isinstance(self._data[k], Config):
                        self._data[k].update(v)
                    else:
                        self._data[k] = Config(v)
                else:
                    self._data[k] = v
        elif isinstance(data, Config):
             self.update(data._data)

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def __repr__(self):
        return f"Config({self._data})"

    def to_dict(self):
        result = {}
        for k, v in self._data.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def __str__(self):
        return str(self.to_dict())

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def resolve_config_path(base_config_path, override_config_name=None):
    if override_config_name is None:
        return base_config_path

    if os.path.isabs(override_config_name) and os.path.exists(override_config_name):
        return override_config_name

    if os.path.exists(override_config_name):
        return override_config_name

    base_dir = os.path.dirname(base_config_path)
    candidates = [os.path.join(base_dir, override_config_name)]
    if not override_config_name.endswith('.yaml'):
        candidates.append(os.path.join(base_dir, override_config_name + '.yaml'))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return override_config_name

def parse_args_and_load_config(default_config_path='lada_band/conf/config.yaml'):
    parser = argparse.ArgumentParser(
        description="Training Script",
        epilog=(
            "Examples:\n"
            "  python train.py --config lada_band/conf/s1_config_1B.yaml\n"
            "  python train.py +cfg_fn=s1_config_1B.yaml train.max_steps=1000 wandb=false"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default=default_config_path,
        help='Path to config file. Recommended way to switch training YAML.',
    )

    args, unknown = parser.parse_known_args()

    cfg_fn_override = None
    filtered_unknown = []
    for arg in unknown:
        if '=' in arg:
            key, value = arg.split('=', 1)
            normalized_key = key[1:] if key.startswith('+') else key
            if normalized_key == 'cfg_fn':
                cfg_fn_override = value
                continue
        filtered_unknown.append(arg)

    config_path = resolve_config_path(args.config, cfg_fn_override)

    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found.")
        config_data = {}
    else:
        config_data = load_yaml(config_path)
        if config_data is None:
            config_data = {}

    for arg in filtered_unknown:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key.startswith('+'):
                key = key[1:]

            if value.lower() == 'true': value = True
            elif value.lower() == 'false': value = False
            elif value.lower() == 'none': value = None
            elif value.isdigit(): value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass

            keys = key.split('.')
            current = config_data
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                if not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

    config_data = resolve_relative_paths(config_data, config_file_path=config_path)

    return Config(config_data)
