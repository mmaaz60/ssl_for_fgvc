from importlib import import_module


def get_object_from_path(path):
    assert type(path) is str
    mod_path = '.'.join(path.split('.')[:-1])
    object_name = path.split('.')[-1]
    mod = import_module(mod_path)
    target_obj = getattr(mod, object_name)
    return target_obj
