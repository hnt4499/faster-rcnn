registry = {}


def register(key, as_name=None):
    if key not in registry:
        registry[key] = {}

    def decorator(func):
        name = as_name if as_name is not None else func.__name__
        if name in registry[key]:
            raise RuntimeError(
                f"Metric {name} already exists in registry {key}")
        registry[key][name] = func
        return func

    return decorator
