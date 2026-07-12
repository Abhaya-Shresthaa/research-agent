__all__ = ["model", "model1", "model2", "make_model"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from importlib import import_module

    model_module = import_module("model.model")
    return getattr(model_module, name)
