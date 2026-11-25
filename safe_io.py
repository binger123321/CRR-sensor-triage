"""Safe loaders: only numpy array reconstruction is allowed when unpickling data files."""
import pickle, torch
_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype")}
class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            if module.startswith("numpy.core"):
                module = module.replace("numpy.core", "numpy._core") if hasattr(__import__("numpy"), "_core") else module
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")
def load_pkl(path):
    with open(path, "rb") as f:
        return _Restricted(f).load()
def load_weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)
