from contextlib import contextmanager

import torch


class ModuleVRAMManager:
    def __init__(self, device, enabled=False, empty_cache=True, verbose=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.empty_cache = empty_cache
        self.verbose = verbose
        self._modules = {}
        self._states = {}
        self._pinned = set()
        self.log_fn = print

    def register_module(self, name, module, pinned=False):
        self._modules[name] = module
        self._states.setdefault(name, None)
        if pinned:
            self._pinned.add(name)

    def pin_modules(self, names):
        self._pinned.update(names)

    def set_logger(self, log_fn):
        self.log_fn = print if log_fn is None else log_fn

    def _memory_stats(self):
        if not torch.cuda.is_available():
            return None
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2,
            "max_reserved_mb": torch.cuda.max_memory_reserved(self.device) / 1024**2,
        }

    def log_memory(self, event, extra=""):
        if not self.verbose:
            return
        stats = self._memory_stats()
        if stats is None:
            self.log_fn(f"[ModuleVRAMManager] {event} {extra}".rstrip())
            return
        self.log_fn(
            "[ModuleVRAMManager] "
            f"{event} {extra} | "
            f"allocated={stats['allocated_mb']:.1f}MB "
            f"reserved={stats['reserved_mb']:.1f}MB "
            f"max_allocated={stats['max_allocated_mb']:.1f}MB "
            f"max_reserved={stats['max_reserved_mb']:.1f}MB"
        )

    def _move_module(self, name, target):
        module = self._modules[name]
        if self._states.get(name) == target:
            return
        module.to(target)
        self._states[name] = target
        if self.verbose:
            self.log_memory("move", f"{name} -> {target}")

    def load_modules(self, names):
        if not self.enabled:
            return
        for name in names:
            if name in self._modules:
                self._move_module(name, self.device)
        self.log_memory("load_done", f"modules={list(names)}")

    def unload_modules(self, names=None, keep=None):
        if not self.enabled:
            return
        keep = set() if keep is None else set(keep)
        if names is None:
            names = list(self._modules.keys())
        for name in names:
            if name in self._pinned or name in keep:
                continue
            if name in self._modules:
                self._move_module(name, torch.device("cpu"))
        if self.empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.log_memory("unload_done", f"modules={list(names)} keep={list(keep)}")

    def keep_only(self, names):
        if not self.enabled:
            return
        names = set(names)
        self.load_modules(names)
        self.unload_modules(
            [name for name in self._modules.keys() if name not in names],
            keep=names,
        )

    @contextmanager
    def activate(self, names):
        self.load_modules(names)
        try:
            yield
        finally:
            self.unload_modules([name for name in names if name not in self._pinned])
