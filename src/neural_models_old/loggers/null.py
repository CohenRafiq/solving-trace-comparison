from neural_models_old.loggers.base import BaseLogger


class NullLogger(BaseLogger):
    def setup(self, cfg=None):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_model(self, *args, **kwargs):
        pass

    def watch_model(self, *args, **kwargs):
        pass

    def finish(self):
        pass
