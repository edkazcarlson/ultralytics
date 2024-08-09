# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    working = False
    try:
        raySessionOutput = ray.train._internal.session._get_session()  # replacement for deprecated ray.tune.is_session_enabled()
        print(f'Got raySessionOutput:\n{raySessionOutput}')
        working = True
    except Exception as e:
        print(f'ray.train._internal.session._get_session() failed, reason:\n{e}')

    if not working:
        try:
            raySessionOutput = ray.train._internal.session.get_session()  # replacement for deprecated ray.tune.is_session_enabled()
            print(f'Got raySessionOutput:\n{raySessionOutput}')
            working = True
        except Exception as e:
            print(f'ray.train._internal.session.get_session() failed, reason:\n{e}')

    if not working:
        print('failed to get raytune to work, falling out....')
        return
    metrics = trainer.metrics
    metrics["epoch"] = trainer.epoch
    session.report(metrics)

callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
