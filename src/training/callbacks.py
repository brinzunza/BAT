import copy
class EarlyStopping:
    """Stop when validation loss has not improved for `patience` epochs; keeps the best weights."""
    def __init__(s, patience=6, min_delta=0.0): s.patience, s.min_delta, s.best, s.bad, s.state = patience, min_delta, float("inf"), 0, None
    def step(s, val_loss, model):
        if val_loss < s.best - s.min_delta: s.best, s.bad, s.state = val_loss, 0, copy.deepcopy(model.state_dict())
        else: s.bad += 1
        return s.bad >= s.patience
    def restore(s, model):
        if s.state is not None: model.load_state_dict(s.state)
