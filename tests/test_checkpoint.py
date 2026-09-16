from pathlib import Path
import torch
from merge_emotion.engine.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, opt, None, None, 3, 17, 0.7, {"x": 1})
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters(): p.add_(10)
    payload = load_checkpoint(path, model, opt, restore_rng=False)
    assert payload["epoch"] == 3 and payload["global_step"] == 17
    for k, v in model.state_dict().items():
        assert torch.equal(v, original[k])
