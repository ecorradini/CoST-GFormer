import torch
from cost_gformer.data import generate_synthetic_dataset, DataModule
from cost_gformer.model import CoSTGFormer
from cost_gformer.trainer import Trainer


def test_trainer_basic():
    dataset = generate_synthetic_dataset(
        num_nodes=3, num_snapshots=7, dynamic_dim=2, seed=0
    )
    data = DataModule(dataset, history=2, horizon=1)
    model = CoSTGFormer()
    w_before = model.travel_head.mlp.w1.clone()
    trainer = Trainer(model=model, data=data, epochs=2, batch_size=2)
    trainer.fit()
    assert not torch.allclose(w_before, model.travel_head.mlp.w1)


def test_trainer_multistep():
    dataset = generate_synthetic_dataset(
        num_nodes=3, num_snapshots=8, dynamic_dim=2, seed=1
    )
    data = DataModule(dataset, history=2, horizon=2)
    model = CoSTGFormer()
    trainer = Trainer(model=model, data=data, epochs=1, batch_size=1)
    trainer.fit()
    hist, fut = data[0]
    tt, cr = model.forward(hist + fut, horizon=2)
    assert tt.shape[0] == 2
    assert cr.shape[0] == 2
