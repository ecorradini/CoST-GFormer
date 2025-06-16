import torch
from cost_gformer.data import generate_synthetic_dataset, DataModule
from cost_gformer.model import CoSTGFormer
from cost_gformer.trainer import Trainer


def test_trainer_basic():
    dataset = generate_synthetic_dataset(num_nodes=3, num_snapshots=7, seed=0)
    data = DataModule(dataset, history=2, horizon=1)
    model = CoSTGFormer()
    w_before = model.travel_head.mlp.w1.clone()
    trainer = Trainer(model=model, data=data, epochs=2, batch_size=2)
    trainer.fit()
    assert not torch.allclose(w_before, model.travel_head.mlp.w1)
