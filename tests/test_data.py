import numpy as np
import pandas as pd
import torch

from favapy.data import FAVADataManager, preprocess_expression, resolve_n_hidden, resolve_n_latents
from favapy.models import FAVAModule
from favapy.training import FAVATrainingPlan


def test_preprocess_expression_log2_and_minmax():
    x = np.array([[0.0, 3.0], [1.0, 7.0]], dtype=np.float32)
    out = preprocess_expression(x)
    assert out.shape == x.shape
    assert np.all(out >= 0)
    assert np.all(out <= 1)


def test_preprocess_skips_log2_for_negative_values():
    x = np.array([[-1.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    out = preprocess_expression(x)
    assert np.all(out >= 0)
    assert np.all(out <= 1)
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[0, 1], 1.0)


def test_architecture_defaults():
    assert resolve_n_hidden(2500) == 1000
    assert resolve_n_hidden(800) == 500
    assert resolve_n_hidden(100) == 50
    assert resolve_n_latents(1000) == 100
    assert resolve_n_latents(500) == 50
    assert resolve_n_latents(50) == 5


def test_dataframe_extraction():
    df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=['gene_a', 'gene_b'],
        columns=['cell_1', 'cell_2'],
    )
    manager = FAVADataManager.from_input(df)
    assert manager.n_genes == 2
    assert manager.n_obs == 2
    assert manager.var_names == ['gene_a', 'gene_b']


def test_legacy_loss_with_fixed_noise():
    torch.manual_seed(0)
    module = FAVAModule(n_obs=4, n_hidden=3, n_latents=2)
    x = torch.rand(2, 4)
    epsilon = torch.zeros(2, 2)
    inference = module.inference(x, epsilon=epsilon)
    generative = module.generative(inference.z)
    loss = module.loss(x, inference, generative)

    recon = (x - generative.reconstruction).square().mean(dim=-1) * 4
    kl = -0.5 * (1 + inference.z_log_sigma - inference.z_mean.square() - inference.z_log_sigma.exp()).sum(dim=-1)
    expected = (0.9 * recon + 0.1 * kl).mean()
    assert torch.allclose(loss.loss, expected)


def test_xavier_initialization():
    module = FAVAModule(n_obs=8, n_hidden=4, n_latents=2)
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            assert layer.bias is not None
            assert torch.all(layer.bias == 0)


def test_per_parameter_gradient_clipping():
    plan = FAVATrainingPlan(n_obs=4, n_hidden=3, n_latents=2)
    optimizer = plan.configure_optimizers()
    x = torch.rand(2, 4)
    loss = plan.training_step((x, torch.tensor([0, 1])), 0)
    loss.backward()
    plan.configure_gradient_clipping(optimizer, None, None)
    for parameter in plan.parameters():
        if parameter.grad is not None:
            assert parameter.grad.norm().item() <= 0.001 + 1e-6
