import os
import tempfile
import pytest
import torch

from apex_classifier_750m.config import Apex748MConfig
from apex_classifier_750m.ast_sandbox import SandboxedMathEvaluator, SecurityError
from apex_classifier_750m.model import Apex748MClassifier
from apex_classifier_750m.train import (
    get_execution_device,
    save_portable_checkpoint,
    load_portable_checkpoint,
)


def test_exact_748m_parameter_count():
    """Verify that Apex748MClassifier has exactly 746,813,455 parameters."""
    config = Apex748MConfig()
    model = Apex748MClassifier(config)
    total_params = model.param_count()
    assert total_params == 746813455, f"Expected 746,813,455 params, got {total_params}"


def test_sandboxed_math_evaluator_valid():
    evaluator = SandboxedMathEvaluator()
    assert evaluator.evaluate("100 + 250") == 350.0
    assert evaluator.evaluate("(50 * 4) - (10 / 2)") == 195.0
    assert evaluator.evaluate("-15.5 + 20") == 4.5
    assert evaluator.evaluate("(100 // 3) + (10 % 3)") == 34.0


def test_sandboxed_math_evaluator_guards():
    evaluator = SandboxedMathEvaluator()

    # Zero division
    with pytest.raises(ZeroDivisionError):
        evaluator.evaluate("10 / 0")

    # Disallowed builtins / calls
    with pytest.raises(SecurityError):
        evaluator.evaluate("__import__('os').system('calc')")

    # Bool rejection (raises TypeError from visit_Constant)
    with pytest.raises((SecurityError, TypeError)):
        evaluator.evaluate("True + 1")

    # Disallowed operators (e.g. Pow)
    with pytest.raises(SecurityError):
        evaluator.evaluate("2 ** 8")


def test_model_forward_pass():
    """Test forward pass on synthetic batch with 2 chunks."""
    config = Apex748MConfig(num_hidden_layers=2, num_memory_layers=1)  # lightweight for fast test
    model = Apex748MClassifier(config)

    batch_size = 2
    num_chunks = 2
    chunk_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, num_chunks, chunk_len))

    # Choice candidates: 3 candidates per batch item
    num_candidates = 3
    candidate_tokens = torch.randint(0, config.vocab_size, (batch_size, num_candidates, chunk_len))

    outputs = model(input_ids, candidate_tokens=candidate_tokens)

    assert "noul_prob" in outputs
    assert outputs["noul_prob"].shape == (batch_size, 1)
    assert (outputs["noul_prob"] >= 0.0).all() and (outputs["noul_prob"] <= 1.0).all()

    assert "score_probs" in outputs
    assert outputs["score_probs"].shape == (batch_size, 11)
    assert torch.allclose(outputs["score_probs"].sum(dim=-1), torch.ones(batch_size), atol=1e-4)

    assert "choice_logits" in outputs
    assert outputs["choice_logits"].shape == (batch_size, num_candidates)

    assert "abstain_prob" in outputs
    assert outputs["abstain_prob"].shape == (batch_size, 1)


def test_checkpoint_roundtrip():
    device, device_name, is_tpu = get_execution_device()
    assert device is not None

    config = Apex748MConfig(num_hidden_layers=1, num_memory_layers=1)
    model = Apex748MClassifier(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = save_portable_checkpoint(model, optimizer, step=42, save_dir=tmpdir)
        assert os.path.exists(ckpt_path)

        model2 = Apex748MClassifier(config)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        step = load_portable_checkpoint(ckpt_path, model2, opt2, device=torch.device("cpu"))
        assert step == 42


def test_cross_optimizer_adafactor_to_adamw():
    """Verify Adafactor checkpoint (Kaggle GPU) safely loads into AdamW (Colab TPU/CPU) without KeyError: 'betas'."""
    config = Apex748MConfig(num_hidden_layers=1, num_memory_layers=1)
    model1 = Apex748MClassifier(config)
    opt1 = torch.optim.Adafactor(model1.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = save_portable_checkpoint(model1, opt1, step=77, save_dir=tmpdir)

        model2 = Apex748MClassifier(config)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        step = load_portable_checkpoint(ckpt_path, model2, opt2, device=torch.device("cpu"))

        assert step == 77
        assert "betas" in opt2.param_groups[0]
        assert opt2.param_groups[0]["betas"] == (0.9, 0.999)

        # Run step to verify no KeyError: 'betas' in AdamW.step()
        dummy_inp = torch.randint(0, config.vocab_size, (1, 1, 16))
        dummy_cand = torch.randint(0, config.vocab_size, (1, 2, 16))
        out = model2(dummy_inp, candidate_tokens=dummy_cand)
        loss = out["noul_logit"].sum()
        loss.backward()
        opt2.step()
        opt2.zero_grad()


def test_cross_optimizer_adamw_to_adafactor():
    """Verify AdamW checkpoint (Colab TPU) safely loads into Adafactor (Kaggle GPU) without unpack errors."""
    config = Apex748MConfig(num_hidden_layers=1, num_memory_layers=1)
    model1 = Apex748MClassifier(config)
    opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = save_portable_checkpoint(model1, opt1, step=88, save_dir=tmpdir)

        model2 = Apex748MClassifier(config)
        opt2 = torch.optim.Adafactor(model2.parameters(), lr=1e-4)
        step = load_portable_checkpoint(ckpt_path, model2, opt2, device=torch.device("cpu"))

        assert step == 88
        assert isinstance(opt2.param_groups[0]["eps"], (tuple, list))

        # Run step to verify no crash in Adafactor.step()
        dummy_inp = torch.randint(0, config.vocab_size, (1, 1, 16))
        dummy_cand = torch.randint(0, config.vocab_size, (1, 2, 16))
        out = model2(dummy_inp, candidate_tokens=dummy_cand)
        loss = out["noul_logit"].sum()
        loss.backward()
        opt2.step()
        opt2.zero_grad()


def test_legacy_checkpoint_without_optimizer_type():
    """Verify backwards-compatibility with checkpoints that lacked 'optimizer_type' in state dict."""
    config = Apex748MConfig(num_hidden_layers=1, num_memory_layers=1)
    model1 = Apex748MClassifier(config)
    opt1 = torch.optim.Adafactor(model1.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = os.path.join(tmpdir, "legacy.pt")
        # Save without 'optimizer_type'
        torch.save({
            "step": 99,
            "model_state": model1.state_dict(),
            "opt_state": opt1.state_dict(),
            "config": config,
        }, target_path)

        model2 = Apex748MClassifier(config)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        step = load_portable_checkpoint(target_path, model2, opt2, device=torch.device("cpu"))

        assert step == 99
        assert "betas" in opt2.param_groups[0]
        assert opt2.param_groups[0]["betas"] == (0.9, 0.999)

        # Ensure AdamW.step() executes without KeyError: 'betas'
        dummy_inp = torch.randint(0, config.vocab_size, (1, 1, 16))
        dummy_cand = torch.randint(0, config.vocab_size, (1, 2, 16))
        out = model2(dummy_inp, candidate_tokens=dummy_cand)
        loss = out["noul_logit"].sum()
        loss.backward()
        opt2.step()
        opt2.zero_grad()

