import pytest

from pcn import PCNetwork

import torch


@pytest.fixture
def net():
    net = PCNetwork([3, 4, 2])
    net.reset()
    return net


def test_member_lengths(net):
    assert len(net.W) == 2
    assert len(net.b) == 2
    assert len(net.eps) == 2
    assert len(net.fp) == 2
    assert len(net.x) == 3


def test_weight_sizes(net):
    assert net.W[0].shape == (4, 3)
    assert net.W[1].shape == (2, 4)


def test_x_sizes(net):
    assert [len(_) for _ in net.x] == [3, 4, 2]


def test_inference_does_not_change_input_sample(net):
    x0 = torch.FloatTensor([0.5, 0.7, 0.2])
    net.x[0] = x0.clone()
    net.infer(torch.FloatTensor([0.3, -0.4]))

    assert torch.allclose(x0, net.x[0])


def test_all_xs_change_during_inference(net):
    old_x = [_.clone() for _ in net.x]
    net.infer(torch.FloatTensor([0.3, -0.4]))

    # inference by definition doesn't update the input sample, x[0]
    for old, new in zip(old_x[1:], net.x[1:]):
        assert not torch.any(torch.isclose(old, new))


def test_all_xs_change_during_forward(net):
    old_x = [_.clone() for _ in net.x]
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    # forward should set x[0] to the input sample
    for old, new in zip(old_x, net.x):
        assert not torch.any(torch.isclose(old, new))


def test_forward_result_is_stationary_point_of_inference(net):
    x0 = torch.FloatTensor([0.5, -0.7, 0.2])
    net.forward(x0)

    old_x = [_.clone() for _ in net.x]
    net.infer(old_x[-1])

    for old, new in zip(old_x, net.x):
        assert torch.allclose(old, new)


def test_weights_change_during_learning(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])

    old_W = [_.clone() for _ in net.W]
    net.learn(x0, y0)

    for old, new in zip(old_W, net.W):
        assert not torch.any(torch.isclose(old, new))


def test_biases_change_during_learning(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])

    old_b = [_.clone() for _ in net.b]
    net.learn(x0, y0)

    for old, new in zip(old_b, net.b):
        assert not torch.any(torch.isclose(old, new))


def test_errors_are_nonzero_in_supervised_mode(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    net.learn(x0, y0)

    for eps in net.eps:
        assert eps.abs().min() > 1e-6


@pytest.mark.parametrize("var", ["W", "b", "eps", "fp"])
def test_forward_does_not_change_anything_but_x(net, var):
    old_var = [_.clone() for _ in getattr(net, var)]
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    new_var = getattr(net, var)
    for old, new in zip(old_var, new_var):
        assert torch.allclose(old, new)


@pytest.mark.parametrize("var", ["W", "b", "eps", "fp"])
def test_update_variables_does_not_change_anything_but_x(net, var):
    old_var = [_.clone() for _ in getattr(net, var)]
    net.update_variables()

    new_var = getattr(net, var)
    for old, new in zip(old_var, new_var):
        assert torch.allclose(old, new)


@pytest.mark.parametrize("var", ["W", "b", "x"])
def test_calculate_errors_does_not_change_anything_but_eps_and_fp(net, var):
    old_var = [_.clone() for _ in getattr(net, var)]
    net.calculate_errors()

    new_var = getattr(net, var)
    for old, new in zip(old_var, new_var):
        assert torch.allclose(old, new)


@pytest.mark.parametrize("var", ["eps", "fp", "x"])
def test_update_weights_does_not_change_anything_but_w_and_b(net, var):
    old_var = [_.clone() for _ in getattr(net, var)]
    net.update_weights()

    new_var = getattr(net, var)
    for old, new in zip(old_var, new_var):
        assert torch.allclose(old, new)


@pytest.mark.parametrize("var", ["eps", "fp", "x", "W", "b"])
def test_no_nan_or_inf_after_a_few_learning_steps(net, var):
    torch.manual_seed(0)
    for i in range(4):
        x = torch.Tensor(3).uniform_()
        y = torch.Tensor(2).uniform_()
        net.learn(x, y)

    for _ in getattr(net, var):
        assert torch.all(torch.isfinite(_))


def test_forward_output_depends_on_input(net):
    y1 = net.forward(torch.FloatTensor([0.1, 0.3, -0.2]))
    y2 = net.forward(torch.FloatTensor([-0.5, 0.1, 0.2]))
    assert not torch.allclose(y1, y2)


def test_forward_sets_first_element_of_x_to_input_sample(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    net.forward(x0)
    assert torch.allclose(net.x[0], x0)


def test_infer_sets_last_element_of_x_to_output_sample(net):
    y0 = torch.FloatTensor([0.5, -0.2])
    net.infer(y0)
    assert torch.allclose(net.x[-1], y0)


def test_learn_sets_first_and_last_elements_of_x_to_input_and_output_samples(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    y0 = torch.FloatTensor([0.5, -0.2])
    net.learn(x0, y0)

    assert torch.allclose(net.x[0], x0)
    assert torch.allclose(net.x[-1], y0)


@pytest.mark.parametrize("var", ["eps", "fp", "x", "W", "b"])
def test_reset_values_same_when_torch_seed_is_same(var):
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = PCNetwork(dims)
    net.reset()

    var1 = [_.clone() for _ in getattr(net, var)]

    torch.manual_seed(seed)
    net.reset()

    var2 = [_.clone() for _ in getattr(net, var)]

    for old, new in zip(var1, var2):
        assert torch.allclose(old, new)


def test_reset_weights_change_for_subsequent_calls_if_seed_not_reset():
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = PCNetwork(dims)

    net.reset()
    var1 = [_.clone() for _ in net.W]

    net.reset()
    var2 = [_.clone() for _ in net.W]

    for old, new in zip(var1, var2):
        assert not torch.any(torch.isclose(old, new))


@pytest.mark.parametrize("var", ["W", "b"])
def test_weights_reproducible_for_same_seed_after_learning(var):
    seed = 321
    dims = [2, 6, 5, 3]

    x = torch.FloatTensor([[0.2, -0.3], [0.5, 0.7], [-0.3, 0.2]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.7], [1.5, 0.6, -0.3], [-0.2, 0.5, 0.6]])

    net = PCNetwork(dims)

    # do some learning
    torch.manual_seed(seed)
    net.reset()
    for crt_x, crt_y in zip(x, y):
        net.learn(crt_x, crt_y)

    var1 = [_.clone() for _ in getattr(net, var)]

    # reset and do the learning again
    torch.manual_seed(seed)
    net.reset()
    for crt_x, crt_y in zip(x, y):
        net.learn(crt_x, crt_y)

    var2 = [_.clone() for _ in getattr(net, var)]

    for old, new in zip(var1, var2):
        assert torch.allclose(old, new)


@pytest.mark.parametrize("var", ["W", "b"])
def test_learning_effects_are_different_for_subsequent_runs(var):
    seed = 321
    dims = [2, 6, 5, 3]

    x = torch.FloatTensor([[0.2, -0.3], [0.5, 0.7], [-0.3, 0.2]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.7], [1.5, 0.6, -0.3], [-0.2, 0.5, 0.6]])

    net = PCNetwork(dims)

    # do some learning
    torch.manual_seed(seed)
    net.reset()
    for crt_x, crt_y in zip(x, y):
        net.learn(crt_x, crt_y)

    var1 = [_.clone() for _ in getattr(net, var)]

    # reset and do the learning again -- without resetting random seed this time!
    net.reset()
    for crt_x, crt_y in zip(x, y):
        net.learn(crt_x, crt_y)

    var2 = [_.clone() for _ in getattr(net, var)]

    for old, new in zip(var1, var2):
        assert not torch.allclose(old, new)


def test_compare_to_reference_implementation():
    from pcn_ref import PCNetworkRef

    seed = 100
    dims = [2, 6, 5, 3]

    x = torch.FloatTensor([[0.2, -0.3], [0.5, 0.7], [-0.3, 0.2]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.7], [1.5, 0.6, -0.3], [-0.2, 0.5, 0.6]])

    net = PCNetwork(dims)

    # do some learning
    torch.manual_seed(seed)
    net.reset()
    for crt_x, crt_y in zip(x, y):
        net.learn(crt_x, crt_y)

    test_x = torch.FloatTensor([0.5, 0.2])
    out = net.forward(test_x)

    net_ref = PCNetworkRef(dims)

    # do some learning
    torch.manual_seed(seed)
    net_ref.reset()
    for crt_x, crt_y in zip(x, y):
        net_ref.learn(crt_x, crt_y)

    out_ref = net_ref.forward(test_x)

    assert torch.allclose(out, out_ref)
