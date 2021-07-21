import pytest

from pcn import PCNetwork

import torch


@pytest.fixture
def net():
    net = PCNetwork([3, 4, 2])
    return net


def test_number_of_layers(net):
    assert len(net.layers) == 2
    assert len(net.x) == 3


def test_weight_sizes(net):
    assert net.layers[0].weight.shape == (4, 3)
    assert net.layers[1].weight.shape == (2, 4)


def test_x_sizes(net):
    net.forward(torch.FloatTensor([0.3, -0.2, 0.5]))
    assert [len(_) for _ in net.x] == [3, 4, 2]


def test_all_xs_not_none_after_forward_constrained(net):
    net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )

    for x in net.x:
        assert x is not None


def test_all_xs_change_during_forward_constrained(net):
    # set some starting values for x
    net.forward(torch.FloatTensor([-0.2, 0.3, 0.1]))

    old_x = [_.clone() for _ in net.x]
    net.forward_constrained(
        torch.FloatTensor([-0.1, 0.2, 0.4]), torch.FloatTensor([0.3, -0.4])
    )

    for old, new in zip(old_x, net.x):
        assert not torch.all(torch.isclose(old, new))


def test_all_xs_not_none_after_forward(net):
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for x in net.x:
        assert x is not None


def test_all_xs_change_during_forward(net):
    # set some starting values for x
    net.forward(torch.FloatTensor([-0.2, 0.3, 0.1]))

    old_x = [_.clone() for _ in net.x]
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for old, new in zip(old_x, net.x):
        assert not torch.any(torch.isclose(old, new))


def test_forward_result_is_stationary_point_of_forward_constrained(net):
    x0 = torch.FloatTensor([0.5, -0.7, 0.2])
    net.forward(x0)

    old_x = [_.clone().detach() for _ in net.x]
    net.forward_constrained(old_x[0], old_x[-1])

    for old, new in zip(old_x, net.x):
        assert torch.allclose(old, new)


def test_weights_and_biases_change_when_optimizing_slow_parameters(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    old_Ws = [_.weight.clone().detach() for _ in net.layers]
    old_bs = [_.bias.clone().detach() for _ in net.layers]

    optimizer = torch.optim.Adam(net.slow_parameters(), lr=1.0)
    net.forward_constrained(x0, y0)

    optimizer.zero_grad()
    loss = net.loss()
    loss.backward()
    optimizer.step()

    for old_W, old_b, new in zip(old_Ws, old_bs, net.layers):
        assert not torch.any(torch.isclose(old_W, new.weight))
        assert not torch.any(torch.isclose(old_b, new.bias))


def test_loss_is_nonzero_after_forward_constrained(net):
    x0 = torch.FloatTensor([-0.3, -0.2, 0.6])
    y0 = torch.FloatTensor([0.9, 0.3])
    net.forward_constrained(x0, y0)

    assert net.loss().abs().item() > 1e-6


def test_forward_does_not_change_weights_and_biases(net):
    old_Ws = [_.weight.clone().detach() for _ in net.layers]
    old_bs = [_.bias.clone().detach() for _ in net.layers]
    net.forward(torch.FloatTensor([0.3, -0.4, 0.2]))

    for old_W, old_b, new in zip(old_Ws, old_bs, net.layers):
        assert torch.allclose(old_W, new.weight)
        assert torch.allclose(old_b, new.bias)


def test_forward_constrained_does_not_change_weights_and_biases(net):
    old_Ws = [_.weight.clone().detach() for _ in net.layers]
    old_bs = [_.bias.clone().detach() for _ in net.layers]
    net.forward_constrained(
        torch.FloatTensor([0.3, -0.4, 0.2]), torch.FloatTensor([-0.2, 0.2])
    )

    for old_W, old_b, new in zip(old_Ws, old_bs, net.layers):
        assert torch.allclose(old_W, new.weight)
        assert torch.allclose(old_b, new.bias)


def test_loss_does_not_change_weights_and_biases(net):
    # ensure the x variables have valid values assigned to them
    net.forward(torch.FloatTensor([0.1, 0.2, 0.3]))

    old_Ws = [_.weight.clone().detach() for _ in net.layers]
    old_bs = [_.bias.clone().detach() for _ in net.layers]
    net.loss()

    for old_W, old_b, new in zip(old_Ws, old_bs, net.layers):
        assert torch.allclose(old_W, new.weight)
        assert torch.allclose(old_b, new.bias)


def test_no_nan_or_inf_after_a_few_learning_steps(net):
    torch.manual_seed(0)

    optimizer = torch.optim.Adam(net.slow_parameters())
    for i in range(4):
        x = torch.Tensor(3).uniform_()
        y = torch.Tensor(2).uniform_()
        net.forward_constrained(x, y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    for layer in net.layers:
        assert torch.all(torch.isfinite(layer.weight))
        assert torch.all(torch.isfinite(layer.bias))

    for x in net.x:
        assert torch.all(torch.isfinite(x))


def test_forward_output_depends_on_input(net):
    y1 = net.forward(torch.FloatTensor([0.1, 0.3, -0.2]))
    y2 = net.forward(torch.FloatTensor([-0.5, 0.1, 0.2]))
    assert not torch.allclose(y1, y2)


def test_forward_sets_first_element_of_x_to_input_sample(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    net.forward(x0)
    assert torch.allclose(net.x[0], x0)


def test_forward_constrained_sets_first_element_of_x_to_input_sample(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    y0 = torch.FloatTensor([0.5, -0.2])
    net.forward_constrained(x0, y0)
    assert torch.allclose(net.x[0], x0)


def test_forward_constrained_sets_last_element_of_x_to_output_sample(net):
    x0 = torch.FloatTensor([0.5, 0.2, 0.1])
    y0 = torch.FloatTensor([0.5, -0.2])
    net.forward_constrained(x0, y0)
    assert torch.allclose(net.x[-1], y0)


def test_initialize_values_same_when_torch_seed_is_same():
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = PCNetwork(dims)

    old_Ws = [_.weight.clone().detach() for _ in net.layers]
    old_bs = [_.bias.clone().detach() for _ in net.layers]

    torch.manual_seed(seed)
    net = PCNetwork(dims)

    new_Ws = [_.weight.clone().detach() for _ in net.layers]
    new_bs = [_.bias.clone().detach() for _ in net.layers]

    for old_W, old_b, new_W, new_b in zip(old_Ws, old_bs, new_Ws, new_bs):
        assert torch.allclose(old_W, new_W)
        assert torch.allclose(old_b, new_b)


def test_initialize_weights_change_for_subsequent_calls_if_seed_not_reset():
    seed = 321
    dims = [2, 6, 5, 3]

    torch.manual_seed(seed)
    net = PCNetwork(dims)

    var1 = [_.weight.clone().detach() for _ in net.layers]

    net = PCNetwork(dims)
    var2 = [_.weight.clone().detach() for _ in net.layers]

    for old, new in zip(var1, var2):
        assert not torch.any(torch.isclose(old, new))


def test_weights_reproducible_for_same_seed_after_learning():
    seed = 321
    dims = [2, 6, 5, 3]

    x = torch.FloatTensor([[0.2, -0.3], [0.5, 0.7], [-0.3, 0.2]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.7], [1.5, 0.6, -0.3], [-0.2, 0.5, 0.6]])

    # do some learning
    torch.manual_seed(seed)
    net = PCNetwork(dims)
    optimizer = torch.optim.Adam(net.slow_parameters())
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    var1 = [_.weight.clone().detach() for _ in net.layers]

    # reset and do the learning again
    torch.manual_seed(seed)
    net = PCNetwork(dims)
    optimizer = torch.optim.Adam(net.slow_parameters())
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    var2 = [_.weight.clone().detach() for _ in net.layers]

    for old, new in zip(var1, var2):
        assert torch.allclose(old, new)


def test_learning_effects_are_different_for_subsequent_runs():
    seed = 321
    dims = [2, 6, 5, 3]

    x = torch.FloatTensor([[0.2, -0.3], [0.5, 0.7], [-0.3, 0.2]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.7], [1.5, 0.6, -0.3], [-0.2, 0.5, 0.6]])

    # do some learning
    torch.manual_seed(seed)
    net = PCNetwork(dims)
    optimizer = torch.optim.Adam(net.slow_parameters())
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    var1 = [_.weight.clone().detach() for _ in net.layers]

    # reset and do the learning again -- without resetting random seed this time!
    net = PCNetwork(dims)
    optimizer = torch.optim.Adam(net.slow_parameters())
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    var2 = [_.weight.clone().detach() for _ in net.layers]

    for old, new in zip(var1, var2):
        assert not torch.allclose(old, new)


def test_compare_to_reference_implementation():
    from pcn_ref import PCNetworkRef

    seed = 100
    dims = [2, 6, 5, 3]

    x = torch.FloatTensor([[0.2, -0.3], [0.5, 0.7], [-0.3, 0.2]])
    y = torch.FloatTensor([[-0.5, 0.2, 0.7], [1.5, 0.6, -0.3], [-0.2, 0.5, 0.6]])

    # do some learning
    torch.manual_seed(seed)
    net = PCNetwork(dims)

    weights0 = [_.weight.clone().detach() for _ in net.layers]
    biases0 = [_.bias.clone().detach() for _ in net.layers]

    optimizer = torch.optim.SGD(net.slow_parameters(), lr=0.2)
    for crt_x, crt_y in zip(x, y):
        net.forward_constrained(crt_x, crt_y)

        optimizer.zero_grad()
        net.loss().backward()
        optimizer.step()

    test_x = torch.FloatTensor([0.5, 0.2])
    out = net.forward(test_x)

    net_ref = PCNetworkRef(dims)

    # do some learning
    torch.manual_seed(seed)
    net_ref.reset()

    # ensure matching weights and biases
    for i, (crt_W, crt_b) in enumerate(zip(weights0, biases0)):
        net_ref.W[i][:] = crt_W
        net_ref.b[i][:] = crt_b

    for crt_x, crt_y in zip(x, y):
        net_ref.learn(crt_x, crt_y)

    out_ref = net_ref.forward(test_x)

    assert torch.allclose(out, out_ref)
