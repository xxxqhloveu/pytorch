# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import export, dynamic_dim
from torch._export.trace import do_not_use_experimental_export
from torch._export.constraints import constrain_as_size, constrain_as_value
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestExperimentalExport(TestCase):
    @unittest.skip("TypeError: <lambda>() missing 1 required positional argument")
    def test_export_simple_model_with_attr(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.float_val = float_val

            def forward(self, x):
                y = x + self.float_val
                return y.cos()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        self.assertEqual(exported_program.fw_module(*inp)[0], mod(*inp))

    def test_export_simple_model(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.float_val = float_val

            def forward(self, x):
                return x.cos()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        self.assertEqual(exported_program.fw_module(*inp)[0], mod(*inp))

    @unittest.skip("TypeError: <lambda>() missing 1 required positional argument")
    def test_export_simple_model_buffer_mutation(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.register_buffer("buffer1", torch.ones(6, 1))

            def forward(self, x):
                self.buffer1.add_(2)
                return x.cos() + self.buffer1.sin()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        mutated_buffer, output = exported_program.fw_module(*inp)
        # TODO (tmanlaibaatar) enable this once we figure out
        # how to do buffer mutation
        # self.assertEqual(mutated_buffer.sum().item(), 30)
        self.assertEqual(output, mod(*inp))


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestDynamismExpression(TestCase):
    def test_export_constraints(self):

        def f(x):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        ref = f(*inp)

        gm = export(f, inp)
        res = gm(*inp)

        self.assertTrue(torchdynamo.utils.same(ref, res))

        gm = make_fx(f, tracing_mode="symbolic")(*inp)
        res = gm(*inp)
        self.assertTrue(torchdynamo.utils.same(ref, res))

    def test_export_constraints_error(self):
        def invalid_size(x):
            b = x.item()
            constrain_as_size(b, min=0, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Unable to set min size"):
            export(invalid_size, inp)

        def invalid_input_conflict_with_inline_constraints(x):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([6]),)
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Invalid value 6 for range"):
            export(invalid_input_conflict_with_inline_constraints, inp)

        def invalid_input_conflict_with_input_constraints(x):
            return x + 1

        inp = torch.zeros([3])
        inp_constraints = [
            dynamic_dim(inp, 0) > 5,
        ]
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "not in range"):
            export(
                invalid_input_conflict_with_input_constraints,
                (inp,),
                constraints=inp_constraints,
            )


        def conflicting_constraints(x):
            b = x.item()
            constrain_as_size(b, min=2, max=3)
            constrain_as_size(b, min=4, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)

        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Invalid ranges"):
            export(conflicting_constraints, inp)

    def test_export_assume_static_by_default(self):
        def branch_on_shape(x: torch.Tensor):
            if x.shape[0] == 4:
                return x + 1
            else:
                return x

        inp = (torch.rand(4, 5),)

        # Being able to export means shape is preserved as static
        export(branch_on_shape, inp)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestExport(TestCase):
    def test_basic(self):
        def f(x, y):
            return x[0] + y

        inp = ([torch.ones(1, 3)], torch.ones(1, 3))
        exported_program = export(f, inp)
        self.assertTrue(torch.allclose(exported_program(*inp), f(*inp)))

    def test_raise_user_error_when_guard_on_data_dependent_operation(self):
        def fn_ddo(x):
            y = x.nonzero()
            z = y.shape[0]
            if z > 2:
                return x.cos()
            else:
                return x.sin()

        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            "trying to get a value out of symbolic int"
        ):
            _ = export(fn_ddo, (torch.tensor([2, 3, 5]),), constraints=None)

    def test_if_functional(self):
        def foo(x):
            x.add_(4)
            y = x.view(x.shape)
            return x.cos() + y.cos()

        gm = export(foo, (torch.tensor([2, 3, 5]),), constraints=None)

        view_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add_.Tensor:
                # No more inplace mutation
                self.assertNotEqual(
                    node.target,
                    torch.ops.aten.add_.Tensor,
                    "There shouldn't be any inplace mutation node in the graph."
                )
            if node.op == "call_function" and node.target == torch.ops.aten.view.default:
                view_count += 1

        # There should be nonzero view nodes in the graph
        self.assertTrue(view_count > 0)

    def test_export_constrain_static(self):
        def f(x, y):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            c = y.dim()
            constrain_as_value(c, min=1, max=3)
            z = y[0:c]
            return torch.empty((b, y.shape[0])), z

        x = torch.tensor([3])
        y = torch.randn([8, 8, 6])
        example_inputs = (x, y)
        constraints = [dynamic_dim(y, 0) >= 6, dynamic_dim(y, 0) <= 10]
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError, "It appears that you're trying to set a constraint " +
            "on a value which we evaluated to have a static value of 3. "
        ):
            export(f, example_inputs, constraints)


if __name__ == '__main__':
    run_tests()
