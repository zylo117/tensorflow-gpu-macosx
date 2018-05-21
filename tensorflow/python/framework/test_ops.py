"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: test_ops.cc
"""

import collections as _collections

from tensorflow.python.eager import execute as _execute
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import tensor_shape as _tensor_shape

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.tf_export import tf_export


@tf_export('A')
def a(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "A", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"A", 1, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  _execute.record_gradient(
      "A", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("A")(None)


@tf_export('B')
def b(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "B", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"B", 1, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  _execute.record_gradient(
      "B", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("B")(None)


@tf_export('CopyOp')
def copy_op(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CopyOp", a=a, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, (a,) = _execute.args_to_matching_eager([a], _ctx)
    _inputs_flat = [a]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"CopyOp", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "CopyOp", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("CopyOp")(None)


@tf_export('DefaultAttrs')
def default_attrs(string_val="abc", string_list_val=["abc", ""], int_val=123, int_list_val=[1, 2, 3], float_val=10, float_list_val=[10], bool_val=True, bool_list_val=[True, False], type_val=_dtypes.int32, type_list_val=[_dtypes.int32, _dtypes.float32], shape_val=[2, 1], shape_list_val=[[], [1]], tensor_val=_execute.make_tensor("""dtype: DT_INT32 tensor_shape { } int_val: 1""", "tensor_val"), tensor_list_val=[_execute.make_tensor(_pb, "tensor_list_val") for _pb in ("""dtype: DT_INT32 tensor_shape { } int_val: 1""",)], name=None):
  r"""TODO: add doc.

  Args:
    string_val: An optional `string`. Defaults to `"abc"`.
    string_list_val: An optional list of `strings`. Defaults to `["abc", ""]`.
    int_val: An optional `int`. Defaults to `123`.
    int_list_val: An optional list of `ints`. Defaults to `[1, 2, 3]`.
    float_val: An optional `float`. Defaults to `10`.
    float_list_val: An optional list of `floats`. Defaults to `[10]`.
    bool_val: An optional `bool`. Defaults to `True`.
    bool_list_val: An optional list of `bools`. Defaults to `[True, False]`.
    type_val: An optional `tf.DType`. Defaults to `tf.int32`.
    type_list_val: An optional list of `tf.DTypes`. Defaults to `[tf.int32, tf.float32]`.
    shape_val: An optional `tf.TensorShape` or list of `ints`. Defaults to `[2, 1]`.
    shape_list_val: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[[], [1]]`.
    tensor_val: An optional `tf.TensorProto`. Defaults to `dtype: DT_INT32 tensor_shape { } int_val: 1`.
    tensor_list_val: An optional list of `tf.TensorProto` objects. Defaults to `[dtype: DT_INT32 tensor_shape { } int_val: 1]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  if string_val is None:
    string_val = "abc"
  string_val = _execute.make_str(string_val, "string_val")
  if string_list_val is None:
    string_list_val = ["abc", ""]
  if not isinstance(string_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'string_list_val' argument to "
        "'default_attrs' Op, not %r." % string_list_val)
  string_list_val = [_execute.make_str(_s, "string_list_val") for _s in string_list_val]
  if int_val is None:
    int_val = 123
  int_val = _execute.make_int(int_val, "int_val")
  if int_list_val is None:
    int_list_val = [1, 2, 3]
  if not isinstance(int_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'int_list_val' argument to "
        "'default_attrs' Op, not %r." % int_list_val)
  int_list_val = [_execute.make_int(_i, "int_list_val") for _i in int_list_val]
  if float_val is None:
    float_val = 10
  float_val = _execute.make_float(float_val, "float_val")
  if float_list_val is None:
    float_list_val = [10]
  if not isinstance(float_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'float_list_val' argument to "
        "'default_attrs' Op, not %r." % float_list_val)
  float_list_val = [_execute.make_float(_f, "float_list_val") for _f in float_list_val]
  if bool_val is None:
    bool_val = True
  bool_val = _execute.make_bool(bool_val, "bool_val")
  if bool_list_val is None:
    bool_list_val = [True, False]
  if not isinstance(bool_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'bool_list_val' argument to "
        "'default_attrs' Op, not %r." % bool_list_val)
  bool_list_val = [_execute.make_bool(_b, "bool_list_val") for _b in bool_list_val]
  if type_val is None:
    type_val = _dtypes.int32
  type_val = _execute.make_type(type_val, "type_val")
  if type_list_val is None:
    type_list_val = [_dtypes.int32, _dtypes.float32]
  if not isinstance(type_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'type_list_val' argument to "
        "'default_attrs' Op, not %r." % type_list_val)
  type_list_val = [_execute.make_type(_t, "type_list_val") for _t in type_list_val]
  if shape_val is None:
    shape_val = [2, 1]
  shape_val = _execute.make_shape(shape_val, "shape_val")
  if shape_list_val is None:
    shape_list_val = [[], [1]]
  if not isinstance(shape_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'shape_list_val' argument to "
        "'default_attrs' Op, not %r." % shape_list_val)
  shape_list_val = [_execute.make_shape(_s, "shape_list_val") for _s in shape_list_val]
  if tensor_val is None:
    tensor_val = _execute.make_tensor("""dtype: DT_INT32 tensor_shape { } int_val: 1""", "tensor_val")
  tensor_val = _execute.make_tensor(tensor_val, "tensor_val")
  if tensor_list_val is None:
    tensor_list_val = [_execute.make_tensor(_pb, "tensor_list_val") for _pb in ("""dtype: DT_INT32 tensor_shape { } int_val: 1""",)]
  if not isinstance(tensor_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'tensor_list_val' argument to "
        "'default_attrs' Op, not %r." % tensor_list_val)
  tensor_list_val = [_execute.make_tensor(_t, "tensor_list_val") for _t in tensor_list_val]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "DefaultAttrs", string_val=string_val,
        string_list_val=string_list_val, int_val=int_val,
        int_list_val=int_list_val, float_val=float_val,
        float_list_val=float_list_val, bool_val=bool_val,
        bool_list_val=bool_list_val, type_val=type_val,
        type_list_val=type_list_val, shape_val=shape_val,
        shape_list_val=shape_list_val, tensor_val=tensor_val,
        tensor_list_val=tensor_list_val, name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = ("string_val", string_val, "string_list_val", string_list_val,
              "int_val", int_val, "int_list_val", int_list_val, "float_val",
              float_val, "float_list_val", float_list_val, "bool_val",
              bool_val, "bool_list_val", bool_list_val, "type_val", type_val,
              "type_list_val", type_list_val, "shape_val", shape_val,
              "shape_list_val", shape_list_val, "tensor_val", tensor_val,
              "tensor_list_val", tensor_list_val)
    _result = _execute.execute(b"DefaultAttrs", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("DefaultAttrs")(None)


_five_float_outputs_outputs = ["a", "b", "c", "d", "e"]
_FiveFloatOutputsOutput = _collections.namedtuple(
    "FiveFloatOutputs", _five_float_outputs_outputs)


@tf_export('FiveFloatOutputs')
def five_float_outputs(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b, c, d, e).

    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
    c: A `Tensor` of type `float32`.
    d: A `Tensor` of type `float32`.
    e: A `Tensor` of type `float32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FiveFloatOutputs", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"FiveFloatOutputs", 5, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "FiveFloatOutputs", _inputs_flat, _attrs, _result, name)
  _result = _FiveFloatOutputsOutput._make(_result)
  return _result

_ops.RegisterShape("FiveFloatOutputs")(None)


@tf_export('FloatInput')
def float_input(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FloatInput", a=a, name=name)
    return _op
  else:
    a = _ops.convert_to_tensor(a, _dtypes.float32)
    _inputs_flat = [a]
    _attrs = None
    _result = _execute.execute(b"FloatInput", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("FloatInput")(None)


@tf_export('FloatOutput')
def float_output(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FloatOutput", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"FloatOutput", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "FloatOutput", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("FloatOutput")(None)


_float_output_string_output_outputs = ["a", "b"]
_FloatOutputStringOutputOutput = _collections.namedtuple(
    "FloatOutputStringOutput", _float_output_string_output_outputs)


@tf_export('FloatOutputStringOutput')
def float_output_string_output(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `string`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FloatOutputStringOutput", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"FloatOutputStringOutput", 2,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "FloatOutputStringOutput", _inputs_flat, _attrs, _result, name)
  _result = _FloatOutputStringOutputOutput._make(_result)
  return _result

_ops.RegisterShape("FloatOutputStringOutput")(None)


_foo1_outputs = ["d", "e"]
_Foo1Output = _collections.namedtuple(
    "Foo1", _foo1_outputs)


@tf_export('Foo1')
def foo1(a, b, c, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `int32`.
    c: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d, e).

    d: A `Tensor` of type `float32`.
    e: A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Foo1", a=a, b=b, c=c, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    a = _ops.convert_to_tensor(a, _dtypes.float32)
    b = _ops.convert_to_tensor(b, _dtypes.int32)
    c = _ops.convert_to_tensor(c, _dtypes.int32)
    _inputs_flat = [a, b, c]
    _attrs = None
    _result = _execute.execute(b"Foo1", 2, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  _execute.record_gradient(
      "Foo1", _inputs_flat, _attrs, _result, name)
  _result = _Foo1Output._make(_result)
  return _result

_ops.RegisterShape("Foo1")(None)


_foo2_outputs = ["d", "e"]
_Foo2Output = _collections.namedtuple(
    "Foo2", _foo2_outputs)


@tf_export('Foo2')
def foo2(a, b, c, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `string`.
    c: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d, e).

    d: A `Tensor` of type `float32`.
    e: A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Foo2", a=a, b=b, c=c, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    a = _ops.convert_to_tensor(a, _dtypes.float32)
    b = _ops.convert_to_tensor(b, _dtypes.string)
    c = _ops.convert_to_tensor(c, _dtypes.string)
    _inputs_flat = [a, b, c]
    _attrs = None
    _result = _execute.execute(b"Foo2", 2, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  _execute.record_gradient(
      "Foo2", _inputs_flat, _attrs, _result, name)
  _result = _Foo2Output._make(_result)
  return _result

_ops.RegisterShape("Foo2")(None)


_foo3_outputs = ["d", "e"]
_Foo3Output = _collections.namedtuple(
    "Foo3", _foo3_outputs)


@tf_export('Foo3')
def foo3(a, b, c, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `string`.
    c: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d, e).

    d: A `Tensor` of type `float32`.
    e: A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Foo3", a=a, b=b, c=c, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    a = _ops.convert_to_tensor(a, _dtypes.float32)
    b = _ops.convert_to_tensor(b, _dtypes.string)
    c = _ops.convert_to_tensor(c, _dtypes.float32)
    _inputs_flat = [a, b, c]
    _attrs = None
    _result = _execute.execute(b"Foo3", 2, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  _execute.record_gradient(
      "Foo3", _inputs_flat, _attrs, _result, name)
  _result = _Foo3Output._make(_result)
  return _result

_ops.RegisterShape("Foo3")(None)


@tf_export('FuncAttr')
def func_attr(f, name=None):
  r"""TODO: add doc.

  Args:
    f: A function decorated with @Defun.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FuncAttr", f=f, name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = ("f", f)
    _result = _execute.execute(b"FuncAttr", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("FuncAttr")(None)


@tf_export('GraphDefVersion')
def graph_def_version(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "GraphDefVersion", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"GraphDefVersion", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "GraphDefVersion", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("GraphDefVersion")(None)


@tf_export('Int64Output')
def int64_output(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Int64Output", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"Int64Output", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "Int64Output", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("Int64Output")(None)


@tf_export('IntAttr')
def int_attr(foo=1, name=None):
  r"""TODO: add doc.

  Args:
    foo: An optional `int`. Defaults to `1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  if foo is None:
    foo = 1
  foo = _execute.make_int(foo, "foo")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "IntAttr", foo=foo, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("foo", _op.get_attr("foo"))
  else:
    _inputs_flat = []
    _attrs = ("foo", foo)
    _result = _execute.execute(b"IntAttr", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "IntAttr", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("IntAttr")(None)


@tf_export('IntInput')
def int_input(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "IntInput", a=a, name=name)
    return _op
  else:
    a = _ops.convert_to_tensor(a, _dtypes.int32)
    _inputs_flat = [a]
    _attrs = None
    _result = _execute.execute(b"IntInput", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("IntInput")(None)


@tf_export('IntInputFloatInput')
def int_input_float_input(a, b, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `int32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "IntInputFloatInput", a=a, b=b, name=name)
    return _op
  else:
    a = _ops.convert_to_tensor(a, _dtypes.int32)
    b = _ops.convert_to_tensor(b, _dtypes.float32)
    _inputs_flat = [a, b]
    _attrs = None
    _result = _execute.execute(b"IntInputFloatInput", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("IntInputFloatInput")(None)


@tf_export('IntInputIntOutput')
def int_input_int_output(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "IntInputIntOutput", a=a, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    a = _ops.convert_to_tensor(a, _dtypes.int32)
    _inputs_flat = [a]
    _attrs = None
    _result = _execute.execute(b"IntInputIntOutput", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "IntInputIntOutput", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("IntInputIntOutput")(None)


@tf_export('IntOutput')
def int_output(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "IntOutput", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"IntOutput", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "IntOutput", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("IntOutput")(None)


_int_output_float_output_outputs = ["a", "b"]
_IntOutputFloatOutputOutput = _collections.namedtuple(
    "IntOutputFloatOutput", _int_output_float_output_outputs)


@tf_export('IntOutputFloatOutput')
def int_output_float_output(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A `Tensor` of type `int32`.
    b: A `Tensor` of type `float32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "IntOutputFloatOutput", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"IntOutputFloatOutput", 2,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "IntOutputFloatOutput", _inputs_flat, _attrs, _result, name)
  _result = _IntOutputFloatOutputOutput._make(_result)
  return _result

_ops.RegisterShape("IntOutputFloatOutput")(None)


@tf_export('KernelLabel')
def kernel_label(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "KernelLabel", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"KernelLabel", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "KernelLabel", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("KernelLabel")(None)


@tf_export('KernelLabelRequired')
def kernel_label_required(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "KernelLabelRequired", input=input, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    input = _ops.convert_to_tensor(input, _dtypes.int32)
    _inputs_flat = [input]
    _attrs = None
    _result = _execute.execute(b"KernelLabelRequired", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "KernelLabelRequired", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("KernelLabelRequired")(None)


@tf_export('ListInput')
def list_input(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of at least 1 `Tensor` objects with the same type.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'list_input' Op, not %r." % a)
  _attr_N = len(a)
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ListInput", a=a, name=name)
    return _op
  else:
    _attr_T, a = _execute.args_to_matching_eager(list(a), _ctx)
    _inputs_flat = list(a)
    _attrs = ("N", _attr_N, "T", _attr_T)
    _result = _execute.execute(b"ListInput", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("ListInput")(None)


@tf_export('ListOutput')
def list_output(T, name=None):
  r"""TODO: add doc.

  Args:
    T: A list of `tf.DTypes` that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `T`.
  """
  if not isinstance(T, (list, tuple)):
    raise TypeError(
        "Expected list for 'T' argument to "
        "'list_output' Op, not %r." % T)
  T = [_execute.make_type(_t, "T") for _t in T]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ListOutput", T=T, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _inputs_flat = []
    _attrs = ("T", T)
    _result = _execute.execute(b"ListOutput", len(T), inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ListOutput", _inputs_flat, _attrs, _result, name)
  return _result

_ops.RegisterShape("ListOutput")(None)


@tf_export('None')
def none(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "None", name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"None", 0, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("None")(None)


@tf_export('Old')
def old(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Old", name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"Old", 0, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("Old")(None)


@tf_export('OpWithDefaultAttr')
def op_with_default_attr(default_float=123, name=None):
  r"""TODO: add doc.

  Args:
    default_float: An optional `float`. Defaults to `123`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  if default_float is None:
    default_float = 123
  default_float = _execute.make_float(default_float, "default_float")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "OpWithDefaultAttr", default_float=default_float, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("default_float", _op.get_attr("default_float"))
  else:
    _inputs_flat = []
    _attrs = ("default_float", default_float)
    _result = _execute.execute(b"OpWithDefaultAttr", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "OpWithDefaultAttr", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("OpWithDefaultAttr")(None)


@tf_export('OpWithFutureDefaultAttr')
def op_with_future_default_attr(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "OpWithFutureDefaultAttr", name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"OpWithFutureDefaultAttr", 0,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
    _result = None
  return _result

_ops.RegisterShape("OpWithFutureDefaultAttr")(None)


@tf_export('RefInputFloatInput')
def ref_input_float_input(a, b, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type mutable `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefInputFloatInput", a=a, b=b, name=name)
    return _op
  else:
    raise RuntimeError(
        "ref_input_float_input op does not support eager execution. Arg 'a'' is a ref.")
    _result = None
  return _result

_ops.RegisterShape("RefInputFloatInput")(None)


@tf_export('RefInputFloatInputIntOutput')
def ref_input_float_input_int_output(a, b, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type mutable `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefInputFloatInputIntOutput", a=a, b=b, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    raise RuntimeError(
        "ref_input_float_input_int_output op does not support eager execution. Arg 'a'' is a ref.")
  _execute.record_gradient(
      "RefInputFloatInputIntOutput", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("RefInputFloatInputIntOutput")(None)


@tf_export('RefInputIntInput')
def ref_input_int_input(a, b, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type mutable `int32`.
    b: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefInputIntInput", a=a, b=b, name=name)
    return _op
  else:
    raise RuntimeError(
        "ref_input_int_input op does not support eager execution. Arg 'a'' is a ref.")
    _result = None
  return _result

_ops.RegisterShape("RefInputIntInput")(None)


@tf_export('RefOutput')
def ref_output(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefOutput", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    raise RuntimeError(
        "ref_output op does not support eager execution. Arg 'a'' is a ref.")
  _execute.record_gradient(
      "RefOutput", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("RefOutput")(None)


_ref_output_float_output_outputs = ["a", "b"]
_RefOutputFloatOutputOutput = _collections.namedtuple(
    "RefOutputFloatOutput", _ref_output_float_output_outputs)


@tf_export('RefOutputFloatOutput')
def ref_output_float_output(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A `Tensor` of type mutable `float32`.
    b: A `Tensor` of type `float32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefOutputFloatOutput", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    raise RuntimeError(
        "ref_output_float_output op does not support eager execution. Arg 'a'' is a ref.")
  _execute.record_gradient(
      "RefOutputFloatOutput", _inputs_flat, _attrs, _result, name)
  _result = _RefOutputFloatOutputOutput._make(_result)
  return _result

_ops.RegisterShape("RefOutputFloatOutput")(None)


@tf_export('RequiresOlderGraphVersion')
def requires_older_graph_version(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RequiresOlderGraphVersion", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"RequiresOlderGraphVersion", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "RequiresOlderGraphVersion", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("RequiresOlderGraphVersion")(None)


@tf_export('ResourceCreateOp')
def resource_create_op(resource, name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ResourceCreateOp", resource=resource, name=name)
    return _op
  else:
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource]
    _attrs = None
    _result = _execute.execute(b"ResourceCreateOp", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("ResourceCreateOp")(None)


@tf_export('ResourceInitializedOp')
def resource_initialized_op(resource, name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ResourceInitializedOp", resource=resource, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource]
    _attrs = None
    _result = _execute.execute(b"ResourceInitializedOp", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "ResourceInitializedOp", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("ResourceInitializedOp")(None)


@tf_export('ResourceUsingOp')
def resource_using_op(resource, name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ResourceUsingOp", resource=resource, name=name)
    return _op
  else:
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource]
    _attrs = None
    _result = _execute.execute(b"ResourceUsingOp", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("ResourceUsingOp")(None)


@tf_export('StringListAttr')
def string_list_attr(a, b, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `strings`.
    b: A `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'string_list_attr' Op, not %r." % a)
  a = [_execute.make_str(_s, "a") for _s in a]
  b = _execute.make_str(b, "b")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StringListAttr", a=a, b=b, name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = ("a", a, "b", b)
    _result = _execute.execute(b"StringListAttr", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("StringListAttr")(None)


@tf_export('StubResourceHandleOp')
def stub_resource_handle_op(container="", shared_name="", name=None):
  r"""Creates a handle to a StubResource

  Args:
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StubResourceHandleOp", container=container, shared_name=shared_name,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
  else:
    _inputs_flat = []
    _attrs = ("container", container, "shared_name", shared_name)
    _result = _execute.execute(b"StubResourceHandleOp", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "StubResourceHandleOp", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("StubResourceHandleOp")(None)


_test_string_output_outputs = ["output1", "output2"]
_TestStringOutputOutput = _collections.namedtuple(
    "TestStringOutput", _test_string_output_outputs)


@tf_export('TestStringOutput')
def test_string_output(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output1, output2).

    output1: A `Tensor` of type `float32`.
    output2: A `Tensor` of type `string`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TestStringOutput", input=input, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    input = _ops.convert_to_tensor(input, _dtypes.float32)
    _inputs_flat = [input]
    _attrs = None
    _result = _execute.execute(b"TestStringOutput", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TestStringOutput", _inputs_flat, _attrs, _result, name)
  _result = _TestStringOutputOutput._make(_result)
  return _result

_ops.RegisterShape("TestStringOutput")(None)


@tf_export('TwoFloatInputs')
def two_float_inputs(a, b, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TwoFloatInputs", a=a, b=b, name=name)
    return _op
  else:
    a = _ops.convert_to_tensor(a, _dtypes.float32)
    b = _ops.convert_to_tensor(b, _dtypes.float32)
    _inputs_flat = [a, b]
    _attrs = None
    _result = _execute.execute(b"TwoFloatInputs", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("TwoFloatInputs")(None)


@tf_export('TwoFloatInputsFloatOutput')
def two_float_inputs_float_output(a, b, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TwoFloatInputsFloatOutput", a=a, b=b, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    a = _ops.convert_to_tensor(a, _dtypes.float32)
    b = _ops.convert_to_tensor(b, _dtypes.float32)
    _inputs_flat = [a, b]
    _attrs = None
    _result = _execute.execute(b"TwoFloatInputsFloatOutput", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "TwoFloatInputsFloatOutput", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("TwoFloatInputsFloatOutput")(None)


@tf_export('TwoFloatInputsIntOutput')
def two_float_inputs_int_output(a, b, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TwoFloatInputsIntOutput", a=a, b=b, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    a = _ops.convert_to_tensor(a, _dtypes.float32)
    b = _ops.convert_to_tensor(b, _dtypes.float32)
    _inputs_flat = [a, b]
    _attrs = None
    _result = _execute.execute(b"TwoFloatInputsIntOutput", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "TwoFloatInputsIntOutput", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("TwoFloatInputsIntOutput")(None)


_two_float_outputs_outputs = ["a", "b"]
_TwoFloatOutputsOutput = _collections.namedtuple(
    "TwoFloatOutputs", _two_float_outputs_outputs)


@tf_export('TwoFloatOutputs')
def two_float_outputs(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TwoFloatOutputs", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"TwoFloatOutputs", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TwoFloatOutputs", _inputs_flat, _attrs, _result, name)
  _result = _TwoFloatOutputsOutput._make(_result)
  return _result

_ops.RegisterShape("TwoFloatOutputs")(None)


@tf_export('TwoIntInputs')
def two_int_inputs(a, b, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `int32`.
    b: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TwoIntInputs", a=a, b=b, name=name)
    return _op
  else:
    a = _ops.convert_to_tensor(a, _dtypes.int32)
    b = _ops.convert_to_tensor(b, _dtypes.int32)
    _inputs_flat = [a, b]
    _attrs = None
    _result = _execute.execute(b"TwoIntInputs", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
    _result = None
  return _result

_ops.RegisterShape("TwoIntInputs")(None)


_two_int_outputs_outputs = ["a", "b"]
_TwoIntOutputsOutput = _collections.namedtuple(
    "TwoIntOutputs", _two_int_outputs_outputs)


@tf_export('TwoIntOutputs')
def two_int_outputs(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A `Tensor` of type `int32`.
    b: A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TwoIntOutputs", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"TwoIntOutputs", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TwoIntOutputs", _inputs_flat, _attrs, _result, name)
  _result = _TwoIntOutputsOutput._make(_result)
  return _result

_ops.RegisterShape("TwoIntOutputs")(None)


@tf_export('Unary')
def unary(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Unary", a=a, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, (a,) = _execute.args_to_matching_eager([a], _ctx)
    _inputs_flat = [a]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"Unary", 1, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  _execute.record_gradient(
      "Unary", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("Unary")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "A"
#   output_arg {
#     name: "out"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "B"
#   output_arg {
#     name: "out"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "CopyOp"
#   input_arg {
#     name: "a"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "b"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
# }
# op {
#   name: "DefaultAttrs"
#   attr {
#     name: "string_val"
#     type: "string"
#     default_value {
#       s: "abc"
#     }
#   }
#   attr {
#     name: "string_list_val"
#     type: "list(string)"
#     default_value {
#       list {
#         s: "abc"
#         s: ""
#       }
#     }
#   }
#   attr {
#     name: "int_val"
#     type: "int"
#     default_value {
#       i: 123
#     }
#   }
#   attr {
#     name: "int_list_val"
#     type: "list(int)"
#     default_value {
#       list {
#         i: 1
#         i: 2
#         i: 3
#       }
#     }
#   }
#   attr {
#     name: "float_val"
#     type: "float"
#     default_value {
#       f: 10
#     }
#   }
#   attr {
#     name: "float_list_val"
#     type: "list(float)"
#     default_value {
#       list {
#         f: 10
#       }
#     }
#   }
#   attr {
#     name: "bool_val"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
#   attr {
#     name: "bool_list_val"
#     type: "list(bool)"
#     default_value {
#       list {
#         b: true
#         b: false
#       }
#     }
#   }
#   attr {
#     name: "type_val"
#     type: "type"
#     default_value {
#       type: DT_INT32
#     }
#   }
#   attr {
#     name: "type_list_val"
#     type: "list(type)"
#     default_value {
#       list {
#         type: DT_INT32
#         type: DT_FLOAT
#       }
#     }
#   }
#   attr {
#     name: "shape_val"
#     type: "shape"
#     default_value {
#       shape {
#         dim {
#           size: 2
#         }
#         dim {
#           size: 1
#         }
#       }
#     }
#   }
#   attr {
#     name: "shape_list_val"
#     type: "list(shape)"
#     default_value {
#       list {
#         shape {
#         }
#         shape {
#           dim {
#             size: 1
#           }
#         }
#       }
#     }
#   }
#   attr {
#     name: "tensor_val"
#     type: "tensor"
#     default_value {
#       tensor {
#         dtype: DT_INT32
#         tensor_shape {
#         }
#         int_val: 1
#       }
#     }
#   }
#   attr {
#     name: "tensor_list_val"
#     type: "list(tensor)"
#     default_value {
#       list {
#         tensor {
#           dtype: DT_INT32
#           tensor_shape {
#           }
#           int_val: 1
#         }
#       }
#     }
#   }
# }
# op {
#   name: "FiveFloatOutputs"
#   output_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "b"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "c"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "d"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "e"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "FloatInput"
#   input_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "FloatOutput"
#   output_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "FloatOutputStringOutput"
#   output_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "b"
#     type: DT_STRING
#   }
# }
# op {
#   name: "Foo1"
#   input_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "b"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "c"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "d"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "e"
#     type: DT_INT32
#   }
# }
# op {
#   name: "Foo2"
#   input_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "b"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "c"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "d"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "e"
#     type: DT_INT32
#   }
# }
# op {
#   name: "Foo3"
#   input_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "b"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "c"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "d"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "e"
#     type: DT_INT32
#   }
# }
# op {
#   name: "FuncAttr"
#   attr {
#     name: "f"
#     type: "func"
#   }
# }
# op {
#   name: "GraphDefVersion"
#   output_arg {
#     name: "version"
#     type: DT_INT32
#   }
#   is_stateful: true
# }
# op {
#   name: "Int64Output"
#   output_arg {
#     name: "out"
#     type: DT_INT64
#   }
# }
# op {
#   name: "IntAttr"
#   output_arg {
#     name: "out"
#     type: DT_INT64
#   }
#   attr {
#     name: "foo"
#     type: "int"
#     default_value {
#       i: 1
#     }
#   }
# }
# op {
#   name: "IntInput"
#   input_arg {
#     name: "a"
#     type: DT_INT32
#   }
# }
# op {
#   name: "IntInputFloatInput"
#   input_arg {
#     name: "a"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "b"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "IntInputIntOutput"
#   input_arg {
#     name: "a"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "b"
#     type: DT_INT32
#   }
# }
# op {
#   name: "IntOutput"
#   output_arg {
#     name: "a"
#     type: DT_INT32
#   }
# }
# op {
#   name: "IntOutputFloatOutput"
#   output_arg {
#     name: "a"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "b"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "KernelLabel"
#   output_arg {
#     name: "result"
#     type: DT_STRING
#   }
# }
# op {
#   name: "KernelLabelRequired"
#   input_arg {
#     name: "input"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "result"
#     type: DT_STRING
#   }
# }
# op {
#   name: "ListInput"
#   input_arg {
#     name: "a"
#     type_attr: "T"
#     number_attr: "N"
#   }
#   attr {
#     name: "N"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
# }
# op {
#   name: "ListOutput"
#   output_arg {
#     name: "a"
#     type_list_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "None"
# }
# op {
#   name: "Old"
#   deprecation {
#     version: 8
#     explanation: "For reasons"
#   }
# }
# op {
#   name: "OpWithDefaultAttr"
#   output_arg {
#     name: "a"
#     type: DT_INT32
#   }
#   attr {
#     name: "default_float"
#     type: "float"
#     default_value {
#       f: 123
#     }
#   }
# }
# op {
#   name: "OpWithFutureDefaultAttr"
# }
# op {
#   name: "RefInputFloatInput"
#   input_arg {
#     name: "a"
#     type: DT_FLOAT
#     is_ref: true
#   }
#   input_arg {
#     name: "b"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "RefInputFloatInputIntOutput"
#   input_arg {
#     name: "a"
#     type: DT_FLOAT
#     is_ref: true
#   }
#   input_arg {
#     name: "b"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "c"
#     type: DT_INT32
#   }
# }
# op {
#   name: "RefInputIntInput"
#   input_arg {
#     name: "a"
#     type: DT_INT32
#     is_ref: true
#   }
#   input_arg {
#     name: "b"
#     type: DT_INT32
#   }
# }
# op {
#   name: "RefOutput"
#   output_arg {
#     name: "a"
#     type: DT_INT32
#     is_ref: true
#   }
# }
# op {
#   name: "RefOutputFloatOutput"
#   output_arg {
#     name: "a"
#     type: DT_FLOAT
#     is_ref: true
#   }
#   output_arg {
#     name: "b"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "RequiresOlderGraphVersion"
#   output_arg {
#     name: "version"
#     type: DT_INT32
#   }
#   is_stateful: true
# }
# op {
#   name: "ResourceCreateOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "ResourceInitializedOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "initialized"
#     type: DT_BOOL
#   }
#   is_stateful: true
# }
# op {
#   name: "ResourceUsingOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "StringListAttr"
#   attr {
#     name: "a"
#     type: "list(string)"
#   }
#   attr {
#     name: "b"
#     type: "string"
#   }
# }
# op {
#   name: "StubResourceHandleOp"
#   output_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "container"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   attr {
#     name: "shared_name"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "TestStringOutput"
#   input_arg {
#     name: "input"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "output1"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "output2"
#     type: DT_STRING
#   }
# }
# op {
#   name: "TwoFloatInputs"
#   input_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "b"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "TwoFloatInputsFloatOutput"
#   input_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "b"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "c"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "TwoFloatInputsIntOutput"
#   input_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "b"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "c"
#     type: DT_INT32
#   }
# }
# op {
#   name: "TwoFloatOutputs"
#   output_arg {
#     name: "a"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "b"
#     type: DT_FLOAT
#   }
# }
# op {
#   name: "TwoIntInputs"
#   input_arg {
#     name: "a"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "b"
#     type: DT_INT32
#   }
# }
# op {
#   name: "TwoIntOutputs"
#   output_arg {
#     name: "a"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "b"
#     type: DT_INT32
#   }
# }
# op {
#   name: "Unary"
#   input_arg {
#     name: "a"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "b"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\014\n\001A\032\007\n\003out\030\001\n\014\n\001B\032\007\n\003out\030\001\n#\n\006CopyOp\022\006\n\001a\"\001T\032\006\n\001b\"\001T\"\t\n\001T\022\004type\n\343\003\n\014DefaultAttrs\"\033\n\nstring_val\022\006string\032\005\022\003abc\"*\n\017string_list_val\022\014list(string)\032\t\n\007\022\003abc\022\000\"\022\n\007int_val\022\003int\032\002\030{\"\"\n\014int_list_val\022\tlist(int)\032\007\n\005\032\003\001\002\003\"\031\n\tfloat_val\022\005float\032\005%\000\000 A\"\'\n\016float_list_val\022\013list(float)\032\010\n\006\"\004\000\000 A\"\024\n\010bool_val\022\004bool\032\002(\001\"#\n\rbool_list_val\022\nlist(bool)\032\006\n\004*\002\001\000\"\024\n\010type_val\022\004type\032\0020\003\"#\n\rtype_list_val\022\nlist(type)\032\006\n\0042\002\003\001\"\036\n\tshape_val\022\005shape\032\n:\010\022\002\010\002\022\002\010\001\")\n\016shape_list_val\022\013list(shape)\032\n\n\010:\000:\004\022\002\010\001\"\037\n\ntensor_val\022\006tensor\032\tB\007\010\003\022\000:\001\001\",\n\017tensor_list_val\022\014list(tensor)\032\013\n\tB\007\010\003\022\000:\001\001\n5\n\020FiveFloatOutputs\032\005\n\001a\030\001\032\005\n\001b\030\001\032\005\n\001c\030\001\032\005\n\001d\030\001\032\005\n\001e\030\001\n\023\n\nFloatInput\022\005\n\001a\030\001\n\024\n\013FloatOutput\032\005\n\001a\030\001\n\'\n\027FloatOutputStringOutput\032\005\n\001a\030\001\032\005\n\001b\030\007\n)\n\004Foo1\022\005\n\001a\030\001\022\005\n\001b\030\003\022\005\n\001c\030\003\032\005\n\001d\030\001\032\005\n\001e\030\003\n)\n\004Foo2\022\005\n\001a\030\001\022\005\n\001b\030\007\022\005\n\001c\030\007\032\005\n\001d\030\001\032\005\n\001e\030\003\n)\n\004Foo3\022\005\n\001a\030\001\022\005\n\001b\030\007\022\005\n\001c\030\001\032\005\n\001d\030\001\032\005\n\001e\030\003\n\025\n\010FuncAttr\"\t\n\001f\022\004func\n!\n\017GraphDefVersion\032\013\n\007version\030\003\210\001\001\n\026\n\013Int64Output\032\007\n\003out\030\t\n\"\n\007IntAttr\032\007\n\003out\030\t\"\016\n\003foo\022\003int\032\002\030\001\n\021\n\010IntInput\022\005\n\001a\030\003\n\"\n\022IntInputFloatInput\022\005\n\001a\030\003\022\005\n\001b\030\001\n!\n\021IntInputIntOutput\022\005\n\001a\030\003\032\005\n\001b\030\003\n\022\n\tIntOutput\032\005\n\001a\030\003\n$\n\024IntOutputFloatOutput\032\005\n\001a\030\003\032\005\n\001b\030\001\n\031\n\013KernelLabel\032\n\n\006result\030\007\n,\n\023KernelLabelRequired\022\t\n\005input\030\003\032\n\n\006result\030\007\n/\n\tListInput\022\t\n\001a\"\001T*\001N\"\014\n\001N\022\003int(\0010\001\"\t\n\001T\022\004type\n)\n\nListOutput\032\006\n\001a2\001T\"\023\n\001T\022\nlist(type)(\0010\001\n\006\n\004None\n\026\n\003OldB\017\010\010\022\013For reasons\n9\n\021OpWithDefaultAttr\032\005\n\001a\030\003\"\035\n\rdefault_float\022\005float\032\005%\000\000\366B\n\031\n\027OpWithFutureDefaultAttr\n%\n\022RefInputFloatInput\022\010\n\001a\030\001\200\001\001\022\005\n\001b\030\001\n5\n\033RefInputFloatInputIntOutput\022\010\n\001a\030\001\200\001\001\022\005\n\001b\030\001\032\005\n\001c\030\003\n#\n\020RefInputIntInput\022\010\n\001a\030\003\200\001\001\022\005\n\001b\030\003\n\025\n\tRefOutput\032\010\n\001a\030\003\200\001\001\n\'\n\024RefOutputFloatOutput\032\010\n\001a\030\001\200\001\001\032\005\n\001b\030\001\n+\n\031RequiresOlderGraphVersion\032\013\n\007version\030\003\210\001\001\n#\n\020ResourceCreateOp\022\014\n\010resource\030\024\210\001\001\n9\n\025ResourceInitializedOp\022\014\n\010resource\030\024\032\017\n\013initialized\030\n\210\001\001\n\"\n\017ResourceUsingOp\022\014\n\010resource\030\024\210\001\001\n0\n\016StringListAttr\"\021\n\001a\022\014list(string)\"\013\n\001b\022\006string\n[\n\024StubResourceHandleOp\032\014\n\010resource\030\024\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n7\n\020TestStringOutput\022\t\n\005input\030\001\032\013\n\007output1\030\001\032\013\n\007output2\030\007\n\036\n\016TwoFloatInputs\022\005\n\001a\030\001\022\005\n\001b\030\001\n0\n\031TwoFloatInputsFloatOutput\022\005\n\001a\030\001\022\005\n\001b\030\001\032\005\n\001c\030\001\n.\n\027TwoFloatInputsIntOutput\022\005\n\001a\030\001\022\005\n\001b\030\001\032\005\n\001c\030\003\n\037\n\017TwoFloatOutputs\032\005\n\001a\030\001\032\005\n\001b\030\001\n\034\n\014TwoIntInputs\022\005\n\001a\030\003\022\005\n\001b\030\003\n\035\n\rTwoIntOutputs\032\005\n\001a\030\003\032\005\n\001b\030\003\n\"\n\005Unary\022\006\n\001a\"\001T\032\006\n\001b\"\001T\"\t\n\001T\022\004type")
