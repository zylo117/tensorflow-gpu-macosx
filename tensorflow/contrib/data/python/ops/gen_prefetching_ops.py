"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: prefetching_ops.cc
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


@tf_export('FunctionBufferingResource')
def function_buffering_resource(string_arg, target_device, shared_name, container, f, buffer_size, thread_pool_size, name=None):
  r"""Creates a resource that fills up a buffer by making function calls.

  Args:
    string_arg: A `Tensor` of type `string`.
      String argument to the function call.
    target_device: A `Tensor` of type `string`.
      Target device to execute the function on.
    shared_name: A `string`.
      If non-empty, this resource will be shared under the given name
      across multiple sessions.
    container: A `string`.
      If non-empty, this resource is placed in the given container.
      Otherwise, a default container is used.
    f: A function decorated with @Defun. Function to be executed.
    buffer_size: An `int`. Size of the buffer.
    thread_pool_size: An `int`. Size of the threadpool doing the prefetching.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. Handle to the resource created.
  """
  shared_name = _execute.make_str(shared_name, "shared_name")
  container = _execute.make_str(container, "container")
  buffer_size = _execute.make_int(buffer_size, "buffer_size")
  thread_pool_size = _execute.make_int(thread_pool_size, "thread_pool_size")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FunctionBufferingResource", string_arg=string_arg,
        target_device=target_device, shared_name=shared_name,
        container=container, f=f, buffer_size=buffer_size,
        thread_pool_size=thread_pool_size, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("shared_name", _op.get_attr("shared_name"), "container",
              _op.get_attr("container"), "f", _op.get_attr("f"),
              "buffer_size", _op.get_attr("buffer_size"), "thread_pool_size",
              _op.get_attr("thread_pool_size"))
  else:
    string_arg = _ops.convert_to_tensor(string_arg, _dtypes.string)
    target_device = _ops.convert_to_tensor(target_device, _dtypes.string)
    _inputs_flat = [string_arg, target_device]
    _attrs = ("shared_name", shared_name, "container", container, "f", f,
              "buffer_size", buffer_size, "thread_pool_size",
              thread_pool_size)
    _result = _execute.execute(b"FunctionBufferingResource", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "FunctionBufferingResource", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("FunctionBufferingResource")(None)


@tf_export('FunctionBufferingResourceGetNext')
def function_buffering_resource_get_next(function_buffer_resource, output_types, name=None):
  r"""Gets the next element from a FunctionBufferingResource.

  Args:
    function_buffer_resource: A `Tensor` of type `resource`.
      The FunctionBufferingResource handle.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
    A list of return values.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'function_buffering_resource_get_next' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FunctionBufferingResourceGetNext",
        function_buffer_resource=function_buffer_resource,
        output_types=output_types, name=name)
    _result = _op.outputs[:]
    if not _result:
      return _op
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"))
  else:
    function_buffer_resource = _ops.convert_to_tensor(function_buffer_resource, _dtypes.resource)
    _inputs_flat = [function_buffer_resource]
    _attrs = ("output_types", output_types)
    _result = _execute.execute(b"FunctionBufferingResourceGetNext",
                               len(output_types), inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "FunctionBufferingResourceGetNext", _inputs_flat, _attrs, _result, name)
  return _result

_ops.RegisterShape("FunctionBufferingResourceGetNext")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "FunctionBufferingResource"
#   input_arg {
#     name: "string_arg"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "target_device"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "shared_name"
#     type: "string"
#   }
#   attr {
#     name: "container"
#     type: "string"
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "buffer_size"
#     type: "int"
#   }
#   attr {
#     name: "thread_pool_size"
#     type: "int"
#   }
#   is_stateful: true
# }
# op {
#   name: "FunctionBufferingResourceGetNext"
#   input_arg {
#     name: "function_buffer_resource"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "output"
#     type_list_attr: "output_types"
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
_op_def_lib = _InitOpDefLibrary(b"\n\263\001\n\031FunctionBufferingResource\022\016\n\nstring_arg\030\007\022\021\n\rtarget_device\030\007\032\014\n\010resource\030\024\"\025\n\013shared_name\022\006string\"\023\n\tcontainer\022\006string\"\t\n\001f\022\004func\"\022\n\013buffer_size\022\003int\"\027\n\020thread_pool_size\022\003int\210\001\001\n{\n FunctionBufferingResourceGetNext\022\034\n\030function_buffer_resource\030\024\032\026\n\006output2\014output_types\"\036\n\014output_types\022\nlist(type)(\0010\001\210\001\001")
