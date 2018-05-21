"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: decode_video_op_py.cc
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


@tf_export('DecodeVideo')
def decode_video(contents, name=None):
  r"""Processes the contents of an audio file into a tensor using FFmpeg to decode

  the file.

  One row of the tensor is created for each channel in the audio file. Each
  channel contains audio samples starting at the beginning of the audio and
  having `1/samples_per_second` time between them. If the `channel_count` is
  different from the contents of the file, channels will be merged or created.

  Args:
    contents: A `Tensor` of type `string`.
      The binary audio file contents, as a string or rank-0 string
      tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "DecodeVideo", contents=contents, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    contents = _ops.convert_to_tensor(contents, _dtypes.string)
    _inputs_flat = [contents]
    _attrs = None
    _result = _execute.execute(b"DecodeVideo", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "DecodeVideo", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "DecodeVideo"
#   input_arg {
#     name: "contents"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "output"
#     type: DT_UINT8
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\'\n\013DecodeVideo\022\014\n\010contents\030\007\032\n\n\006output\030\004")
