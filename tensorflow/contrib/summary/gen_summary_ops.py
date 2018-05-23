"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_summary_ops.cc
"""

import collections as _collections
import six as _six

from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import errors as _errors
from tensorflow.python.framework import tensor_shape as _tensor_shape

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.tf_export import tf_export


@tf_export('close_summary_writer')
def close_summary_writer(writer, name=None):
  r"""Flushes and closes the summary writer.

  Also removes it from the resource manager. To reopen, use another
  CreateSummaryFileWriter op.

  Args:
    writer: A `Tensor` of type `resource`.
      A handle to the summary writer resource.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CloseSummaryWriter", writer=writer, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "CloseSummaryWriter", name,
        _ctx._post_execution_callbacks, writer)
      return _result
    except _core._FallbackException:
      return close_summary_writer_eager_fallback(
          writer, name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def close_summary_writer_eager_fallback(writer, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function close_summary_writer
  """
  _ctx = _context.context()
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  _inputs_flat = [writer]
  _attrs = None
  _result = _execute.execute(b"CloseSummaryWriter", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("CloseSummaryWriter")(None)


@tf_export('create_summary_db_writer')
def create_summary_db_writer(writer, db_uri, experiment_name, run_name, user_name, name=None):
  r"""Creates summary database writer accessible by given resource handle.

  This can be used to write tensors from the execution graph directly
  to a database. Only SQLite is supported right now. This function
  will create the schema if it doesn't exist. Entries in the Users,
  Experiments, and Runs tables will be created automatically if they
  don't already exist.

  Args:
    writer: A `Tensor` of type `resource`.
      Handle to SummaryWriter resource to overwrite.
    db_uri: A `Tensor` of type `string`. For example "file:/tmp/foo.sqlite".
    experiment_name: A `Tensor` of type `string`.
      Can't contain ASCII control characters or <>. Case
      sensitive. If empty, then the Run will not be associated with any
      Experiment.
    run_name: A `Tensor` of type `string`.
      Can't contain ASCII control characters or <>. Case sensitive.
      If empty, then each Tag will not be associated with any Run.
    user_name: A `Tensor` of type `string`.
      Must be valid as both a DNS label and Linux username. If
      empty, then the Experiment will not be associated with any User.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CreateSummaryDbWriter", writer=writer, db_uri=db_uri,
        experiment_name=experiment_name, run_name=run_name,
        user_name=user_name, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "CreateSummaryDbWriter", name,
        _ctx._post_execution_callbacks, writer, db_uri, experiment_name,
        run_name, user_name)
      return _result
    except _core._FallbackException:
      return create_summary_db_writer_eager_fallback(
          writer, db_uri, experiment_name, run_name, user_name, name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def create_summary_db_writer_eager_fallback(writer, db_uri, experiment_name, run_name, user_name, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function create_summary_db_writer
  """
  _ctx = _context.context()
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  db_uri = _ops.convert_to_tensor(db_uri, _dtypes.string)
  experiment_name = _ops.convert_to_tensor(experiment_name, _dtypes.string)
  run_name = _ops.convert_to_tensor(run_name, _dtypes.string)
  user_name = _ops.convert_to_tensor(user_name, _dtypes.string)
  _inputs_flat = [writer, db_uri, experiment_name, run_name, user_name]
  _attrs = None
  _result = _execute.execute(b"CreateSummaryDbWriter", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("CreateSummaryDbWriter")(None)


@tf_export('create_summary_file_writer')
def create_summary_file_writer(writer, logdir, max_queue, flush_millis, filename_suffix, name=None):
  r"""Creates a summary file writer accessible by the given resource handle.

  Args:
    writer: A `Tensor` of type `resource`.
      A handle to the summary writer resource
    logdir: A `Tensor` of type `string`.
      Directory where the event file will be written.
    max_queue: A `Tensor` of type `int32`.
      Size of the queue of pending events and summaries.
    flush_millis: A `Tensor` of type `int32`.
      How often, in milliseconds, to flush the pending events and
      summaries to disk.
    filename_suffix: A `Tensor` of type `string`.
      Every event file's name is suffixed with this suffix.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CreateSummaryFileWriter", writer=writer, logdir=logdir,
        max_queue=max_queue, flush_millis=flush_millis,
        filename_suffix=filename_suffix, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "CreateSummaryFileWriter", name,
        _ctx._post_execution_callbacks, writer, logdir, max_queue,
        flush_millis, filename_suffix)
      return _result
    except _core._FallbackException:
      return create_summary_file_writer_eager_fallback(
          writer, logdir, max_queue, flush_millis, filename_suffix, name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def create_summary_file_writer_eager_fallback(writer, logdir, max_queue, flush_millis, filename_suffix, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function create_summary_file_writer
  """
  _ctx = _context.context()
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  logdir = _ops.convert_to_tensor(logdir, _dtypes.string)
  max_queue = _ops.convert_to_tensor(max_queue, _dtypes.int32)
  flush_millis = _ops.convert_to_tensor(flush_millis, _dtypes.int32)
  filename_suffix = _ops.convert_to_tensor(filename_suffix, _dtypes.string)
  _inputs_flat = [writer, logdir, max_queue, flush_millis, filename_suffix]
  _attrs = None
  _result = _execute.execute(b"CreateSummaryFileWriter", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                             name=name)
  _result = None
  return _result

_ops.RegisterShape("CreateSummaryFileWriter")(None)


@tf_export('flush_summary_writer')
def flush_summary_writer(writer, name=None):
  r"""Flushes the writer's unwritten events.

  Args:
    writer: A `Tensor` of type `resource`.
      A handle to the summary writer resource.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FlushSummaryWriter", writer=writer, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "FlushSummaryWriter", name,
        _ctx._post_execution_callbacks, writer)
      return _result
    except _core._FallbackException:
      return flush_summary_writer_eager_fallback(
          writer, name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def flush_summary_writer_eager_fallback(writer, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function flush_summary_writer
  """
  _ctx = _context.context()
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  _inputs_flat = [writer]
  _attrs = None
  _result = _execute.execute(b"FlushSummaryWriter", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("FlushSummaryWriter")(None)


@tf_export('import_event')
def import_event(writer, event, name=None):
  r"""Outputs a `tf.Event` protocol buffer.

  When CreateSummaryDbWriter is being used, this op can be useful for
  importing data from event logs.

  Args:
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    event: A `Tensor` of type `string`.
      A string containing a binary-encoded tf.Event proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ImportEvent", writer=writer, event=event, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "ImportEvent", name,
        _ctx._post_execution_callbacks, writer, event)
      return _result
    except _core._FallbackException:
      return import_event_eager_fallback(
          writer, event, name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def import_event_eager_fallback(writer, event, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function import_event
  """
  _ctx = _context.context()
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  event = _ops.convert_to_tensor(event, _dtypes.string)
  _inputs_flat = [writer, event]
  _attrs = None
  _result = _execute.execute(b"ImportEvent", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("ImportEvent")(None)


@tf_export('summary_writer')
def summary_writer(shared_name="", container="", name=None):
  r"""Returns a handle to be used to access a summary writer.

  The summary writer is an in-graph resource which can be used by ops to write
  summaries to event files.

  Args:
    shared_name: An optional `string`. Defaults to `""`.
    container: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. the summary writer resource. Scalar handle.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    if shared_name is None:
      shared_name = ""
    shared_name = _execute.make_str(shared_name, "shared_name")
    if container is None:
      container = ""
    container = _execute.make_str(container, "container")
    _, _, _op = _op_def_lib._apply_op_helper(
        "SummaryWriter", shared_name=shared_name, container=container,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("shared_name", _op.get_attr("shared_name"), "container",
              _op.get_attr("container"))
    _execute.record_gradient(
      "SummaryWriter", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "SummaryWriter", name,
        _ctx._post_execution_callbacks, "shared_name", shared_name,
        "container", container)
      return _result
    except _core._FallbackException:
      return summary_writer_eager_fallback(
          shared_name=shared_name, container=container, name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def summary_writer_eager_fallback(shared_name="", container="", name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function summary_writer
  """
  _ctx = _context.context()
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  _inputs_flat = []
  _attrs = ("shared_name", shared_name, "container", container)
  _result = _execute.execute(b"SummaryWriter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "SummaryWriter", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("SummaryWriter")(None)


@tf_export('write_audio_summary')
def write_audio_summary(writer, step, tag, tensor, sample_rate, max_outputs=3, name=None):
  r"""Writes a `Summary` protocol buffer with audio.

  The summary has up to `max_outputs` summary values containing audio. The
  audio is built from `tensor` which must be 3-D with shape `[batch_size,
  frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
  assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

  Args:
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    step: A `Tensor` of type `int64`. The step to write the summary for.
    tag: A `Tensor` of type `string`.
      Scalar. Used to build the `tag` attribute of the summary values.
    tensor: A `Tensor` of type `float32`. 2-D of shape `[batch_size, frames]`.
    sample_rate: A `Tensor` of type `float32`.
      The sample rate of the signal in hertz.
    max_outputs: An optional `int` that is `>= 1`. Defaults to `3`.
      Max number of batch elements to generate audio for.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    if max_outputs is None:
      max_outputs = 3
    max_outputs = _execute.make_int(max_outputs, "max_outputs")
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteAudioSummary", writer=writer, step=step, tag=tag, tensor=tensor,
        sample_rate=sample_rate, max_outputs=max_outputs, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "WriteAudioSummary", name,
        _ctx._post_execution_callbacks, writer, step, tag, tensor,
        sample_rate, "max_outputs", max_outputs)
      return _result
    except _core._FallbackException:
      return write_audio_summary_eager_fallback(
          writer, step, tag, tensor, sample_rate, max_outputs=max_outputs,
          name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def write_audio_summary_eager_fallback(writer, step, tag, tensor, sample_rate, max_outputs=3, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function write_audio_summary
  """
  _ctx = _context.context()
  if max_outputs is None:
    max_outputs = 3
  max_outputs = _execute.make_int(max_outputs, "max_outputs")
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  tensor = _ops.convert_to_tensor(tensor, _dtypes.float32)
  sample_rate = _ops.convert_to_tensor(sample_rate, _dtypes.float32)
  _inputs_flat = [writer, step, tag, tensor, sample_rate]
  _attrs = ("max_outputs", max_outputs)
  _result = _execute.execute(b"WriteAudioSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("WriteAudioSummary")(None)


@tf_export('write_graph_summary')
def write_graph_summary(writer, step, tensor, name=None):
  r"""Writes a `GraphDef` protocol buffer to a `SummaryWriter`.

  Args:
    writer: A `Tensor` of type `resource`. Handle of `SummaryWriter`.
    step: A `Tensor` of type `int64`. The step to write the summary for.
    tensor: A `Tensor` of type `string`.
      A scalar string of the serialized tf.GraphDef proto.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteGraphSummary", writer=writer, step=step, tensor=tensor,
        name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "WriteGraphSummary", name,
        _ctx._post_execution_callbacks, writer, step, tensor)
      return _result
    except _core._FallbackException:
      return write_graph_summary_eager_fallback(
          writer, step, tensor, name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def write_graph_summary_eager_fallback(writer, step, tensor, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function write_graph_summary
  """
  _ctx = _context.context()
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tensor = _ops.convert_to_tensor(tensor, _dtypes.string)
  _inputs_flat = [writer, step, tensor]
  _attrs = None
  _result = _execute.execute(b"WriteGraphSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("WriteGraphSummary")(None)


@tf_export('write_histogram_summary')
def write_histogram_summary(writer, step, tag, values, name=None):
  r"""Writes a `Summary` protocol buffer with a histogram.

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing a histogram for `values`.

  This op reports an `InvalidArgument` error if any value is not finite.

  Args:
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    step: A `Tensor` of type `int64`. The step to write the summary for.
    tag: A `Tensor` of type `string`.
      Scalar.  Tag to use for the `Summary.Value`.
    values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      Any shape. Values to use to build the histogram.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteHistogramSummary", writer=writer, step=step, tag=tag,
        values=values, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "WriteHistogramSummary", name,
        _ctx._post_execution_callbacks, writer, step, tag, values)
      return _result
    except _core._FallbackException:
      return write_histogram_summary_eager_fallback(
          writer, step, tag, values, name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def write_histogram_summary_eager_fallback(writer, step, tag, values, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function write_histogram_summary
  """
  _ctx = _context.context()
  _attr_T, (values,) = _execute.args_to_matching_eager([values], _ctx, _dtypes.float32)
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  _inputs_flat = [writer, step, tag, values]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"WriteHistogramSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("WriteHistogramSummary")(None)


@tf_export('write_image_summary')
def write_image_summary(writer, step, tag, tensor, bad_color, max_images=3, name=None):
  r"""Writes a `Summary` protocol buffer with images.

  The summary has up to `max_images` summary values containing images. The
  images are built from `tensor` which must be 4-D with shape `[batch_size,
  height, width, channels]` and where `channels` can be:

  *  1: `tensor` is interpreted as Grayscale.
  *  3: `tensor` is interpreted as RGB.
  *  4: `tensor` is interpreted as RGBA.

  The images have the same number of channels as the input tensor. For float
  input, the values are normalized one image at a time to fit in the range
  `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
  normalization algorithms:

  *  If the input values are all positive, they are rescaled so the largest one
     is 255.

  *  If any input value is negative, the values are shifted so input value 0.0
     is at 127.  They are then rescaled so that either the smallest value is 0,
     or the largest one is 255.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_images` is 1, the summary value tag is '*tag*/image'.
  *  If `max_images` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

  The `bad_color` argument is the color to use in the generated images for
  non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
  Each element must be in the range `[0, 255]` (It represents the value of a
  pixel in the output image).  Non-finite values in the input tensor are
  replaced by this tensor in the output image.  The default value is the color
  red.

  Args:
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    step: A `Tensor` of type `int64`. The step to write the summary for.
    tag: A `Tensor` of type `string`.
      Scalar. Used to build the `tag` attribute of the summary values.
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `float32`, `half`.
      4-D of shape `[batch_size, height, width, channels]` where
      `channels` is 1, 3, or 4.
    bad_color: A `Tensor` of type `uint8`.
      Color to use for pixels with non-finite values.
    max_images: An optional `int` that is `>= 1`. Defaults to `3`.
      Max number of batch elements to generate images for.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    if max_images is None:
      max_images = 3
    max_images = _execute.make_int(max_images, "max_images")
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteImageSummary", writer=writer, step=step, tag=tag, tensor=tensor,
        bad_color=bad_color, max_images=max_images, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "WriteImageSummary", name,
        _ctx._post_execution_callbacks, writer, step, tag, tensor, bad_color,
        "max_images", max_images)
      return _result
    except _core._FallbackException:
      return write_image_summary_eager_fallback(
          writer, step, tag, tensor, bad_color, max_images=max_images,
          name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def write_image_summary_eager_fallback(writer, step, tag, tensor, bad_color, max_images=3, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function write_image_summary
  """
  _ctx = _context.context()
  if max_images is None:
    max_images = 3
  max_images = _execute.make_int(max_images, "max_images")
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], _ctx, _dtypes.float32)
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  bad_color = _ops.convert_to_tensor(bad_color, _dtypes.uint8)
  _inputs_flat = [writer, step, tag, tensor, bad_color]
  _attrs = ("max_images", max_images, "T", _attr_T)
  _result = _execute.execute(b"WriteImageSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("WriteImageSummary")(None)


@tf_export('write_scalar_summary')
def write_scalar_summary(writer, step, tag, value, name=None):
  r"""Writes a `Summary` protocol buffer with scalar values.

  The input `tag` and `value` must have the scalars.

  Args:
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    step: A `Tensor` of type `int64`. The step to write the summary for.
    tag: A `Tensor` of type `string`. Tag for the summary.
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      Value for the summary.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteScalarSummary", writer=writer, step=step, tag=tag, value=value,
        name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "WriteScalarSummary", name,
        _ctx._post_execution_callbacks, writer, step, tag, value)
      return _result
    except _core._FallbackException:
      return write_scalar_summary_eager_fallback(
          writer, step, tag, value, name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def write_scalar_summary_eager_fallback(writer, step, tag, value, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function write_scalar_summary
  """
  _ctx = _context.context()
  _attr_T, (value,) = _execute.args_to_matching_eager([value], _ctx)
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  _inputs_flat = [writer, step, tag, value]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"WriteScalarSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("WriteScalarSummary")(None)


@tf_export('write_summary')
def write_summary(writer, step, tensor, tag, summary_metadata, name=None):
  r"""Outputs a `Summary` protocol buffer with a tensor.

  Args:
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    step: A `Tensor` of type `int64`. The step to write the summary for.
    tensor: A `Tensor`. A tensor to serialize.
    tag: A `Tensor` of type `string`. The summary's tag.
    summary_metadata: A `Tensor` of type `string`.
      Serialized SummaryMetadata protocol buffer containing
      plugin-related metadata for this summary.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteSummary", writer=writer, step=step, tensor=tensor, tag=tag,
        summary_metadata=summary_metadata, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._handle, _ctx.device_name, "WriteSummary", name,
        _ctx._post_execution_callbacks, writer, step, tensor, tag,
        summary_metadata)
      return _result
    except _core._FallbackException:
      return write_summary_eager_fallback(
          writer, step, tensor, tag, summary_metadata, name=name)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def write_summary_eager_fallback(writer, step, tensor, tag, summary_metadata, name=None):
  r"""This is the slowpath function for Eager mode.
  This is for function write_summary
  """
  _ctx = _context.context()
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], _ctx)
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  summary_metadata = _ops.convert_to_tensor(summary_metadata, _dtypes.string)
  _inputs_flat = [writer, step, tensor, tag, summary_metadata]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"WriteSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _result = None
  return _result

_ops.RegisterShape("WriteSummary")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "CloseSummaryWriter"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "CreateSummaryDbWriter"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "db_uri"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "experiment_name"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "run_name"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "user_name"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "CreateSummaryFileWriter"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "logdir"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "max_queue"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "flush_millis"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "filename_suffix"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "FlushSummaryWriter"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "ImportEvent"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "event"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "SummaryWriter"
#   output_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "shared_name"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   attr {
#     name: "container"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteAudioSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "tensor"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "sample_rate"
#     type: DT_FLOAT
#   }
#   attr {
#     name: "max_outputs"
#     type: "int"
#     default_value {
#       i: 3
#     }
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteGraphSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tensor"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteHistogramSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "values"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#         type: DT_INT32
#         type: DT_UINT8
#         type: DT_INT16
#         type: DT_INT8
#         type: DT_INT64
#         type: DT_BFLOAT16
#         type: DT_UINT16
#         type: DT_HALF
#         type: DT_UINT32
#         type: DT_UINT64
#       }
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteImageSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "tensor"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "bad_color"
#     type: DT_UINT8
#   }
#   attr {
#     name: "max_images"
#     type: "int"
#     default_value {
#       i: 3
#     }
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_UINT8
#         type: DT_FLOAT
#         type: DT_HALF
#       }
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteScalarSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "value"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#         type: DT_INT32
#         type: DT_UINT8
#         type: DT_INT16
#         type: DT_INT8
#         type: DT_INT64
#         type: DT_BFLOAT16
#         type: DT_UINT16
#         type: DT_HALF
#         type: DT_UINT32
#         type: DT_UINT64
#       }
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tensor"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "summary_metadata"
#     type: DT_STRING
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
#   is_stateful: true
# }
_op_def_lib = _InitOpDefLibrary(b"\n#\n\022CloseSummaryWriter\022\n\n\006writer\030\024\210\001\001\nd\n\025CreateSummaryDbWriter\022\n\n\006writer\030\024\022\n\n\006db_uri\030\007\022\023\n\017experiment_name\030\007\022\014\n\010run_name\030\007\022\r\n\tuser_name\030\007\210\001\001\nj\n\027CreateSummaryFileWriter\022\n\n\006writer\030\024\022\n\n\006logdir\030\007\022\r\n\tmax_queue\030\003\022\020\n\014flush_millis\030\003\022\023\n\017filename_suffix\030\007\210\001\001\n#\n\022FlushSummaryWriter\022\n\n\006writer\030\024\210\001\001\n\'\n\013ImportEvent\022\n\n\006writer\030\024\022\t\n\005event\030\007\210\001\001\nR\n\rSummaryWriter\032\n\n\006writer\030\024\"\031\n\013shared_name\022\006string\032\002\022\000\"\027\n\tcontainer\022\006string\032\002\022\000\210\001\001\nn\n\021WriteAudioSummary\022\n\n\006writer\030\024\022\010\n\004step\030\t\022\007\n\003tag\030\007\022\n\n\006tensor\030\001\022\017\n\013sample_rate\030\001\"\032\n\013max_outputs\022\003int\032\002\030\003(\0010\001\210\001\001\n8\n\021WriteGraphSummary\022\n\n\006writer\030\024\022\010\n\004step\030\t\022\n\n\006tensor\030\007\210\001\001\ng\n\025WriteHistogramSummary\022\n\n\006writer\030\024\022\010\n\004step\030\t\022\007\n\003tag\030\007\022\013\n\006values\"\001T\"\037\n\001T\022\004type\032\0020\001:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\210\001\001\n\204\001\n\021WriteImageSummary\022\n\n\006writer\030\024\022\010\n\004step\030\t\022\007\n\003tag\030\007\022\013\n\006tensor\"\001T\022\r\n\tbad_color\030\004\"\031\n\nmax_images\022\003int\032\002\030\003(\0010\001\"\026\n\001T\022\004type\032\0020\001:\007\n\0052\003\004\001\023\210\001\001\n_\n\022WriteScalarSummary\022\n\n\006writer\030\024\022\010\n\004step\030\t\022\007\n\003tag\030\007\022\n\n\005value\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\210\001\001\n^\n\014WriteSummary\022\n\n\006writer\030\024\022\010\n\004step\030\t\022\013\n\006tensor\"\001T\022\007\n\003tag\030\007\022\024\n\020summary_metadata\030\007\"\t\n\001T\022\004type\210\001\001")
