from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils

from tensorflow.python.eager import imperative_grad
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest



# from tensorflow/.../convolutional.py
def cal_output_shape_Conv2D(data_format,input_shape,filters,kernel_size,strides):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    kernel_size = utils.normalize_tuple(kernel_size,2,'kernel_size')
    strides = utils.normalize_tuple(strides,2,'strides')
    dilation_rate = utils.normalize_tuple(1,2,'dilation_rate')

    if data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                #padding='valid',
                padding='same',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space + [filters]).as_list()
    else:
        space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                #padding='valid',
                padding='same',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0], filters] + new_space).as_list()

def cal_output_shape_Conv2D_pad_val(data_format,input_shape,filters,kernel_size,strides):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    kernel_size = utils.normalize_tuple(kernel_size,2,'kernel_size')
    strides = utils.normalize_tuple(strides,2,'strides')
    dilation_rate = utils.normalize_tuple(1,2,'dilation_rate')

    if data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                padding='valid',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space + [filters]).as_list()
    else:
        space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                padding='valid',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0], filters] + new_space).as_list()




#
def cal_output_shape_Pooling2D(data_format,input_shape,pool_size,strides):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()

    pool_size = utils.normalize_tuple(pool_size, 2, 'pool_size')
    strides = utils.normalize_tuple(strides, 2, 'strides')
    padding = 'same'

    if data_format == 'channels_first':
        rows = input_shape[2]
        cols = input_shape[3]
    else:
        rows = input_shape[1]
        cols = input_shape[2]

    rows = utils.conv_output_length(
            rows,
            pool_size[0],
            padding,
            strides[0]
        )

    cols = utils.conv_output_length(
            cols,
            pool_size[1],
            padding,
            strides[1]
        )

    if data_format == 'channels_first':
        return tensor_shape.TensorShape([input_shape[0], input_shape[1], rows, cols]).as_list()
    else:
        return tensor_shape.TensorShape([input_shape[0], rows, cols, input_shape[3]]).as_list()



# from tensorflow/python/eager/backprop.py
def gradient(self, target, sources, output_gradients=None):
    """Computes the gradient using operations recorded in context of this tape.

    Args:
      target: Tensor (or list of tensors) to be differentiated.
      sources: a list or nested structure of Tensors or Variables. `target`
        will be differentiated against elements in `sources`.
      output_gradients: a list of gradients, one for each element of
        target. Defaults to None.

    Returns:
      a list or nested structure of Tensors (or IndexedSlices, or None),
      one for each element in `sources`. Returned structure is the same as
      the structure of `sources`.

    Raises:
      RuntimeError: if called inside the context of the tape, or if called more
       than once on a non-persistent tape.
    """
    if self._tape is None:
      raise RuntimeError("GradientTape.gradient can only be called once on "
                         "non-persistent tapes.")
    if self._recording:
      if not self._persistent:
        self._pop_tape()
      else:
        logging.log_first_n(logging.WARN,
                            "Calling GradientTape.gradient on a persistent "
                            "tape inside it's context is significantly less "
                            "efficient than calling it outside the context (it "
                            "causes the gradient ops to be recorded on the "
                            "tape, leading to increased CPU and memory usage). "
                            "Only call GradientTape.gradient inside the "
                            "context if you actually want to trace the "
                            "gradient in order to compute higher order "
                            "derrivatives.", 1)

    flat_sources = nest.flatten(sources)
    flat_sources = [_handle_or_self(x) for x in flat_sources]

    if output_gradients is not None:
      output_gradients = [None if x is None else ops.convert_to_tensor(x)
                          for x in nest.flatten(output_gradients)]

    flat_grad = imperative_grad.imperative_grad(
        _default_vspace, self._tape, nest.flatten(target), flat_sources,
        output_gradients=output_gradients)

    if not self._persistent:
      self._tape = None

    grad = nest.pack_sequence_as(sources, flat_grad)
    return grad
