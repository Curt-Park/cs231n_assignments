from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CustomRNN(object):
    """
    A CustomRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CustomRNN.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, rnn_layers=2,
                 cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.rnn_layers = rnn_layers
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        for i in range(rnn_layers):
            # Initialize CNN -> hidden state projection parameters
            self.params['W_proj%d'%(i)] = np.random.randn(input_dim, hidden_dim)
            self.params['W_proj%d'%(i)] /= np.sqrt(input_dim)
            self.params['b_proj%d'%(i)] = np.zeros(hidden_dim)

            # Initialize parameters for the RNN
            if i == 0:
                in_dim = wordvec_dim
            else:
                in_dim = hidden_dim
            self.params['Wx%d'%(i)] = np.random.randn(in_dim, dim_mul * hidden_dim)
            self.params['Wx%d'%(i)] /= np.sqrt(in_dim)
            self.params['Wh%d'%(i)] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
            self.params['Wh%d'%(i)] /= np.sqrt(hidden_dim)
            self.params['b%d'%(i)] = np.zeros(dim_mul * hidden_dim)

            if i < rnn_layers - 1:
                out_dim = hidden_dim
            else:
                out_dim = vocab_size
            self.params['W_hidden%d'%(i)] = np.random.randn(hidden_dim, out_dim)
            self.params['W_hidden%d'%(i)] /= np.sqrt(hidden_dim)
            self.params['b_hidden%d'%(i)] = np.zeros(out_dim)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        loss, grads = 0.0, {}
        cache = {}

        # Forward
        output_rnn = None
        input_rnn, cache['word_embed'] = word_embedding_forward(captions_in, self.params['W_embed'])
        for i in range(self.rnn_layers):
            W_proj, b_proj = self.params['W_proj%d'%(i)], self.params['b_proj%d'%(i)]
            Wx, Wh, b = self.params['Wx%d'%(i)], self.params['Wh%d'%(i)], self.params['b%d'%(i)]
            W_hidden, b_hidden = self.params['W_hidden%d'%(i)], self.params['b_hidden%d'%(i)]

            h0 = features.dot(W_proj) + b_proj

            if self.cell_type is 'rnn':
                h, cache['rnn%d'%(i)] = rnn_forward(input_rnn, h0, Wx, Wh, b)
            elif self.cell_type is 'lstm':
                h, cache['lstm%d'%(i)] = lstm_forward(input_rnn, h0, Wx, Wh, b)

            output_rnn, cache['temp_affine%d'%(i)] = temporal_affine_forward(h, W_hidden, b_hidden)

            if i < self.rnn_layers - 1:
                input_rnn, cache['relu%d'%(i)] = relu_forward(output_rnn)

        loss, d_temp_affine = temporal_softmax_loss(output_rnn, captions_out, mask)

        # Backward
        for i in range(self.rnn_layers-1, -1, -1):
            if i < self.rnn_layers - 1:
                d_temp_affine = relu_backward(d_temp_affine, cache['relu%d'%(i)])

            dh, grads['W_hidden%d'%(i)], grads['b_hidden%d'%(i)] =\
                temporal_affine_backward(d_temp_affine, cache['temp_affine%d'%(i)])

            if self.cell_type is 'rnn':
                d_temp_affine, d_h0, grads['Wx%d'%(i)], grads['Wh%d'%(i)], grads['b%d'%(i)] = \
                  rnn_backward(dh, cache['rnn%d'%(i)])
            elif self.cell_type is 'lstm':
                d_temp_affine, d_h0, grads['Wx%d'%(i)], grads['Wh%d'%(i)], grads['b%d'%(i)] = \
                  lstm_backward(dh, cache['lstm%d'%(i)])

            grads['b_proj%d'%(i)] = d_h0.sum(axis=0)
            grads['W_proj%d'%(i)] = features.T.dot(d_h0)

        grads['W_embed'] = word_embedding_backward(d_temp_affine, cache['word_embed'])

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        cache = {}
        for i in range(self.rnn_layers):
            W_proj, b_proj = self.params['W_proj%d'%(i)], self.params['b_proj%d'%(i)]
            cache['h_prev%d'%(i)] = features.dot(W_proj) + b_proj
            if self.cell_type is 'lstm':
                cache['c_prev%d'%(i)] = np.zeros_like(cache['h_prev%d'%(i)])

        W_embed = self.params['W_embed']
        input_rnn = W_embed[self._start,:] * np.ones((N, W_embed.shape[1]))
        captions[:, 0] = self._start
        for t in range(1, max_length):
            output_rnn = None
            for i in range(self.rnn_layers):
                # rnn
                Wx, Wh, b = self.params['Wx%d'%(i)], self.params['Wh%d'%(i)], self.params['b%d'%(i)]
                if self.cell_type is 'rnn':
                    h_t, _ = rnn_step_forward(input_rnn, cache['h_prev%d'%(i)], Wx, Wh, b)
                elif self.cell_type is 'lstm':
                    h_t, cache['c_prev%d'%(i)], _ = lstm_step_forward(input_rnn, cache['h_prev%d'%(i)], cache['c_prev%d'%(i)], Wx, Wh, b)
                cache['h_prev%d'%(i)] = h_t

                # temporal affine
                W_hidden, b_hidden = self.params['W_hidden%d'%(i)], self.params['b_hidden%d'%(i)]
                output_rnn = h_t.dot(W_hidden) + b_hidden
                if i < self.rnn_layers - 1:
                    input_rnn, _ = relu_forward(output_rnn)

            word_t = np.argmax(output_rnn, axis=1)
            captions[:, t] = word_t

            input_rnn = W_embed[word_t, :]

        return captions
