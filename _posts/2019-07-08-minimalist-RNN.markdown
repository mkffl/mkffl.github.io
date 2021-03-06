---
title: Minimalist Recurrent Neural Network
layout: post
---

<h2>Motivations</h2>

Recurrent neural networks are a type of NN that can model sequences of inputs by capturing the contribution of previous inputs at all time steps. Although challenged by recent model architectures like the Transformer which are faster to train, recurrent nets are still widely used as they achieve high accuracy on various tasks like text generation, sentence classification, text-to-speech recognition or time series forecasting.
<br><br>
To understand how neural nets work in practice, I find writing simplified implementations more useful than looking at the source code from packages like Tensorflow or PyTorch. Strip out all the boilerplate, efficiency tricks and modular code, to focus on the logic of a model. Although not efficient and probably unstable, minimalist implementations can help make sense of the most complex NN architectures.
<br><br>
The maths behind RNN's have been covered extensively and a few simplified implementations are available. Probably the most famous was the character-based RNN code published 4 years ago by Andrej Karpathy along with an <a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">article</a> that explains the reasons behind their good performance. Others have used and commented Karpathy's code, including this <a href="https://eli.thegreenplace.net/2018/understanding-how-to-implement-a-character-based-rnn-language-model/">blog post</a> which comes with a <a href="https://github.com/eliben/deep-learning-samples/blob/master/min-char-rnn/min-char-rnn.py">fork</a> of the original code that includes more commentary.
<br><br>
However, even after reading every comment in Eli Bendersky's gist above, I found myself with more questions than answers. The backward pass in lines 128-153 are especially cryptic. It took me some time and a few pages of maths derivation to really understand what was going on.
<br><br>
Here I look at recurrent neural nets from the perspective of a code implementation. I take the forked code mentioned above as the reference code base and I attempt to break down the forward and backward passes in <code>lossFunc</code> (lines 73-159, and copied at the end for convenience). For a refresh on the inner workings of RNNs, in particular backpropagation through time (BPTT), I include a few references at the bottom of this article.


<h2>Notations</h2>

I stay close to the symboles used in the code for easy code-math comparisons:

<ul style="list-style-type:disc;">
  <li>$z^n = W^{xh}\cdot x + W^{hh}\cdot h^{n-1}$</li>
  <li>$h^n = tanh(z^n)$</li>
  <li>$y^n = W^{hy}\cdot h^n$</li>
  <li>$p^n = softmax(y^n)$</li>
  <li>$Loss^n = crossEntropy(p^n, targets^n)$</li>
</ul>

$W^{xh}$, $W^{hh}$ and $W^{hy}$ are the parameters we want to optimise with backpropagation. $W^{xh}$ turns the one-hot encoded character vectors into embeddings of size vocab_size. 
$y$ is the vector of unnormalised scores, $p$ refers to the probabilities after normalising $y$ with softmax, $targets$ is the one-hot vector of labels, and $Loss$ is the loss function used for optimisation.


<h2>General comments</h2>
Underlying code available <a href="https://github.com/eliben/deep-learning-samples/blob/master/min-char-rnn/min-char-rnn.py">here</a>.

<h4>Input/Outputs</h4>

For reminders, RNNs take a whole sequence as inputs. In this example, sequences are made of characters, of length seq_length and instantiated on lines 245-246:
```python 
inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
```

The code follows the standard feed-forward / backprop programme for RNNs:
<ol>
  <li>take a sequence of inputs (characters) and their correponding output (next character)</li>
  <li>Run inputs chronologically through the forward and backward passes</li>
  <li>Add up the losses and gradients at each input step, and update gradients after the last input is processed</li>
</ol>

<h4>Truncated BPTT</h4>

`lossFun` takes sequences of fixed size instead of e.g. variable sentence-length sequences.


```python
inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
```

That's because the code implements Truncated Backpropagation Through Time. Some sentences can be very long i.e. include many time steps, which would a) require a lot of memory and b) pose computational problems. Regarding a), unrolling the RNN for a very long sequence requires to save the history of all activations. In lossFunc you can see that via the <code>t</code> index for each activation, see for example

```python
# Hidden state value at time step t
 hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
```

As regards b), this refers to the vanishing gradient problem, a limit of vanilla RNN which more sophisticated memory cells like GRU can solve. This point is well covered in this <a href="http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/">WildML article</a>.

<h4>Forward pass</h4>

Not much to say; lines 92-115 are pretty straightforward and the cross-entropy calculation is well detailed in the comments.

<h2>Backward pass</h2>
This section covers the second half of <code>lossFun</code> on lines 128-153

<h4>dWhy</h4>

The first parameter to update is $W^{hy}$, the matrix that turns the hidden layer into a score vector p. The jacobian ha the same size as $W^{hy}$ which is (vocab_size x hidden_size). Note that vocab_size is equal to the number of classes, often denoted K in the literature, as a character-based RNN predicts the next character among K options.
<br><br>
The value of the derivative is
$$
\frac{\partial Loss^n}{\partial W^{hy}_{ij}}=
\begin{cases}
    (p_i-1)\cdot h_j,& \text{if } i = y_k\\
    p_i\cdot h_j,              & \text{otherwise}
\end{cases}
$$
<a href="http://cs231n.github.io/neural-networks-case-study/#grad">Here</a> is an intuitive explanation and a detailed derivation is available <a href="https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/">here</a>.
The following lines implement the above equation:

```python
dy = np.copy(ps[t])
dy[targets[t]] -= 1

# Compute gradients for the Why parameters.
dWhy += np.dot(dy, hs[t].T)
```

A quick check confirms that the resulting jacobian matrix, dWhy, has the right size: dy is (vocab_size x 1) and the transpose of hs[t] is (1 x hidden_size). Also the second line substracts 1 from dy where $i = y_k$ so the vector dot product returns the same result as the equation above - it takes 30 sec to write it down to get convinced.

<h4>dWxh and dWhh</h4>

The next lines in the code address the derivatives for $W^{xh}$ and $W^{hh}$. The approach is similar for both and I will take $W^{xh}$ as an example. I found these lines to be the most difficult to make sense of.
<br><br>
Simple chain rule gives the following partial derivative a time step n:

$$
\frac{\partial Loss^n}{\partial W^{xh}}=\frac{\partial Loss^n}{\partial p^n}\cdot\frac{\partial p^n}{\partial y^n}\cdot\frac{\partial y^n}{\partial h^n}\cdot\frac{\partial h^n}{\partial z^n}\cdot\frac{\partial z^n}{\partial W^{xh}}
$$

It would be tempting to put the pen down and start implementing code but we are not done with the derivation - note that there is both a direct and an indirect dependence between $z^n$ and $W^{xh}$ via $h^{n-1}$:

$$
\frac{\partial z^n}{\partial W^{xh}}=\frac{\partial z^n}{\partial W^{xh}}+\frac{\partial z^n}{\partial h^{n-1}}\cdot\frac{\partial h^{n-1}}{\partial z^{n-1}}\cdot\frac{\partial z^{n-1}}{\partial W^{xh}}
$$

The first term in the above sum is the direct dependence and the second term is the indirect dependence. Next, note that the same dependence applies via previous time steps, for example:

$$
\frac{\partial z^{n-1}}{\partial W^{xh}}=\frac{\partial z^{n-1}}{\partial W^{xh}}+\frac{\partial z^{n-1}}{\partial h^{n-2}}\cdot\frac{\partial h^{n-2}}{\partial z^{n-2}}\cdot\frac{\partial z^{n-2}}{\partial W^{xh}}
$$

And it continues until $h^0$ so the partial derivative becomes
$$
\frac{\partial Loss^n}{\partial W^{xh}}=\frac{\partial Loss^n}{\partial p^n}\cdot\frac{\partial p^n}{\partial y^n}\cdot\frac{\partial y^n}{\partial h^n}\cdot\frac{\partial h^n}{\partial z^n}\sum_{t=1}^{n}\frac{\partial z^t}{\partial h^{t-1}}\cdot\frac{\partial h^{t-1}}{\partial z^{t-1}}\cdot\frac{\partial z^{t-1}}{\partial W^{xh}}
\tag{1}\label{1}
$$

In the above formula, it is obvious that there is a long chain of dependences, which is why vanilla RNN are subject to vanishing gradients, as explained <a href="http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/">in this article</a>. This is why in practice people use more complex hidden neurons like GRU and LSTM units.
<br><br>
RNN derivations stop here and most articles about Karpathy's RNN code would not explain further, suggesting that the code just implements the above equation. But that's not true and if you look at lines 141-148 you may wonder how the code ties back to the sum above.
<br><br>
Basically, here is what happens in code: as the author loops through each time step in reverse order, he does not compute all the elements from the above sum. Instead, at time step n, he computes the direct dependence and leaves indirect dependences to be computed at previous steps. Might sound confusing so let's see how it works in detail.
<br><br>
At time step n, let's define the recursive error term e:
$$
e^{nfn}=\frac{\partial Loss^n}{\partial p^n}\cdot\frac{\partial p^n}{\partial y^n}\cdot\frac{\partial y^n}{\partial h^n}
$$
The first index in this term's superscript refers to the indirect dependences back in time, for example step (n-1) is $e^{n-1fn}=\frac{\partial Loss^n}{\partial p^n}\cdot\frac{\partial p^n}{\partial y^n}\cdot\frac{\partial y^n}{\partial h^n}\cdot\frac{\partial h^n}{\partial z^n}\cdot\frac{\partial z^n}{\partial h^{n-1}}$. And since the term is recursive we can write   $e^{n-1fn}=e^{nfn}\cdot\frac{\partial h^n}{\partial z^n}\cdot\frac{\partial z^n}{\partial h^{n-1}}$. For step t this gives 

$$
e^{tfn}=e^{t+1fn}\cdot\frac{\partial h^{t+1}}{\partial z^{t+1}}\cdot\frac{\partial z^{t+1}}{\partial h^{t}}
$$

Using this notation, loss equation (1) becomes
$$
\frac{\partial Loss^n}{\partial W^{xh}}=\sum_{t=1}^{n}e^{tfn}\cdot\frac{\partial h^t}{\partial z^t}\cdot\frac{\partial z^t}{\partial W^{xh}}
$$

In the code the BPTT loop visits each time step n but only computes $e^{nfn}$, the error for the direct dependence; the loop also collects previously computed error terms from variable dh, which allows to update dwxh in the following way:
$$ 
dwxh += \sum_{m=n}^N e^{nfm}\cdot\frac{\partial h^n}{\partial z^n}\cdot\frac{\partial z^n}{\partial W^{xh}}
$$

This is equivalent to computing equation (1) but it's more efficient from a code point-of-view.

At this stage the backward pass in the code should make sense. Below is a detailed view for Wxh's backpropagation in a simple example with 3 time steps. I explicit the values stored in the code variables at each time step to clarify what they do. Lines of interest are (I stripped out the comments):

```python 
  dy = np.copy(ps[t])
  dy[targets[t]] -= 1

  dh = np.dot(Why.T, dy) + dhnext

  dhraw = (1 - hs[t] * hs[t]) * dh

  dWxh += np.dot(dhraw, xs[t].T)

  dhnext = np.dot(Whh.T, dhraw)
```

<h4>Time step 3</h4>
<h5><i>(Remember, we are going in reverse time order)</i></h5>

$$ dy \leftarrow \frac{\partial Loss^3}{\partial p^3}\cdot\frac{\partial p^3}{\partial y^3}$$
$$ dh \leftarrow dy\cdot\frac{\partial y^3}{\partial h^3} + \vec{0} = e^{3f3} $$
$$ dhraw \leftarrow e^{3f3}\cdot\frac{\partial h^3}{\partial z^3} $$
$$ dWxh += e^{3f3}\cdot\frac{\partial h^3}{\partial z^3}\cdot\frac{\partial z^3}{\partial W^{xh}} $$
$$ dhnext \leftarrow e^{3f3}\cdot\frac{\partial h^3}{\partial z^3}\cdot\frac{\partial z^3}{\partial h^2} = e^{2f3}$$
 

<h4>Time step 2</h4>

$$ dy \leftarrow \frac{\partial Loss^2}{\partial p^2}\cdot\frac{\partial p^2}{\partial y^2}$$
$$ dh \leftarrow e^{2f2} + e^{2f3}$$
$$ dhraw \leftarrow (e^{2f2} + e^{2f3})\cdot\frac{\partial h^2}{\partial z^2} $$
$$ dWxh += (e^{2f2} + e^{2f3})\cdot\frac{\partial h^2}{\partial z^2}  \cdot\frac{\partial z^2}{\partial W^{xh}} $$
$$ dhnext \leftarrow e^{1f2} + e^{1f3}$$

<h4>Time step 1</h4>

$$ dy \leftarrow \frac{\partial Loss^1}{\partial p^1}\cdot\frac{\partial p^1}{\partial y^1}$$
$$ dh \leftarrow e^{1f1} + e^{1f2} + e^{1f3}$$
$$ dhraw \leftarrow (e^{1f1} + e^{1f2} + e^{1f3})\cdot\frac{\partial h^1}{\partial z^1} $$
$$ dWxh += (e^{1f1} + e^{1f2} + e^{1f3})\cdot\frac{\partial h^1}{\partial z^1}\cdot\frac{\partial z^1}{\partial W^{xh}} $$

By the time we complete t=1, variable dWxh has accumulated all the gradients associated with each recursive error terms. Hence its final value is $\frac{\partial Loss^n}{\partial W^{xh}}$. A simple way to see it is to add up all the lines for dWxh and re-order them to retrieve equation (1).
<br>
<br>
End of <code>lossFunc</code>. The last two lines update gradients and clip values to a range between -5 and 5.


#### References
<ul style="list-style-type:disc;">
  <li><a href="https://eli.thegreenplace.net/2018/understanding-how-to-implement-a-character-based-rnn-language-model/">Understanding how to implement a character-based RNN language model</a></li>
  <li><a href="http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/">Recurrent Neural Networks Tutorial, Part 3 – Backpropagation Through Time and Vanishing Gradients"></a></li>
  <li><a href="http://willwolf.io/2016/10/18/recurrent-neural-network-gradients-and-lessons-learned-therein/">Recurrent Neural Network Gradients, and Lessons Learned Therein</a></li>
  <li><a href="https://www.deeplearningbook.org/contents/rnn.html">RNN section in the Deep Learning book</a></li>
</ul>


#### Code

<a href="https://github.com/eliben/deep-learning-samples/blob/master/min-char-rnn/min-char-rnn.py">Source</a>
```python 
def lossFun(inputs, targets, hprev):
  """Runs forward and backward passes through the RNN.
  inputs, targets: Lists of integers. For some i, inputs[i] is the input
                   character (encoded as an index into the ix_to_char map) and
                   targets[i] is the corresponding next character in the
                   training data (similarly encoded).
  hprev: Hx1 array of initial hidden state
  returns: loss, gradients on model parameters, and last hidden state
  """
  # Caches that keep values computed in the forward pass at each time step, to
  # be reused in the backward pass.
  xs, hs, ys, ps = {}, {}, {}, {}

  # Initial incoming state.
  hs[-1] = np.copy(hprev)
  loss = 0
  # Forward pass
  for t in range(len(inputs)):
    # Input at time step t is xs[t]. Prepare a one-hot encoded vector of shape
    # (V, 1). inputs[t] is the index where the 1 goes.
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1

    # Compute h[t] from h[t-1] and x[t]
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)

    # Compute ps[t] - softmax probabilities for output.
    ys[t] = np.dot(Why, hs[t]) + by
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

    # Cross-entropy loss for two probability distributions p and q is defined as
    # follows:
    #
    #   xent(q, p) = -Sum q(k)log(p(k))
    #                  k
    #
    # Where k goes over all the possible values of the random variable p and q
    # are defined for.
    # In our case taking q is the "real answer" which is 1-hot encoded; p is the
    # result of softmax (ps). targets[t] has the only index where q is not 0,
    # so the sum simply becomes log of ps at that index.
    loss += -np.log(ps[t][targets[t],0])

```
