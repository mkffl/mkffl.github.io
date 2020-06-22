---
title: Introduction to Expectation Maximization - part 2 (draft)
layout: post
---

## Beyond Gaussian Mixtures

### PCA, LDA and HMM

- Discrete to continuous latent spaces

- Exact to approximate inference

- Non iid data

### Variational Autoencoders

VAEs are a recent type of machine learning models that have expanded the applications of generative models
to new data types, including images and audio signals, by combining variational Bayes with deep learning architectures. 
Expectation Maximization and VAE both maximise a lower bound of the maximum likelihood function so many aspects uncovered previously
will come in handy.

The following will emphasize some of the key similarities between EM and VAEs but it is not intended
to provide a comprehensive treatment of VAEs. Others have done it better than I could hope to and
interested readers can refer to [cite Kingma] and other papers listed at the end.

#### Continuous latent space

The diagram below compares the generative process from the latent space to the data space for VAEs vs discrete mixture models. 
For discrete models the parameters to estimate are the latent variable priors and the condititional distributions rv parameters, 
e.g. for gaussians the mean and variance. 

<figure align="center">
  <img src="{{site.url}}/assets/vae-vs-gmm.png" alt="my alt text"/>
    <figcaption>
    Prior probability and transformation: the Poisson mixture prior is drawn from a categorical RV with weights $\pi$ vs an isometric gaussian (or other continuous distributions) for VAEs. The VAE then passes the $t$ sample to $f$ to output the parameters of the conditional probability.
    Conditional probability: conditional parameters are $\lambda_c$ in the discrete Poisson case vs $f(t \vert \theta)$ in the continuous case, which corresponds to a vector of 784 probabilities of success (one for every pixel/dimension in $x_i$).
    Marginal probability: in the discrete case the blue outline that denotes the complex marginal probability is the combination of single probabilities; the marginal probability of a VAE can’t easily be plotted but random draws from the pink region in the prior distribution would results in images that share similarities e.g. 7 and 9 digits (source image taken from the Tensorflow documentation).
    </figcaption>
</figure>


With a discrete latent space the number of parameters to estimate depend on the number of mixture components. With a continuous space there is no finite number of mixture components so instead the parameters to estimate are the coefficiences of the transformation from $t$ to 
the parameters of the mixture random variable, e.g. probability of success $p$ for a Bernoulli rv. 

The number of parameters thus depends on the transformation function, $f$. This function can be as simple as a logit although most applications would use deep neural networks to address non-linear data structures like images or sound waves. The mixture distribution is said to be **parametrised** by the function $f$, which in the absence of a closed form solution, is typically optimised with stochastic gradient descent (SGD).

For a given latent variable $t$ and a set of weights $\theta$ the MNIST digit geneated is assumed to be distributed as

$$
x_i \sim \rm Bernoulli(f(t \vert \theta))
$$

and as with previous examples the log-likelihood can be marginalised out

$$
\begin{equation}
\begin{aligned}

\sum_{i=1}^N \log P(x_i|\theta)
& = \sum_{i=1}^N \log \int  P(x_i, t \vert \theta) dt \\
& = \sum_{i=1}^N \log \int P_{\mathcal{N}}(t \vert 0, I) * P_{\rm Bernoulli}(x_i \vert f(t \vert \theta)) dt

\end{aligned}
\end{equation}
\tag{4.1}
$$

#### Maximium Likelihood Estimation

Using a random variable parameterised by a function that has no closed form solution means that optimisation will use approximation methods instead of analytical ones. 

Removing the constraints associated with analytical solutions gives hope for direct optimisation of the log-likelihood, as it can be optimised with gradient optimisation methods after approximating the integral with a sampling method. For example, using Monte Carlo approximation, () becomes


$$
\sum_{i=1}^N \log P(x_i|\theta) \approx \sum_{i=1}^N \log \sum_{m=1}^M P_{\rm Bernoulli}(x_i \vert f(t_m \vert \theta)) dt
$$


where $t_m$ refers refers to the m'th sample from a normalised gaussian. The expression to evaluate is an unbiased estimator of the integral of the log-likelihood, which is good but not necessarily efficient. 

Using a sentence from the original paper in a different context, "this series of operations can be expressed as a symbolc graph in software like Tensorflow or Chainer, and effortlessly differentiated wrt the parameters $\theta$", and I would add "although painfully computed unless using quantum machines from the future."

The additional bit refers to sampling from $t$ being inefficient because it requires a very large number of samples to make marginal improvements. Most samples from $t$ will contribute next to nothing to the observations $x$, which means that gradients will take a lot of time to converge. As a reminder, stochastic gradient descent nudges gradients to the right direction by getting feedback from the loss function. Here feedback will often be poor as the loss function does not get many examples of correct $t$ values. 

It is like a tourist asking for directions by enumerating all the wrong routes to locals. There is a large number of routes to rule out before one finds the correct itinerary. If the symbolic graph, i.e. the tourist, could make more educated suggestions then it could get to its destination faster. The lower bound addresses this issue via the posterior function.


#### Lower bound

Both EM and the VAE aim to maximise a lower bound on the log-likelihood of the joint probability. These two methods are connected via the lower bound. The derivation that gave equation (4) in part 1 applies to continuous latent models with minimum changes. $q$ still refers to any probability distribution over the hidden variables

$$
\sum_{i=1}^N \log P(x_i|\theta) = L(\theta) + KL(q(t)\parallel p(t|x,\theta))
$$

As with discrete mixtures, the objective function can be written as the sum of a lower bound and a measure of the divergence between the latent variable posterior distribution and the arbitrary distribution $q$. In the handwritten digit example, the posterior probability returns the most likely latent value $t$ for a given MNIST picture.

The lower bound is 

$$
L(\theta) = \sum_i^N\{ \int q(t) \log P(x_i, t) dt - \int q(t) \log q(t) dt \}

\tag{4.2}
$$

We will assume to have a good approximation of the posterior distribution, $q_{magic}(t \vert x_i) \approx p(t_i \vert x_i)$. $q_{magic}$ is a temporary placeholder to focus on the similarities with EM. Solving for $\theta$ then corresponds to

$$
{argmax}{_\theta} L(\theta) = \sum_i^N\ \int q_{magic}(t) \log P(x_i, t) dt
$$


As with the discrete mixture case this is close to the observed component likelihood function with $q_{magic}$ filling in the gap for the latent variable. Applying the distributional assumptions for the MNIST example and approximating the integral gives 

$$
\approx \sum_i^N\ \sum_{m=1}^M q_{magic}(t_m) \log \{ P_{\mathcal{N}}(t \vert 0, I) * P_{\rm Bernoulli}(x_i \vert f(x_i \vert t, \theta)) \} dt

\tag{4.3}
$$

This quantity, which can be evaluated with stochastic gradient methods, contrasts with the direct log-likelihood function above because sampling over the approximate posterior is more efficient than sampling over the prior distribution. Instead of looping through many unlikely $t$ values, the model is forced to loop throuh the most likely latent values that generated the observations, which $q_{magic}$ does as an approximation of the posterior.


Finally, the VAE optimises a rearranged version of the lower bound that is more aligned with the machine learning view. Moving the prior probability $p(t)$ to the RHS expression then (4.3) can be written

$$
\sum_i^N\ \int q(t \vert x_i) \log P_{\rm Bernoulli}(x_i \vert f(x_i \vert t, \theta)) dt  - KL(q(t \vert x_i) \parallel P_{\mathcal{N}}(t \vert 0, I))

\tag{4.4}
$$

The previous form, sometimes called "free energy", can be seen as maximising the observed component function even as latent components are not observed. As described with the Poisson mixture, the posterior $q$ is the best substitute for the unobserved latent variable.

The new form, sometimes called "penalised model fit", can be seen as a regularised function, e.g. as with the Lasso regression where L1 acts as the regularisation term. The LHS expression measures the model fit through the marginal probability, while the KL divergence acts a regularisation term as it forces the posterior approximation close to an isometric gaussian.


#### Variational Bayes

EM sets $q$ equal to the posterior $p(t \vert x)$ so the lower bound is equal to the log-likelihood using Bayes' formula, which was possible in the presence of a closed form solution for mixture distributions. With the complex function $f$ it would be hard because the denominator of Bayes' formula is an integral of a function with no closed-form solution. 

The VAE optimiation scheme proposed in the article, called Auto Encoding Variational Bayes (AEVB), differs from EM because it uses an approximation of the posterior instead of plugging the result from E into the M objective function. At a high level, AEVB sets q to be a simple random variable, typically a symmetric gaussian distribution, parametrised by a neural network based transformation $g$ with weights $\phi$. The lower bound $L(\theta, \phi)$ is then a function of two sets of parameters optimised jointly with batch gradient descent methods. 

The reason why this scheme converges to a local optimum is beyond the present scope and can be found in the stochastic variational inference literature - references at the bottom. The implementation details are well covered in the article and most neural network packages documentation. The following glosses over some key implementation aspects using a detailed formulation of the objective function $L$, which I find is often missing in articles, blogs and other documentation. 


Finally, it took me some time to comprehend how AEVB approximates the unseen posterior with a function. What I found unsettling is that the variational function to estimate is an input into another function, $f$, whose parameters are alos estimated. This is probably because I did not come to VAEs with a background in variational inference. However this type of inference problems is similar to other statistical estimation - guess the values of invisible parameters by doing stuff with the visible output.

The variational assumptions are

$$
q(t \vert \ x_i) \sim \mathcal{N} (\mu, \Sigma)
$$

$$
\mu = g_{\mu} (x_i \vert \phi)
$$

$$
\Sigma = g_{\Sigma} (x_i \vert \phi)
$$

The first line means that $q$, which approximates the complicated posterior, is set to be a gaussian distribution. Its covariance $\Sigma$ is symmetric i.e. latent variables are assumed independent. It is unlikely to be true but it reduces the number of parameters vs a free covariance matrix. The second and third lines mean that $g$ has two outputs corresponding to the two parameters of the distribution that it parameterises, $q$.

Plugging this into the lower bound,


$$
L(\theta, \phi) = \sum_i^N\ \int P_{\mathcal{N}}(t \vert \mu, \Sigma) \log P_{\rm Bernoulli}(x_i \vert f(x_i \vert t, \theta)) dt  - KL(P_{\mathcal{N}}(t \vert \mu, \Sigma) \parallel P_{\mathcal{N}}(t \vert 0, I))

\tag{4.5}
$$

The integral is approximated as before but M=1 [note on variance]. $t^{\ast}$ refers to that one sample drawn from ${\mathcal{N}}(\mu, \Sigma)$. 

$$
\approx \sum_i^N\ \log P_{\rm Bernoulli}(x_i \vert f(x_i \vert t^{\ast}, \theta)) dt  - KL(P_{\mathcal{N}}(t \vert \mu, \Sigma) \parallel P_{\mathcal{N}}(t \vert 0, I))
$$

To be accurate the sample $\epsilon$ is drawn from ${\mathcal{N}}(0, I)$ then multiplied by $\Sigma$ and added to $\mu$. This location-scale transformation does not change $t$'s distribution but [TBC] allows $\phi$ to remain fixed during backprogagation. It's called the "reparameterization trick" and can be reflected in the lower bound as

$$
\approx \sum_i^N\ \log P_{\rm Bernoulli}(x_i \vert f(x_i \vert t = g_{\mu} (x_i \vert \phi) + g_{\Sigma} (x_i \vert \phi) * \epsilon^{\ast}, \theta)) dt - KL(P_{\mathcal{N}}(t \vert \mu, \Sigma) \parallel P_{\mathcal{N}}(t \vert 0, I))

\tag{4.6}
$$

The encoder-decoder framework helps make sense of the LHS nested conditional probability. Starting with the innermost condition, $g_{\mu} (x_i \vert \phi) + g_{\Sigma} (x_i \vert \phi) * \epsilon^{\ast}$ is the encoder i.e. mapping $x_i$ to the latent space. Then $f(x_i \vert t)$ is the decoder i.e. the transformation from $t$ to the observed space, and $P_{\rm Bernoulli}$ is the distributional assumption for the marginal likelihood.

#### Lower bound code example

The [Keras blog](https://blog.keras.io/building-autoencoders-in-keras.html) provides a slick code implementation of the above objective function in (4.6)

```python
def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)
```

In `vae_loss`, `x_ent` corresponds to the recontruction loss and `kl_loss` is the regularisation term. The former uses binary cross entropy, which is the information theoretical equivalent of a Bernoulli density, best seen by looking at the Keras [source implementation](https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/backend.py#L4668) 

```python
  # Compute cross entropy from probabilities.
  bce = target * math_ops.log(output + epsilon())
  bce += (1 - target) * math_ops.log(1 - output + epsilon())
  return -bce
```

The optimiser will minimise the loss function, and minimising the inverse of `bce` is equivalent to maximising the Bernoulli probability density. If the ground truth $x_i$ is denoted `target` and the predicted output is denoted `x_decoded_mean` then equation 4.6 is similar to the Keras BCE function

$$
\begin{equation}
\begin{aligned}

\log P_{\rm Bernoulli}(x_i \vert f(x_i \vert t)) 
& = {\rm target}_i \log f(x_i \vert t) + (1-{\rm target}_i) \log (1-f(x_i \vert t)) \\
& = - {\rm target}_i \log f({\rm x\_decoded\_mean}) + (1-{\rm target}_i) \log (1-{\rm x\_decoded\_mean})

\end{aligned}
\end{equation}
$$


Last, `kl_loss` implements the [closed form solution](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions) for the KL divergence of two gaussians. 






