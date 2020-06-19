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

With a discrete latent space the number of parameters to estimate depend on the number of mixture components. With a continuous space there is no finite number of mixture components so instead the parameters to estimate are the coefficiences of the transformation from $t$ to 
the parameters of the mixture random variable, e.g. probability of success $p$ for a Bernoulli rv. 

The number of parameters thus depends on the transformation function, $f$. This function can be as simple as a logit although most applications would use deep neural networks to address non-linear data structures like images or sound waves. The mixture distribution is said to be **parametrised** by the function $f$, which in the absence of a closed form solution, is typically optimised with stochastic gradient descent (SGD).

For a given latent variable $t$ and a set of weights $\theta$ the MNIST digit geneated is assumed to be distributed as

$$
x_i \sim \rm Bernoulli(f(t \vert \theta))
$$

and as with previous examples the log-likelihood can be marginalised out

$$
\sum_{i=1}^N \log P(x_i|\theta)
$$

$$
= \sum_{i=1}^N \log \int  P(x_i, t \vert \theta) dt
$$

$$
= \sum_{i=1}^N \log \int P_{\mathcal{N}}(t \vert 0, I) * P_{\rm Bernoulli}(x_i \vert f(t \vert \theta)) dt
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

$
\sum_{i=1}^N \log P(x_i|\theta) = L(\theta) + KL(q(t)\parallel p(t|x,\theta))
$

As with discrete mixtures, the objective function can be written as the sum of a lower bound and a measure of the divergence between the latent variable posterior distribution and the arbitrary distribution $q$. In the handwritten digit example, the posterior probability returns the most likely latent value $t$ for a given MNIST picture.

The lower bound is 

$
L(\theta) = \sum_i^N\{ \int q(t) \log P(x_i, t) dt - \int q(t) \log q(t) dt \}
$

We will assume to have a good approximation of the posterior distribution, $q_{magic}(t \vert x_i) \approx p(t_i \vert x_i)$. $q_{magic}$ is a temporary placeholder to focus on the similarities with EM. Solving for $\theta$ then corresponds to

$$
{argmax}{\theta} L(\theta) = \sum_i^N\ \int q_{magic}(t) \log P(x_i, t) dt
$$


As with the discrete mixture case this is close to the observed component likelihood function with $q_{magic}$ filling in the gap for the latent variable. Applying the distributional assumptions for the MNIST example and approximating the integral gives 

$$
\approx \sum_i^N\ \sum_{m=1}^M q_{magic}(t_m) \log \{ P_{\mathcal{N}}(t \vert 0, I) * P_{\rm Bernoulli}(x_i \vert f(x_i \vert t, \theta)) \} dt
$$

This quantity, which can be evaluated with stochastic gradient methods, contrasts with the direct log-likelihood function above because sampling over the approximate posterior is more efficient than sampling over the prior distribution. Instead of looping through many unlikely $t$ values, the model is forced to loop throuh the most likely latent values that generated the observations, which $q_{magic}$ does as an approximation of the posterior.


Finally, the VAE optimises a rearranged version of the lower bound that is more aligned with the machine learning view. Moving the prior probability $p(t)$ to the RHS expression the lower bound can be written as 

$$
\sum_i^N\ \int q(t \vert x_i) \log P_{\rm Bernoulli}(x_i \vert f(x_i \vert t, \theta)) dt  - KL(q(t \vert x_i) \parallel P_{\mathcal{N}}(t \vert 0, I))
$$

The previous form, sometimes called "free energy", can be seen as maximising the observed component function even as latent components are not observed. As described with the Poisson mixture, the posterior $q$ is the best substitute for the unobserved latent variable.

The new form, sometimes called "penalised model fit", can be seen as a regularised function, e.g. as with the Lasso regression where L1 acts as the regularisation term. The LHS expression measures the model fit through the marginal probability, while the KL divergence as a regularisation term as it forces the posterior approximation close to an isometric gaussian.


#### Variational Bayes

EM sets $q$ equal to the posterior $p(t  \vert x)$ to bring the lower bound to the log-likelihood using Bayes' formula, which was possible in the presence of a close form solution for mixture distributions.

As the VAE distribution is parametrised with a complex function $f$, this is no longer an option, however it is possible to approximate the posterior and this is where EM and the VAE optimising scheme differ.

The short answer for the solution to $q$ is that VAEs approximate the posterior distribution with a simple random variable, typically a symmetric gaussian distribution. As with the function $f$ described above, the posterior distribution is "parametrised" by a neural network function $g$, with weights $\phi$. The function $g$ and its parameters are plugged into the lowerbound, with SGD optimising jointly $\theta$ and $\phi$.

Two questions may be Why does this approach converge to local or global maxmima and How is it implemented in practice? The first question, which I find more interesting, is not covered here as the answer would take more than the few paragraphs left - I am also still looking for the right Donkey Kong chart to visualise it. The second question is well covered in the original paper and in subsequent blog articles with examples listed at the bottom. Some elements are covered now.


How is it possible to estimate the real posterior with an approximating function if the posterior is not observed? Well, in a sense this is what statistical inference is all about - guessing the values of invisible parameters by doing stuff with the visible output. If, like myself, you are not familiar with variational inference, it may feel unsettling because the variational function to estimate is an input into another function with weights that we also estimate. So this problem has a lot of parameters, but hopefully it comes with a lot of data.

$$
q(t \vert \ x_i) \sim \mathcal{N} (\mu, \Sigma)
$$

$$
\mu = g_{\mu} (x_i \vert \phi)
$$

$$
\Sigma = g_{\Sigma} (x_i \vert \phi)
$$


$
\sum_i^N\ \int P_{\mathcal{N}}(t \vert \mu, \Sigma) \log P_{\rm Bernoulli}(x_i \vert f(x_i \vert t, \theta)) dt  - KL(P_{\mathcal{N}}(t \vert \mu, \Sigma) \parallel P_{\mathcal{N}}(t \vert 0, I))
$

$
\approx \sum_i^N\ \log P_{\rm Bernoulli}(x_i \vert f(x_i \vert t^{\ast}, \theta)) dt  - KL(P_{\mathcal{N}}(t \vert \mu, \Sigma) \parallel P_{\mathcal{N}}(t \vert 0, I))
$

$
\approx \sum_i^N\ \log P_{\rm Bernoulli}(x_i \vert f(x_i \vert t = g_{\mu} (x_i \vert \phi) + g_{\Sigma} (x_i \vert \phi) * \epsilon^{\ast}, \theta)) dt - KL(P_{\mathcal{N}}(t \vert \mu, \Sigma) \parallel P_{\mathcal{N}}(t \vert 0, I))
$



The parameters of $g$ are estimated using mini-batch stochastic gradient descent for both $\phi$ and $\theta$. The details for how and why this schema converges are found in the variational inference literaturre which is beyond the scope of this text. 


