---
title: My Machine Learnt... Now What? - Part 1
layout: post
---

## A. Introduction

### Motivations
There is a widely-accepted approach to building ML solutions that starts with learning parameters of a model and then applying a threshold to the model's scores to get labels. The thresholding part rests on an equation that balances two types of errors, with decision aided by so-called confusion matrices. For example, see this [tutorial](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/) - scroll down until "We can summarize this procedure below.".

The whole process is intuitive and can be applied to a number of problems. Over the last couple of years, however, I have stumbled upon applications that required more or less tweaking of this standard approach. For example, how should I determine the score thresholds of an ML system that is an input into a downstream system? What if the downstream task aggregates the upstream outputs, e.g. classifying paragraphs, to make decisions on documents?

Earlier this year, I came across the Speech Recognition Evaluation (SRE), an ML competition organised by [NIST](https://www.nist.gov/programs-projects/speaker-and-language-recognition), a U.S. research institute. Over the last two decades, its research community has developed a framework to assess speech recognition systems. One recurring application is to determine if a speech segment includes only one or two distinct speakers. 

Though NIST papers mention various applications of this evaluation framework in biometrics and forensics, I think that it is useful for anyone working on classification applications, and it surely helped me firm up my understanding of model evaluation. I particularly like that it combines solid theoretical foundations with implementations via the BOSARIS toolkit - see this [matlab implementation](https://projets-lium.univ-lemans.fr/sidekit/api/bosaris/index.html), or more recent ones in [julia](https://github.com/davidavdav/ROCAnalysis.jl) or [python](https://gitlab.eurecom.fr/nautsch/pybosaris).

The following two parts review traditional assessment frameworks, in particular the ROC curve, and the third part is an introduction to the Applied Probability of Error (APE), a framework for model calibration assessment.

### Fraud detection

I will refer to a recognizer as a black box that takes an input of data features and outputs a numerical value called score, such that higher values are more likely to be targets than not. A classifier binarizes the scores to return classes, also called labels or hard decisions.

To test and illustrate the evaluation frameworks, I use an imaginary problem of fraud detection with two types of transactions: 
- `Fraudster` also called positive classes, or target labels or $\omega_1$
- `Regular` also called negative classes, or non-target labels or $\omega_0$

This blog article is split into 3 parts - the first two parts are concerned with automated systems that output hard decisions, i.e. target or not, using available features, and the evaluation frameworks will help to construct labels given some risk objectives. Part 3 is concerned with systems that output calibrated results, in the form of probabilities or log-likelihood ratios.

There is a tendency among ML practitioners to default to hard decision systems without really asking if it is the best approach. I can only speculate what the reasons may be, but that removes any indication about how likely a class is given an instance, which is often a valuable piece of information in the decision-making process.

It is worth noting that a system may combine hard and soft decisions, for example, a fraud detection application that predicts if a transaction is non-fraudulent, fraudulent or to be investigated. The 3rd case may correspond to instances with a probability of, say, 0.4 to 0.7, which an analyst manually reviews. For more information, jump straight to [part 3]({{ site.baseurl }}{% link _posts/2022-03-02-Decisions-Part-3.markdown %}).

### Machine learning with scala

The code underlying these posts is available in my [github](https://github.com/mkffl/decisions) repository. The charts' definitions can be found in [Recipe.scala](https://github.com/mkffl/decisions/blob/main-pub/Decisions/src/Recipes.scala). 

Scala may not be an obvious choice for running stats and ML scripts, but the language has evolved quite a lot in recent years. What I have particularly enjoyed:

- Concise syntax - even more so in [scala 3](https://docs.scala-lang.org/scala3/book/why-scala-3.html) though this project is based on scala 2.13
- Rich standard library to explore and transform data - see some [examples](https://twitter.github.io/scala_school/collections.html)
- Range of ML and analytics libraries, e.g. this project uses [Smile](https://haifengl.github.io/) for ML models and a [Kotlin package](https://github.com/sanity/pairAdjacentViolators) for the Paired Adjacent Violators algorithm
- Scripting tools to get going quickly, this project is compiled with [Mill](https://github.com/com-lihaoyi/mill) and I have used the [Ammonite repl](https://ammonite.io/) to play around with the data
- Choice between functional programming - when I am feeling adventurous - and OOP - when I want to get things done quickly

### Simulation

I will rely on the `probability_monad` [package](https://github.com/jliszka/probability-monad) to test the findings using simulated data. By testing, I mean empirically validating a general statistical property by sampling from a random variable, which this package allows to do in a nice and concise way. It also comes with a useful toolkit, e.g. a `p` method to estimate probabilities.

Note that `probability_monad` does just what it says on the tin. It is not a framework for Bayesian inference, e.g. it does not include any routine like MCMC to infer distribution parameters, so you wouldn't use it for statistical inference or to train bayesian NN, but that still leaves out a number of interesting uses.

[Example.scala](https://github.com/jliszka/probability-monad/blob/master/src/main/scala/probability-monad/Examples.scala) in the github repo includes a number of common statistical fallacies - check out the [Monty Hall problem](https://github.com/jliszka/probability-monad/blob/1740054366b43c4e7a7c333bf8637daed11802bf/src/main/scala/probability-monad/Examples.scala#L254) for example - that can be written in just a few lines of code that make a lot of sense.

The bank transaction synthetic dataset consists of numerical features and a binary target field - Fraudster or Regular. The generative process for this data (DGP) is based on MADELON, an algorithm defined [here](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjygsHx9_nzAhXdQkEAHYjmBxUQFnoECAQQAQ&url=http%3A%2F%2Fclopinet.com%2Fisabelle%2FProjects%2FNIPS2003%2FSlides%2FNIPS2003-Datasets.pdf&usg=AOvVaw2e2nAV1wMjg-8TfNYk5z_d) that also underpins sklearn's `make_classification` [module](https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/datasets/_samples_generator.py#L39). I implement a [simplified version](https://github.com/mkffl/decisions/blob/e49290f5f01faadef2f4c383d663cfa28c457741/Decisions/src/Data.scala#L9) of MADELON that is enough to support my use case.

As its name suggests, the package is written using monads, a pillar of the functional programming paradigm that is dear to the scala community. A review of monads is beyond our scope, but it is also probably not necessary. I am by no means a monad expert, and yet I got started writing code in no time because most of the monad stuff is abstracted away. I can simply chain random variables to get whatever probabilistic graph I need.

### Plan

The next part briefly reviews a metric for discrimination ability, then we will jump into Bayes optimal decisions in C, and implement and validate an evaluation framework from scratch in D and E.

<h2 id="concordance">B. Concordance Metrics</h2>

If I have got some data and a trained recognizer that outputs scores, how good is it at differentiating between ${\omega_0}$ and ${\omega_1}$ instances? Let's start by looking at the distribution of scores by class, $p(s \vert w_i)$, and visually inspect if ${\omega_1}$ instances have higher scores. Good separation means that the two distributions barely overlap, and perfect separation means no overlap at all.

For example, fitting a Support Vector Machine (SVM) with default parameters on the transaction data gives the following separation on a validation dataset. The histograms estimate the score probability conditioned on the class, so I call it a Class Conditional Distributions (CCD) chart.

{% include demo11-ccd.html %}

Scores below -2 are almost certainly associated with non-targets, while scores between -1  and 1 are more uncertain. The concordance metric is the probability that  $\omega_0$ have lower scores than $\omega_1$ instance i.e. $p(s_{\omega_0} < s_{\omega_1})$. [TODO: show the value for SVM]

A naive implementation calculates the ratio of ordered pairs (i.e. $s_{\omega_0} < s_{\omega_0}$) over all pairs in the sample. If we call the probability A, that's what `naiveA` does in the code below.

[source](https://github.com/mkffl/decisions/blob/edc8cf34d8e3d82e7fdd1cdb48914e1bd1bfbbd3/Decisions/src/Evaluations.scala#L243)
```scala
  def getPermutations(A: Row, B: Row): Vector[Tuple2[Double, Double]] = for {
    a <- A
    b <- B
  } yield (a, b)

  /* count [score_w1 > score_w0] */
  def TarSupN(non: Row, tar: Row): Int = getPermutations(non, tar) filter {
    score => score._2 > score._1
  } size

  /* Estimate P(score_w1 > score_w0) */
  def naiveA(non: Row, tar: Row): Double = {
    val num = TarSupN(non, tar)
    val den = non.size * tar.size
    num / den.toDouble
  }
```

This implementation should provide an overview of the concordance probability, although it runs into memory issues when the number of observations becomes moderately large. The appendix describes a more efficient approach.

<h2 id="bayes-optimal">C. Bayes optimal decisions</h2>

### One feature
Suppose that we need to label instances using only one predictor. A randomly drawn dataset has one feature column $x$ and one target field $y$ with values in {$\omega_0, \omega_1$}. We want to decide on a classification rule that assigns new instance labels based on their $x$ values. 

Further, assume equal prior probabilities i.e. $p(\omega_0) = p(\omega_1) = 0.5$ and that the rule should minimise the number of misclassified instances. The class-conditional distributions $p(x \vert \omega_i)$ below suggests a rule.

{% include demo12-ccd.html %}

We should choose $\omega_0$ when its likelihood is greater, and $\omega_1$ otherwise. In this sample data, $p(x \vert \omega_1) > p(x \vert \omega_0)$ when $x>-0.25$, so any new instance with $x > -0.25$ should be classified as a target. Doing this consistently with every new instances guarantees that we'll minimise the error rate.

Any other rule corresponds to a higher probability of error. For example, if a new instance has $x=1.0$ then assigning $\omega_0$ would not make sense because the chances of being wrong are more than 4 times higher than being correct.

This line of reasoning is at the core of Bayes decisions criteria, which tells us that picking the option that minimises the risk for every single instance will minimise the overall risk. The common approach to selecting classifier thresholds implicitly relies on Bayes decision criteria, and the link will become obvious soon.

The following is based on the first chapter of [Pattern Classification](https://www.amazon.fr/Pattern-Classification-2e-RO-Duda/dp/0471056693). It's a bit of an old, albeit surprisingly refreshing textbook, which covers Bayes decisions in depth. I simplify their notation to the minimum necessary for a binary classification problem, though they address more general problems.

Making decisions when the environment is uncertain requires to consider the costs of different actions under uncertain states of nature:

<div class="mermaid"> 
    graph LR;
    id0([start]) --> id11[class ω0];
    id0([start]) --> id12[class ω1];
    id11[class ω0]-->id21[α0]; 
    id11[class ω0]-->id22[α1]; 
    id12[class ω1]-->id31[α0]; 
    id12[class ω1]-->id32[α1];
    id21[action α0] --> id41[cost c_00 = 0.0];
    id22[action α1] --> id42[cost c_10 = Cfa];
    id31[action α0] --> id43[cost c_01 = Cmiss];
    id32[action α1] --> id44[cost c_11 = 0.0];
</div>

The first node corresponds to choosing action $\alpha_0$ when the true state is $\omega_0$, i.e. choosing non-target when the instance is actually a non-target, which we assume does not cost anything. The next entry corresponds to a false alarm, with cost Cfa. The next is when we miss a target, which has a cost of Cmiss, and the last node corresponds to a true positive, which does not cost anything.

For every instance with feature $x$, the Bayes decisions to choose $\alpha_0$ or $\alpha_1$ depends on their respective risk, which in turn depend on the (posterior) probability of each state of nature.

$$
risk(\alpha_0 | x) = c_{00}*p(\omega_0|x) + Cmiss*p(\omega_1|x) = Cmiss*p(\omega_1|x) \\
risk(\alpha_1 | x) = c_{11}*p(\omega_1|x) + Cfa*p(\omega_0|x) = Cfa*p(\omega_0|x)
$$

So, we choose $\alpha_1$ if $risk(\alpha_1 \vert x) < risk(\alpha_0 \vert x)$ and $\alpha_0$ otherwise. Reorganising the terms gives the following rule:

<p id="rule-1-1">Bayes-optimal decision rule</p>

$$
\text{For any new instance x, choose}
\begin{cases}
    \alpha_1 \text{if } \frac{ p(x \vert \omega_1)}{p(x \vert \omega_0)} > \frac{p(\omega_0)*Cfa}{p(\omega_1)*Cmiss} \\
    \alpha_0, & \text{otherwise}
\end{cases}
\tag{1.1}
$$

We can rephrase the rule as "Decide on 'target' if the likelihood ratio for 'target' is greater than the cost-adjusted ratio of prior probabilities". The rule neatly separates the ratio of likelihoods (on the left) from the characteristics of a particular application (on the right). From now, I will call these characteristics application types to follow the NIST SRE terminology.

The left-hand side of the rule depends on the likelihood ratio that was learnt from a sample of data. This ratio tells us how much more or less likely an observed $x$ value is when it's generated by $\omega_1$ vs the alternative state. This relationship can be learnt from any dataset, no matter the prior probability $p(\omega_0)$. The right-hand side does not depend on instance data, but on the application-wide characteristics - target prevalence and error costs. It is a threshold above which we should choose $\alpha_1$ and that same threshold is used for every new instance.

The previous example with equal priors and the error rate objective corresponds to application parameters $p(\omega_1)=0.5; \text{Cmiss}=\text{Cfa}=1$. The error rate is the average risk when the two types of misclassification have the same cost (Cmiss=Cfa=1). The corresponding threshold of 1.0 matches our intuition to pick $\alpha_1$ when $x$ is to the right of the vertical line.

Higher $p(\omega_1)$ and/or higher Cmiss would increase the decision region for $\alpha_1$ by moving the decision cut-off to the left. If the cost of misclassifying targets gets bigger, or if there are more targets than non-targets, then we should lean more towards $\omega_1$.

It's worth noting that absolute parameter values do not impact the decision region, as only the relationship between 
$$
p(\omega_0)*Cfa
$$ 
and 
$$
p(\omega_0)*Cfa
$$ 
changes the RHS of the Bayes decision threshold.

<h3 id="more-than-one-features">More than one features</h3>
In general, there are multiple features and class-conditional sample density functions are more complex than above, possibly leading to many decision regions. Fortunately, the Bayes decision procedure applies to a recognizer's scores. In fact, we could think of a recognizer as a map from a large to a one-dimensional feature space, $f: \mathbb{R}^d \mapsto \mathbb{R}$. 

Furthermore, the map returns scores such that higher values correspond to a more likely $\omega_1$ state. In J. Hanley and B. McNeil (1982), a score is also called a degree of suspicion, which I think captures well the idea of ordered values and their relationship to hypothesis testing. 

I think of a score as the noise that a patient would make when a doctor gently presses parts of a sore spot to locate a sprained tendon. A stronger shriek means that the doctor is getting closer to the torn ligament, however, its intensity doesn't tell us how far the instrument is from the damaged tendon.

Applying the Bayes decision procedure to scores means finding the cut-off point where the likelihood ratio equals the theshold, 
$$
\frac{ p(s \vert \omega_1)}{p(s \vert \omega_0)} = \frac{p(\omega_0)*Cfa}{p(\omega_1)*Cmiss}
$$

Any score $s$ above the threshold should be assigned $\alpha_1$ i.e. classified as a target.

In practice, the logarithm is applied to the likelihood ratio, and I will use the BOSARIS threshold notation 
$$
\theta = \log\frac{p(\omega_1)*\text{Cmiss}}{p(\omega_0)*\text{Cfa}}
$$ 
(Notice the inverted numerator vs the Bayes decision rule). Hence, the cutoff point $c$ is at $llr = -\theta$.


### Implementations

Before, we found an optimal threshold at -0.25 by simply looking at the CCD chart. But eyeballing a graph is not reliable and becomes difficult if we move away from the simple case of equal priors and costs set to 1. Let's write a procedure, first using log-likelihood ratios, then using cumulative error probabilities, Pmiss and Pfa. The latter approach provides not only an optimal cut-off $\text{c}$ but also the corresponding expected risk at that threshold. The last section simulates hundreds of end-to-end applications to validate the estimated average risk. I find simulation to be helpful to take a step back and see the full deployment process.

The next use cases will be based on the following application type with prior probabilities assumed to be equal and missing targets costs 5 times more than missing non-targets.

```scala
val pa = AppParameters(p_w1=0.5,Cmiss=25,Cfa=5)
````

To find the optimal threshold, we need a few things:

a) The histogram counts of scores - the same as the CCD chart - which consists of the 3 vectors below; The histogram implementation is hidden as it's not particularly exciting

[source](https://github.com/mkffl/decisions/blob/edc8cf34d8e3d82e7fdd1cdb48914e1bd1bfbbd3/Decisions/src/Evaluations.scala#L151)
```scala
val w0Counts: Row = ... // counts of non-target labels
val w1Counts: Row = ... // counts of target labels
val thresh: Row = ... // bins
```

b) Common operations applied to sample data density, most of which will come in handy in later sections

[source](https://github.com/mkffl/decisions/blob/edc8cf34d8e3d82e7fdd1cdb48914e1bd1bfbbd3/Decisions/src/package.scala#L212)
```scala
val proportion: Row => Row = counts => {
    val S = counts.sum.toDouble
    counts.map(v => v / S)
}
val cumulative: Row => Row = freq => freq.scanLeft(0.0)(_ + _)
val oneMinus: Row => Row = cdf => cdf.map(v => 1 - v)
val decreasing: Row => Row = data => data.reverse
val odds: Tuple2[Row, Row] => Row = w0w1 =>
    w0w1._1.zip(w0w1._2).map { case (non, tar) => tar / non }
val logarithm: Row => Row = values => values.map(math.log)

val pdf: Row => Row = proportion
val cdf: Row => Row = pdf andThen cumulative
val rhsArea: Row => Row = cdf andThen oneMinus
val logodds: Tuple2[Row, Row] => Row = odds andThen logarithm
```

c) A class that encapsulates sample score predictions and the related evaluation methods, starting with the CCD estimates 

[source](https://github.com/mkffl/decisions/blob/edc8cf34d8e3d82e7fdd1cdb48914e1bd1bfbbd3/Decisions/src/Evaluations.scala#L159)
```scala
  case class Tradeoff(w1Counts: Row, w0Counts: Row, thresholds: Row) {

    val asCCD: Matrix = {
      val w0pdf = pdf(w0Counts)
      val w1pdf = pdf(w1Counts)
      Vector(w0pdf, w1pdf)
    }
```

The `asCCD` value provides the class-conditional density estimates previously used in the plots. It computes the proportion of counts corresponding to every threshold, and is readily available from histogram counts and the probability density transform `pdf`.

#### i) Using log-likelihood ratios

Next, we implement `minS`, a method to find the Bayes decision cut-off point - the score value $\text{s}$ that minimises the expected risk given our application type.

`asLLR` gives the log-likelihood ratio of the scores, which is the left-hand side of [eq. 1.1](#rule-1-1), and `minusθ` converts the application parameters into $-\theta$, which is the right-hand side of [eq. 1.1](#rule-1-1). `argminRisk` uses these two inputs to find the array index of the closest match, then used by `minS` to provide the cut-off point $c$.

```scala
case class Tradedoff(...){
        // ...

    val asLLR: Row = {
      val infLLR = logodds((pdf(w0Counts), pdf(w1Counts)))
      clipToFinite(infLLR)
    }

    def argminRisk(pa: AppParameters): Int =
      this.asLLR.getClosestIndex(minusθ(pa))

    def minS(pa: AppParameters): Double = {
      val ii = argminRisk(pa)
      thresholds(ii)
    }

}

def paramToθ(pa: AppParameters): Double = 
    log(pa.p_w1/(1-pa.p_w1)*(pa.Cmiss/pa.Cfa))

def minusθ(pa: AppParameters) = -1*paramToθ(pa)
```

<h4 id="using-pmiss-pfa">ii) Using Pmiss and Pfa</h4>

We can estimate the expected risk of a Bayes decision classifer using the evaluation data:

<p id="rule-1-2">The expected risk depends on two error rates, Pmiss and Pfa</p>

$$
E(\text{risk}) = \text{Cmiss}*p(\omega_1)*\text{Pmiss} + \text{Cfa}*p(\omega_0)*\text{Pfa}
\tag{1.2}
$$

where Pmiss is the proportion of targets with scores below the Bayes decision cutoff, $c$, and Pfa is the proportion of non-targets with scores above $c$. 

Getting the expected risk at every threshold makes it possible to find the optimal cut-off and the corresponding risk estimate. That approach made more sense to me after I wrote the proportions as a function of the threshold.

$$
E(\text{risk(c)}) = \text{Cmiss}*p(\omega_1)*\text{Pmiss(c)} + \text{Cfa}*p(\omega_0)*\text{Pfa(c)}
\tag{1.3}
$$

A note on the derivation of the expected risk - the Pmiss and Pfa notation comes from the BOSARIS documentation, however, the formula is used in many assessment frameworks and is quite intuitive. That is perhaps why its derivation is often not documented, so I include it below.

The expected risk is the risk of every action chosen for all instances in the data. But the action chosen depends on the instance region, $R_0$ and $R_1$:

$$
\begin{equation}
\begin{aligned}
E(\text{risk}) 
& = \int_{-\infty}^{+\infty} \text{risk}(\alpha(x) | x)*p(x) dx \\
& = \int_{x \in R_0} \text{risk}(\alpha_0 | x)*p(x) dx + \int_{x \in R_1} \text{risk}(\alpha_1 | x)*p(x) dx
\end{aligned}
\end{equation}
$$

With binary scores, the region $R_i$ is determined by the Bayes cutoff $c$, and every instance $x$ is mapped to a score $s$, so instead of sliding through the $x$'s we slide through the $s$'s:

$$
\begin{equation}
\begin{aligned}
E(\text{risk})
& = \int_{s < c} \text{risk}(\alpha_0 | s)*p(s) ds + \int_{s > c} \text{risk}(\alpha_1 | s)*p(s) ds \\
& = \text{Cmiss}*\int_{s < c} p(\omega_1 | s)*p(s) ds + \text{Cfa}*\int_{s > c} p(\omega_0 | s)*p(s) ds \\
& = \text{Cmiss}*p(\omega_1)\int_{s < c} p(s | \omega_1) ds + \text{Cfa}*p(\omega_0)*\int_{s > c} p(s | \omega_0) ds
\end{aligned}
\end{equation}
$$

And we get eq. 1.2 because the first and second integrals are estimated with Pmiss and Pfa, respectively. If $N_{\omega_1}$ is the number of targets, then
$$
\text{Pmiss} = \sum_{s \in \omega_1}\frac{[s < c]}{N_{\omega_1}}
$$ 
and 
$$
\text{Pfa} = \sum_{s \in \omega_0}\frac{[s >= c]}{N_{\omega_0}}
$$

Note that $\text{Pmiss}(c)$ is the cdf of the target distribution while $\text{Pfa}(c)$ is 1 minus the cdf of the non-targets distribution, hence the code implementation for `asPmissPfa`

```scala
case class Tradeoff(...){
    // ...
    val asPmissPfa: Matrix = {
      val pMiss = cdf(w1Counts)
      val pFa = rhsArea(w0Counts)
      Vector(pMiss, pFa)
    }

    def minRisk(pa: AppParameters): Double = {
      val ii = argminRisk(pa)
      val bestPmissPfa =
        (this.asPmissPfa.apply(0)(ii + 1), this.asPmissPfa.apply(1)(ii + 1))
      paramToRisk(pa)(bestPmissPfa)
    }
}

def paramToRisk(
    pa: AppParameters
)(operatingPoint: Tuple2[Double, Double]): Double =
  pa.p_w1 * operatingPoint._1 * pa.Cmiss + (1 - pa.p_w1) * operatingPoint._2 * pa.Cfa

```

## E. Tests

Let's check that the calculations for $c$ and $E(\text{risk})$ return sensible results.

### Optimal threshold

`expectedRisks` calculates risks at every threshold. We want the minimum risk to match the risk at $c$ on a given sample.

Below, the bottom graph plots `expectedRisks`, which confirms that $c$ gives the minimum, so using llr is equivalent to using the Pmiss/Pfa approach.

[source](https://github.com/mkffl/decisions/blob/e49290f5f01faadef2f4c383d663cfa28c457741/Decisions/src/Evaluations.scala#L175)
```scala
case classs Tradeoff(...){
    //...

    def expectedRisks(pa: AppParameters): Row = {
      val risk: Tuple2[Double, Double] => Double = paramToRisk(pa)
      this.asPmissPfa.transpose.map { case Vector(pMiss, pFa) =>
        risk((pMiss, pFa))
      }
    }  
}
```

{% include demo14-bayesdecisions1.html %}

### Expected risk

Now, let's check that the estimated risk is reliable, i.e. if we use the corresponding cut-off on new instances, do we get close to the sample expected risk? 

I will simulate transactions to show that the optimal threshold is reliable. Note that the result matters less than the process used to get there. As the expected value of a random variable, minRisk will be close to its sample average, but writing this random variable explicitly can help step out from the details and see the big picture again.

All the steps are grouped into one random variable that generates a data instance, makes a hard prediction and calculates the corresponding risk.

```scala
    /** Expected risk simulation
      *
      * @param nRows the number of rows in the simulated dataset
      * @param pa the application type
      * @param data the transaction's data generation process
      * @param classifier a predictive pipeline that outputs the user type
      */
    def oneClassifierExpectedRisk(
        nRows: Integer,
        pa: AppParameters,
        data: Distribution[Transaction],
        classifier: (Array[Double] => User)
    ): Distribution[Double] = data
      .map { transaction =>
        {
          val binaryPrediction = classifier(
            transaction.features.toArray
          ) // Generate a transaction's predicted user and
          val dcf = cost(
            pa,
            transaction.UserType,
            binaryPrediction
          ) // calculate its dcf
          dcf
        }
      }
      .repeat(nRows) // Generate a dataset of dcf's
      .map { values =>
        values.sum.toDouble / nRows // Get the average dcf
      }
```

To run the above, we need a classifier and a cost function. 

The classifier is a predictive pipeline that applies the recognizer on a transaction and applies the optimal cut-off $c$ to predict the user type.

```scala
    val cutOff: Double = hisTo.minS(pa)
    val thresholder: (Double => User) = score =>
      if (score > cutOff) { Fraudster }
      else { Regular }

    def classifier: (Array[Double] => User) =
      recognizer andThen logit andThen thresholder
```

The cost function for one instance, also called Detection Cost Function (DCF), simply applies the error cost to an actual decision.

```scala
    // Simulate the risk of one transaction
    def cost(p: AppParameters, actual: User, pred: User): Double = pred match {
      case Fraudster if actual == Regular => p.Cfa
      case Regular if actual == Fraudster => p.Cmiss
    }
```

Note that `classifier` applies a `logit` transform to the `recognizer`'s output. That is because SVM scores are originally in $[0,1]$, and I want them in $\mathbb{R}$ to emphasize that scores need not be "probability-like" values - they could also be projected onto $[0,inf]$ for example. The logit is a monotonously increasing function, so it does not affect the score index returned by the `minS` method.

Let's sample 500 evaluations and plot the results.

```bash
@ oneClassifierExpectedRisk(1000, pa, transact(pa.p_w1), classifier).sample(500)
```


{% include demo13-simulation.html %}

The evaluation sample risk is the vertical dotted bar near the peak of the distribution, which is good news. The grey shaded area comprises 95% of the simulated data, and I want the sample estimate to fall in that interval, which it does.

The [next part]({{ site.baseurl }}{% link _posts/2021-10-28-Decisions-Part-2.markdown %}), explores the connection between Bayes decision rules and the Receiving Operator Characteristics (ROC) curve.

## References
- R. Duda et al (2001), Pattern Classification.
- J. Hanley and B. McNeil (1982), The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve.

### Appendix

#### Efficient implementation of the concordance probability

The inefficient bit is to generate all pairs, an operation that I isolate in `getPermutations`. Its complexity is a function of $N_{non}*N_{tar}$ if there are $N_{non}$ non-target instances and $N_{tar}$ target instances.

The good news is that some people came up with a clever method to reduce the complexity. It is implemented in the `smartA` method below. Feel free to skip the following details if they are not of interest. 

The key is that we can count [$s_{\omega_0} < s_{\omega_1}$] by summing the ranks of the target instances in the combined, sorted samples of instance scores, then subtracting the sum of ranks in the target sub-sample. The resulting number figure is called U and happens to also be a test statistics, for a test called Wilcoxon. Its null hypothesis is not necessary to our discussion.

The expensive part of the rank-sum approach is to rank instances, but that can be done efficiently with built-in sorting algorithms, so I guess the overall complexity is $log(N_{non}+N_{tar})$ and, in any case, it runs fast on my modest 1.6 GHz Intel Core i5. 

I find this approach very cool because it's not only clever but also intuitive. In the combined, sorted dataset of scores, we can count the concordance value for a single target instance by taking its overall rank and subtracting its rank in the subsample of target instances. If we repeat this procedure for all target instances and rearrange the operations, we get the value computed by `U`.

[source](https://github.com/mkffl/decisions/blob/edc8cf34d8e3d82e7fdd1cdb48914e1bd1bfbbd3/Decisions/src/Evaluations.scala#L267)
```scala
  def rankAvgTies(input: Row): Row // See source code

  /* Wilcoxon Statistic, also named U */
  def wmwStat(s0: Row, s1: Row): Int = {
    val NTar = s1.size
    val ranks = rankAvgTies(s0 ++ s1)
    val RSum = ranks.takeRight(NTar).sum
    val U = RSum - NTar * (NTar + 1) / 2
    U toInt
  }

  /* Estimate P(score_w1 > score_w0) */
  def smartA(non: Row, tar: Row) = {
    val den = non.size * tar.size
    val U = wmwStat(non, tar)
    val A = U.toDouble / den
    A
  }
```