---
title: My Machine Learnt... Now What? - Part 1
layout: post
---

## A. Introduction

### Motivations
There is a widely-accepted approach to building ML solutions that starts with learning parameters of a model and then applying a threshold to the model's scores to get categories/labels - see [this example](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/). The thresholding part rests on an equation that balances two types of errors, with decision aided by so-called confusion matrices.

The whole process is rather intuitive and can be applied almost like a recipe to a number of problems. Over the last couple of years, however, I have stumbled upon applications that required more or less tweaking of the standard approach. For example, how should I determine the score thresholds of an ML sytem that is an input into a downstream system? What if the downstream task aggregates the upstream outputs, e.g. classifying paragraphs to make decisions on documents?

As I brushed up on traditional frameworks, I discovered the Speech Recognition Evaluation (SRE), a ML competition organised by an U.S. Institute called [NIST](https://www.nist.gov/programs-projects/speaker-and-language-recognition). Over the last two decades, the research community around this institution has developed a framework to assess speech recognition systems.

One recurring application is to determine if a speech segment includes only one or two distinct speakers. Though the NIST often refers to various applications of its evaluation framework in biometric and forensic contexts, I think that the concepts and tools developed are useful for anyone working on classification applications. The set of research that I found, which I list at the end, certainly helped me firm up my understanding of model evaluation. I particuarly like that it combines solid theoretical foundations with implementations via the BOSARIS toolkit - links also at the end.

The following two parts review traditional assessment frameworks, in particular the ROC curve, and the third part is an introduction to the Applied Probability of Error (APE), a framework for model calibration assessment.

### Fraud detection

Following the NIST SRE literature, I refer to a recognizer as a black box that takes an input of data features and outputs a numerical value called score, such that higher values are more likely to be targets than not. A classifier binarizes scores to return classes or labels.

I use an imaginary problem of fraud detection to test and illustrate the evaluation frameworks. It is a binary classification example with two types of transactions: 
- `Fraudster` also called positive classes, or target labels or $\omega_1$
- `Regular` also called negative classes, or non-target labels or $\omega_0$


In the first 2 parts, the objective is to build an automated system which predicts if a transaction is a target or a non-target given observed features, and the evaluation frameworks will help construct labels that meet our risk objectives. Part 3 adds some nuance to binary classification and will consider the need and means to output probabilities instead.

### Machine learning with scala

The code underlying these posts is written in scala, and one may wonder why. A proper answer is beyond the scope of this article, a very short answer is "Why not?", and a more helpful answer is that scala has a few advantages over scripting languages, which people working with machine learning or statistical analysis typically use, e.g. python, R, julia, matlab or stata.

As someone who works primarily with scripting languages, and is not a software developer by trade, I occasionally need to write code that runs on the java virtual machine (jvm), which often supports enterprise backend systems. So far, I have not found java particularly attractive to learn, while scala offers
- (More) Concise syntax
  - Even more so with scala 3
- A rich native library to transform data
  - The collections library
- An ecosytem of ML libraries 
  - Smile, (other TBD)
- Scripting tools to get going quickly
  - , e.g. I used Mill to compile this project and Ammonite to play around with data and models

### Simulation

I will rely on the `probability_monad` scala package to test some of the findings using simulated data. By testing, I mean validating a (general) statistical property using an example distribution. The package allows me to define the distribution in a nice and concise way, and it comes with a useful toolkit, e.g. a `p` method to estimate probabilities.

This package does just what it says on the tin. It's not a framework for Bayesian inference, e.g. it does not include any routine like MCMC to infer distribution parameters, but it is useful for learning purposes. [Example.scala](https://github.com/jliszka/probability-monad/blob/master/src/main/scala/probability-monad/Examples.scala) in the github repo includes a number of common statistical fallacies - check the [Monty Hall problem](https://github.com/jliszka/probability-monad/blob/1740054366b43c4e7a7c333bf8637daed11802bf/src/main/scala/probability-monad/Examples.scala#L254) for example - that can be written in just a few lines of code that make a lot of sense.

I will illustrate the evaluation frameworks using a synthetic dataset of banking transactions that consists of numerical features and a binary target field. The target corresponds to a transaction being fraudulent or not. The generative process for this data (DGP) is based on MADELON, an algorithm defined [here](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjygsHx9_nzAhXdQkEAHYjmBxUQFnoECAQQAQ&url=http%3A%2F%2Fclopinet.com%2Fisabelle%2FProjects%2FNIPS2003%2FSlides%2FNIPS2003-Datasets.pdf&usg=AOvVaw2e2nAV1wMjg-8TfNYk5z_d) that also underpins sklearn's `make_classification` [module](https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/datasets/_samples_generator.py#L39). I implement a simplified version of the DGP that fits the need of my use case (link to my source code).

As its name suggests, the package is written using monads, a pillar of the functional programming paradigm that is dear to the sala community. A review of monads is beyond our scope but I actually believe it is not necessary. I am not a monad expert yet writing a distributions with `probability_monad` felt intuitive because I could just plug random variables in a simple manner and get whatever probabilistic graph I wanted.

## B. Concordance Metrics

Given a recognizer that outputs scores, we may ask if the tool can differentiate between ${\omega_0}$ and ${\omega_1}$ instances. We can first look at the distribution of scores by class, $p(s \vert w_i)$, and visually inspect if ${\omega_1}$ instances have higher scores. Good separation means that the two distributions barely overlap, and perfect separation means no overlap at all.

For example, fitting a Support Vector Machine (SVM) with default parameters on the transaction data gives the following separation. The histograms estimate the score probability conditioned on the class so I call it a Class Conditional Distributions (CCD) chart.

{% include demo11-ccd-5.html %}

Scores below -2 are almost certainly associated with non-targets while scores between -1  and 1 are more uncertain. We can measure the recognizer's discrimination power in one metric that estimates the probability that $\omega_0$ have lower scores than $\omega_1$ instance i.e. $p(s_{\omega_0} < s_{\omega_1})$.

An naive implementation estimates the probability with a simple ratio of ordered pairs (i.e. $s_{\omega_0} < s_{\omega_0}$) over all pairs in the sample. If we call the probability A, that's what `naiveA` does in the code below.

```scala
    def getPermutations(A: Row, B: Row): Vector[Tuple2[Double,Double]] = for {
            a <- A
            b <- B
        } yield (a,b)

    /* count [score_w1 > score_w0] */
    def TarSupN(non:Row, tar:Row): Int = getPermutations(non,tar) filter {score => score._2 > score._1} size
    
    /* Estimate P(score_w1 > score_w0) */
    def naiveA(non: Row, tar: Row): Double = {
        val num = TarSupN(non,tar)
        val den = non.size*tar.size
        num/den.toDouble
    }
```

Unfortunately, this method stumbles upon memory issues when the number of observations becomes moderately large. The computationally expensive part is to generate all pairs, an operation that I isolate in `getPermutations`. Its complexity is a function of $N_{non}*N_{tar}$ if there are $N_{non}$ non-target instances and $N_{tar}$ target instances.

The good news is that some people came up with a clever method to make the computation less expensive. It is implemented in the `smartA` method below. Feel free to skip the following details if they are not of interest. 

The key is that we can count [$s_{\omega_0} < s_{\omega_1}$] by summing the ranks of the target instances in the combined, sorted samples of instance scores, then subtracting the sum of ranks in the target sub-sample. The resulting number figure is called U and happens to also be a test statistics, for a test called Wilcoxon. Its null hypothesis is relevant to our topic but I will not indulge another digression. 

The expensive part of the rank-sum approach is to rank instances, but that can be done efficiently with built-in sorting algorithms, so I guess the overall complexity is $log(N_{non}+N_{tar})$ and, in any case, it runs fast on my modest 1.6 GHz Intel Core i5. 

I find this approach very cool because it's not only clever but also intuitive. In the combined, sorted dataset of scores, we can count the concordance value for a single target instance by taking its overall rank and subtracting its rank in the subsample of target instances. If we repeat this procedure for all target instances and rearrange the operations, we get the value computed by `U`.

```scala
    def rankAvgTies(input: Row): Row // See source code

    /* Wilcoxon Statistic, also called U */
    def wmwStat(s0: Row, s1: Row): Int = {
        val NTar = s1.size
        val ranks = rankAvgTies(s0 ++ s1)
        val RSum = ranks.takeRight(NTar).sum
        val U = RSum - NTar*(NTar+1)/2
        U toInt
    }
    
    /* Estimate P(score_w1 > score_w0) */
    def smartA(non:Row, tar:Row) = {
        val den = non.size*tar.size
        val U = wmwStat(non,tar)
        val A = U.toDouble/den
        A
    }
```

Demo 12 (AUC)



## C. Bayes optimal decisions
### One feature
Suppose that we need to label instances using only one predictor. A randomly drawn dataset has one feature column $x$ and one target field $y$ with values in $\{\omega_0$, $\omega_1\}$. We want to decide on a classification rule that assigns new instance labels based on their $x$ values. 

Further assume equal prior probabilities i.e. $p(\omega_0) = p(\omega_1) = 0.5$ and that the rule should minimise the number of misclassified instances. The class-conditional distributions $p(x \vert \omega_i)$ below suggests a rule.

{% include demo12-ccd-3.html %}

We should choose $\omega_0$ when its likelihood is greater, and $\omega_1$ otherwise. In this sample data, $p(x \vert \omega_1) > p(x \vert \omega_0)$ when $x>-0.25$, so any new instance with $x > -0.25$ should be classified as a target. Doing this consistently with every new instances guarantees that we'll minimise the error rate.

Any other rule corresponds to a higher probability of error. For example, if a new instance has $x=1.0$ then assigning $\omega_0$ would not make sense because the chances of being wrong are more than 4 times higher than being correct.

This line of reasoning is at the core of Bayes decisions criteria, which tells us that picking the option that minimises the risk for every single instance will minimise the overall risk. The common approach to selecting classifier thresholds implicitly relies on Bayes decision criteria and the link will become obvious soon if it's not already clear.

The following is based on the first chapter of [Pattern Classification](https://www.amazon.fr/Pattern-Classification-2e-RO-Duda/dp/0471056693). It's a bit old but also a surprisingly refreshing textbook that covers Bayes decision theory in depth. I simplify their notation to the minimum necessary for a binary classification problem, though they address more general problems.

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

[Units and interpretation of costs]

The first node corresponds to choosing action $\alpha_0$ when the true state is $\omega_0$, i.e. choosing non-target when the instance is actually a non-target, which we assume does not cost anything. The next entry corresponds to a false alarm, with cost Cfa. The next is when we miss a target, which has a cost of Cmiss, and the last node correspodns to a true positive, which does not cost anything.

For every instance with feature $x$, the Bayes decisions to choose $\alpha_0$ or $\alpha_1$ depends on their respective risk, which in turn depend on the (posterior) probability of each state of nature.

$$
risk(\alpha_0 | x) = c_{00}*p(\omega_0|x) + Cmiss*p(\omega_1|x) = Cmiss*p(\omega_1|x) \\
risk(\alpha_1 | x) = c_{11}*p(\omega_1|x) + Cfa*p(\omega_0|x) = Cfa*p(\omega_1|x)
$$

So, we should choose $\alpha_1$ if $risk(\alpha_1 \vert x) < risk(\alpha_0 \vert x)$ and $\alpha_1$ otherwise. Reorganising the terms gives the following Bayes decision rule:

$$
\text{For any new instance x, choose}
\begin{cases}
    \alpha_1 \text{if } \frac{ p(x \vert \omega_1)}{p(x \vert \omega_0)} < \frac{p(\omega_0)*Cfa}{p(\omega_1)*Cmiss} \\
    \alpha_0, & \text{otherwise}
\end{cases}
\tag{1.1}
$$

We can rephrase the rule as "Decide on 'target' if the likelihood ratio for 'target' is greater than the cost-adjusted ratio of prior probabilities". The rule neatly separates the ratio of likelihoods (on the left) from the characteristics of a particular application (on the right). From now, I will call these characteristics application parameters as in the BOSARIS literature.

The left-hand side of the rule depends on the likelihood ratio that was learnt from a sample of data. This ratio tells us how much more or less likely an observed $x$ value is when it's generated by $\omega_1$ vs the alternative state. This relationship can be learnt from any dataset, no matter the prior probability $p(\omega_0)$. The right-hand side does not depend on the observed feature value, it only depends on the application that we release the decision model on. It represents a threshold above which we should choose $\alpha_1$ and that same threshold is used for every new instance.

The previous example with equal priors and the error rate objective corresponds to application parameters $p(\omega_1)=0.5; Cmiss=Cfa=1$. The error rate is the average risk when the two types of misclassification have the same cost (Cmiss=Cfa=1). The corresponding threshold of 1.0 matches our intuition to choose $\alpha_1$ when $x$ is to the right of the vertical line.

Higher $p(\omega_1)$ and/or higher Cmiss would increase the decision region for $\alpha_1$ by moving the decision cut-off to the left. If the cost of misclassifying targets gets bigger, or if there are more targets than non-targets, then we should lean more towards $\omega_1$.

It's worth noting that absolute parameter values do not impact the decision region, as only the relationship between 
$$
p(\omega_0)*Cfa
$$ 
and 
$$
p(\omega_0)*Cfa
$$ 
changes the RHS of the Bayes decision threshold. This property will come up again in the APE framework.

### More than one feature
In general, there are more than one features and class-conditional sample density functions are more complex than above, possibly leading to many decision regions. Fortunately, the Bayes decision procedure applies to a recognizer's scores. In fact, we could think of a recognizer as a map from a large feature space to a one-dimensional space, $f: \mathbb{R}^d \mapsto \mathbb{R}$. 

Furthermore, the map returns scores such that higher values correspond to a more likely $\omega_1$ state. In J. Hanley and B. McNeil (1982), a score is also called a degree of suspicion, which I think captures well the idea of ordered values and their relationship to hypothesis testing. I think of a score as the noise that a patient would make when a doctor gently presses parts of a sore spot to locate a sprained tendon. A stronger shriek means that the doctor is getting closer to the torn ligament, however, its intensity doesn't tell us how far the instrument is from the damaged tendon.

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


### Implementation
[minS on Tradeoff object]

Going back to the SVM scores used in the CCD plot at the start, we would like to find the Bayes decision criterion $c$ for particular application parameters $\{p(\omega_1), p(\omega_0), \text{Cmiss}, \text{Cfa}\}$. The optimal criterion value $\text{c}$ will convert the recognizer into a classifier, which we will apply on simulated data to check that the solution's risk in the right ballpark area.

The parameters will be stored in a data object called `AppParameters` with the following values `val pa = AppParameters(p_w1=0.5,Cmiss=25,Cfa=5)`. Prior probabilities are assumed to be equal missed targets have a cost 5 times higher than missed non-targets.

We need a few extra bits in place:

- The histogram counts of scores - the same as the CCD chart - which consists of the 3 vectors below; The histogram implementation is not particularly interesting so I don't show it, but it's available in the source code

```scala
val w0Counts: Row = ... // counts of non-target labels
val w1Counts: Row = ... // counts of target labels
val thresh: Row = ... // bins
```

- Common operations applied to sample data density, most of which will come in handy in later sections

```scala
val proportion: Row => Row = counts => {
    val S = counts.sum.toDouble
    counts.map(v => v/S)
}

val cumulative: Row => Row = freq => freq.scanLeft(0.0)(_ + _)
val oneMinus: Row => Row = cdf => cdf.map(v => 1-v)
val decreasing: Row => Row = data => data.reverse
val odds: Tuple2[Row,Row] => Row = w0w1 => w0w1._1.zip(w0w1._2).map{case (non,tar) => tar/non}
val logarithm: Row => Row = values => values.map(math.log)        

val pdf: Row => Row = proportion
val cdf: Row => Row = pdf andThen cumulative
val rhsArea: Row => Row = cdf andThen oneMinus
val logodds: Tuple2[Row,Row] => Row = odds andThen logarithm
```

- An object to encapsulate the sample score predictions and the related evaluation methods, starting with the CCD estimates 

```scala
case class Tradeoff(w1Counts: Row, w0Counts: Row, thresholds: Row) {

    val asCCD: Matrix = {
        val w0pdf = pdf(w0Counts)
        val w1pdf = pdf(w1Counts)
        Vector(w0pdf,w1pdf)
    }
}
```

The `asCCD` value provides the class-conditional density estimates previously used in the plots. It really just computes the proportion of counts corresponding to every threshold. This is what the `pdf` function, an alias for `proportion`, does.

Next, we implement `minS`, a method to find the Bayes decision cut-off point, i.e. the score value $\text{s}$ that minimises the expected risk given some application parameters.

```scala
case class Tradedoff(...){
        // ...

        val asLLR: Row = {
            val infLLR = logodds((pdf(w0Counts),pdf(w1Counts)))
            clipToFinite(infLLR)
        }

        def argminRisk(pa: AppParameters): Int = this.asLLR.getClosestIndex(minusθ(pa))

        def minS(pa: AppParameters): Double = {
            val ii = argminRisk(pa)
            thresholds(ii)
        }

}

def paramToTheta(pa: AppParameters): Double = log(pa.p_w1/(1-pa.p_w1)*(pa.Cmiss/pa.Cfa))

def minusθ(pa: AppParameters) = -1*paramToTheta(pa)
```

`asLLR` returns the log-likelihood ratio of the scores - the left-hand side of eq. 1.1 - using the proportion of targets in the score's corresponding bin. The `minusθ` method convert the application parameters into $-\theta$, which is the right-hand side of eq. 1.1. Then, `argminRisk` uses these two inputs to find the array index of the closest match, which is used by `minS` to provide the cutoff point $c$.

### D. Average risk

We can estimate the expected risk of a Bayes decision classifer using the evaluation data:

$$
E(\text{risk}) = \text{Cmiss}*p(\omega_1)*\text{Pmiss} + \text{Cfa}*p(\omega_0)*\text{Pfa}
\tag{1.2}
$$

where Pmiss is the proportion of targets with scores below the Bayes decision cutoff, $c$, and Pfa is the proportion of non-targets with scores above $c$.

The Pmiss and Pfa notation comes from the BOSARIS documentation, however, the formula is used in many assessment frameworks and is quite intuitive. That is perhaps why its derivation is often not documented, so I include it below.

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

With binary scores, the region $R_i$ is determined by the Bayes cutoff $c$, and every instance $x$ is mapped to a score $s$, so instead of sliding through $x$ we slide through all $s$'s:

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
Pmiss = \sum_{s \in \omega_1}\frac{[s < c]}{N_{\omega_1}}
$$ 
and 
$$
Pfa = \sum_{s \in \omega_0}\frac{[s >= c]}{N_{\omega_0}}
$$

Note that $\text{Pmiss}(c)$ is the cdf of the target distribution while $\text{Pfa}(c)$ is 1 minus the cdf of the non-targets distribution, hence the code implementation for `asPmissPfa`

```scala
case class Tradeoff(...){
    // ...
    val asPmissPfa: Matrix = {
        val pMiss = cdf(w1Counts)
        val pFa = rhsArea(w0Counts)
        Vector(pMiss,pFa)
    }

    def minRisk(pa: AppParameters): Double = {
        val ii = argminRisk(pa)
        val bestPmissPfa = (this.asPmissPfa.apply(0)(ii+1),this.asPmissPfa.apply(1)(ii+1))
        paramToRisk(pa)(bestPmissPfa)
    }
}

def paramToRisk(pa: AppParameters)(operatingPoint: Tuple2[Double,Double]): Double = 
        pa.p_w1*operatingPoint._1*pa.Cmiss + (1-pa.p_w1)*operatingPoint._2*pa.Cfa

```

## E. Tests

Let's check that the calculations for $c$ and $E(\text{risk})$ make sense using sample data. 

### Optimal threshold

The method `expectedRisks` calculates risks at every threshold. We want the minimum risk to match the risk at $c$ on a given sample.


```scala
case classs Tradeoff(...){
    //...

    def expectedRisks(pa: AppParameters): Row = {
        val risk: Tuple2[Double,Double] => Double = paramToRisk(pa)
        this.asPmissPfa.transpose.map{case Vector(pMiss,pFa) => risk((pMiss,pFa))}
    }    
}
```

{% include demo14-bayesdecisions1-20.html %}

The bottom pane plots `expectedRisks` and confirms that $c$ gives the minimum.

### Expected risk

Now check that the estimate for the minimum $E(\text{risk})$ is reliable. That is, if use the corresponding cutoff "in the wild", do we get the expected risk? Data simulation allows us to answer the question empircally. The four main steps are 
- Get a clasifier that applies the $c$ cut-off
- Generate a few hundred datasets
- Compute the risk
- Check that the sample estimate is similar to the simulated values

```scala
val cutOff: Double = hisTo.minS(pa) // hisTo is the Tradeoff instance
val thresholder: (Double => User) = score => if (score > cutOff) {Fraudster} else {Regular}
def classifier:(Array[Double] => User) = recognizer andThen logit andThen thresholder

// Simulate the risk of one transaction
def cost(p: AppParameters, actual: User, pred: User): Double = pred match {
        case Fraudster if actual == Regular => p.Cfa
        case Regular if actual == Fraudster => p.Cmiss
        case _ => 0.0
}

def simulateTransact: Distribution[Double] = for {
    transaction <- transact(pa.p_w1)
    prediction = classifier(transaction.features.toArray)
    risk = cost(pa, transaction.UserType, prediction)
} yield risk

val nrows = 1000
val nsimulations = 500

// Simulate average risk for a dataset of 1,000 rows 
val simData: Distribution[Double] = simulateTransact.repeat(nrows).map(_.sum.toDouble / nrows)

// Repeat 500 times
val simRisk: Row = simData.sample(nsimulations).toVector
```

Note that `classifier` applies a `logit` transform to the `recognizer`'s output. That is because SVM scores are originally in $[0,1]$, and I want them in $\mathbb{R}$ to emphasize that scores need not be "probability-like" values - they could also be projected onto $[0,inf]$ for example. The logit is a monotonously increasing function, so it does not affect the score index returned by the `minS` method.

{% include demo13-simulation-14.html %}

The evaluation sample risk is the vertical dotted bar near the peak of the distribution, which is good news. The grey shaded area comprises 95% of the simulated data, and I want the sample estimate to fall in that interval, which it does.

In the next part of this blog article, we will explore the connection between Bayes decision rules and the Receiving Operator Characteristics (ROC) curve.

## References
- R. Duda et al (2001), Pattern Classification.
- J. Hanley and B. McNeil (1982), The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve.

## NIST SRE 
This is my personal reading list and is not a comprehensive index of the NIST-related research.

- N. Brümmer et al (2021), Out of a Hundred Trials, How Many Errors does your Speaker Verifier Make?
- A. Nautsch (2019), Speaker Recognition in Unconstrained Environments
- N. Brümmer et al (2013), Likelihood-ratio Calibration Using Prior-Weighted Proper Scoring Rules
- N. Brümmer and E. de Villiers (2011), The BOSARIS Toolkit: Theory, Algorithm and Code for Surviving the New DCF
- N. Brümmer (2010), Measuring, Refining and Calibrating Speaker and Language Information Extracted from Speech
- D. A. van Leeuwen and N. Brümmer (2007), An Introduction to Application-Independent Evaluation of Speaker Recognition Systems
- N. Brümmer and J. du Preez (2006), Application-Independent Evaluation of Speaker Detection

D. van Leeuwen and N. Brümmer (2007) is my go-to source as it is a simple and practical introduction. A. Nautsch (2019) provides a clear, recent and extensive review of the the speaker recognition research. N. Brümmer (2010) also covers a lot of ground into lots of details, with information theoretic interpretations that I found useful.