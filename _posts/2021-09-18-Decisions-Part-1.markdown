---
title: My Machine Learnt... Now What? - Part 1
layout: post
---

## Part 1: Bayes Decision Criteria

### Introduction
- Motivations
- BOSARIS background and notation (borrow a number of concepts: target/non, recognisers; liked to have clearly defined concepts while ML is too often a hodge podge of ill-defined hyped terms)
- Scripting with data simulations
   - Why Scala?
   - probability_monad

### Section 1 - Concordance Metrics
- [Overview] Recognisers ability to separate $\omega_1$ from $\omega_0$ instances can be summarised in one metric, called A, that is $p(s_{\omega_0} < s_{\omega_0})$; efficient computation is possible via the rank-sum algorithm 

Given a recognizer that outputs scores, we may ask if the tool can differentiate between ${\omega_0}$ and ${\omega_1}$ instances. We can first look at the distribution of scores by class, $p(s \vert w_i)$, and visually inspect if ${\omega_1}$ instances have higher scores. Good separation means that the two distributions barely overlap, and perfect separation means no overlap at all.

For example, fitting a Support Vector Machine (SVM) with default parameters on the transaction data gives the following separation. The histograms estimate the score probability conditioned on the class so I call it a Class Conditional Distributions (CCD) chart.

{% include demo11-ccd-5.html %}

Scores below -2 are almost certainly associated with non-targets while scores between -1  and 1 are more uncertain. We can measure the recognizer's discrimination power in one metric that estimates the probability that $\omega_0$ have lower scores than $\omega_1$ instance i.e. $p(s_{\omega_0} < s_{\omega_0})$.

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

I find this approach very cool because it's not only clever but also intuitive. In the combined, sorted dataset of scores, we can count the concordance value for a single target instance by taking its overall rank and subtracting its rank in the subsample of target instances. If we repeat this procedure for all target instances and rearrange the operations, we get the value computed by `U`. You can play around with a tiny example of 2 non-target and 2 target instances to get convinced.

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



### Section 2 - Bayes decisions rules
- Correspond to optimal decisions in terms of risk-minimisation; For recognizers, that corresponds to cut-off points that minimize risk given an application context

#### Univariate use case
Suppose that we need to label instances using only one predictor. A randomly drawn dataset has one feature column $x$ and one target field $y$ with values in $\{\omega_0$, $\omega_1\}$. We want to decide on a classification rule that assigns new instance labels based on their $x$ values. 

Further assume equal prior probabilities i.e. $p(\omega_0) = p(\omega_1) = 0.5$ and that the rule should minimise the number of misclassified instances. The class-conditional distributions $p(x \vert \omega_i)$ below suggests a rule.

{% include demo12-ccd-3.html %}

We should choose $\omega_0$ when its likelihood is greater, and $\omega_1$ otherwise. In this sample data, $p(x \vert \omega_1) > p(x \vert \omega_0)$ when $x>-0.25$, so any new instance with $x > -0.25$ should be classified as a target. Doing this consistently with every new instances guarantees that we'll minimise the error rate.

Any other rule corresponds to a higher probability of error. For example, if a new instance has $x=1.0$ then assigning $\omega_0$ would not make sense because the chances of being wrong are more than 4 times higher than being correct.

This line of reasoning is at the core of Bayes decisions criteria, which tells us that picking the option that minimises the risk for every single instance will minimise the overall risk. Common approaches to selecting classifier thresholds implicitly rely on Bayes decision criteria and the link will become obvious soon if it's not already clear.

The following is based on the first chapter of [Pattern Classification](https://www.amazon.fr/Pattern-Classification-2e-RO-Duda/dp/0471056693) by Hart and others. It's an old but surprisingly refreshing textbook that covers Bayes decision theory in depth. I simplify their notation to the minimum necessary for a binary classification problem, though they address more general problems.

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

The first node is corresponds to choosing action $\alpha_0$ when the true state is $\omega_0$, i.e. choosing non-target when the instance is actually a non-target, which is often assumed to cost 0.0. The next node corresponds to a false alarm, with cost Cfa. The next is when we miss a target, which has a cost of Cmiss, and the last node correspodns to a true positive, which does not cost anything.

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

#### More than one feature
In general, there are more than one features and class-conditional sample density functions are more complex than above, possibly leading to many decision regions. Fortunately, the Bayes decision procedure applies to a recognizer's scores. In fact, we could think of a recognizer as a map from a large feature space to a one-dimensional space, $f: \mathbb{R}^d \mapsto \mathbb{R}$. 

Furthermore, the map returns scores such that higher values correspond to a more likely $\omega_1$ state. In this article (AUC/WMC paper), the authors call scores "degrees of suspicion", which I thought captures well the idea of ordered values and their relationship to hypothesis testing. I think of a score as the noise that a patient would make when a doctor gently presses parts of a sore spot to locate a sprained tendon. A stronger shriek means that the doctor is getting closer to the torn ligament, however, its intensity doesn't tell us how far the instrument is from the damaged tendon.

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


[minS on Tradeoff object]

Going back  to the SVM scores used in the CCD plot at the start, we would like to find the Bayes decision criterion $c$ to convert the recognizer into a classifier. We will need a few bits in place: 

- The histogram counts of scores - the same as the CCD chart - which consists of the 3 vectors below. The histogram implementation is not particularly interesting so I don't show it, but it's available in the source code.
```scala
val w0Counts: Row = ... // counts of non-target labels
val w1Counts: Row = ... // counts of target labels
val thresh: Row = ... // bins
```

- Common operations applied to sample data density, most of which will come in handy in later sections.
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

- An object to encapsulate the sample data and related methods. 

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

Next, we implement `minS`, a method to find the Bayes decision cut-off point, i.e. the score value that minimises the expected risk given some application parameters.

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

The `asLLR` value returns the log-likelihood ratio of the scores, which is the left-hand side of the equation (TODO: ref.). The `minusθ` methods convert the application parameters into $-\theta$, which is the right-hand side of the equations. Then, `argminRisk` uses these two inputs to find the array index of the closest match, which is used by `minS` to provide the cutoff point $c$.

##### Average risk

We can estimate the expected risk of a Bayes decision classifer using the evaluation data:

$$
E(risk) = Cmiss*p(\omega_1)*Pmiss + Cfa*p(\omega_0)*Pfa
$$

where Pmiss is the proportion of targets with scores below the Bayes decision cutoff, $c$, and Pfa is the proportion of non-targets with scores above $c$.

The Pmiss and Pfa notation comes from the BOSARIS documentation, however, the formula is used in many assessment frameworks and is quite intuitive. That is perhaps why its derivation is often not documented, so I include it below.

The expected risk is the risk of every action chosen for all instances in the data. But the action chosen depends on the instance region, $R_0$ and $R_1$:

$$
\begin{equation}
\begin{aligned}
E(risk) 
& = \int_{-infinity}^{+infinity} risk(\alpha(x) | x)*p(x) dx \\
& = \int_{x \in R_0} risk(\alpha_0 | x)*p(x) dx + \int_{x \in R_1} risk(\alpha_1 | x)*p(x) dx
\end{aligned}
\end{equation}
$$

With binary scores, the region $R_i$ is determined by the Bayes cutoff $c$, and every instance $x$ is mapped to a score $s$, so instead of sliding through $x$ we slide through all $s$'s:

$$
\begin{equation}
\begin{aligned}
E(risk)
& = \int_{s < c} risk(\alpha_0 | s)*p(s) ds + \int_{s > c} risk(\alpha_1 | s)*p(s) ds \\
& = Cmiss*\int_{s < c} p(\omega_1 | s)*p(s) ds + Cfa*\int_{s > c} p(\omega_0 | s)*p(s) ds \\
& = Cmiss*p(\omega_1)\int_{s < c} p(s | \omega_1) ds + Cfa*p(\omega_0)*\int_{s > c} p(s | \omega_0) ds
\end{aligned}
\end{equation}
$$

The first and second integrals are estimated with Pmiss and Pfa, respectively. If $N_{\omega_1}$ is the number of targets, then
$$
Pmiss = \sum_{s \in \omega_1}\frac{[s < c]}{N_{\omega_1}}
$$ 
and 
$$
Pfa = \sum_{s \in \omega_0}\frac{[s >= c]}{N_{\omega_0}}
$$


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

Let's wrap up by checking that the calculations for $c$ and $E(r)$ make sense with some data. 

The method `expectedRisks` calculates risks at every threshold. We want the minimum risk to match the risk at $c$.


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

To check $E(r)$, let's 
- Get a clasifier that applies the $c$ cut-off
- Generate a few hundred datasets
- Compute the risk
- Check that the sample estimate is similar to the simulated values

```scala
val cutOff: Double = hisTo.minS(pa) // hisTo is the Tradeoff instance
val thresholder: (Double => User) = score => if (score > cutOff) {Fraudster} else {Regular}
def classifier:(Array[Double] => User) = recognizer andThen logit andThen thresholder

// Simulate one transaction's risk
def cost(p: AppParameters, actual: User, pred: User): Double

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

{% include demo13-simulation-14.html %}

The evaluation sample risk is the vertical dotted bar near the peak of the distribution, which is good news. The grey shaded area comprises 95% of the simulated data, and I want the sample estimate to fall in that interval, which it does.

In the next part of this blog article, we will explore the connection between Bayes decision rules and the Receiving Operator Characteristics (ROC) curve.
# Part 2: Receiving Operator Characteristics

### Section 3 - The ROC Assessment framework
- [Overview] Emphasizes the tradeoff between the two error types by plotting Pfa against (1-Pmiss)

- s is the classifier output

- While the CCD chart shows cut-off points and likelihood points, ROC curves show 
```
∫p(x>c|ω1)dx = 1-Pmiss called True Positive Rate (tpr)
and ∫p(x>c|ω0)dx = Pfa called False Positive Rate (fpr)
```

The CCD plot information is implicitly shown in ROC space because the gradient of each point is the likelihood ratio of a threshold. With the CCD plot, we swiped through every threshold from right to left and reported the corresponding LLR to find the minTheta match. With the ROC plot, we also swipe through from right to left but we report the corresponding right-hand side area under the curve, i.e. fpr and tpr, whose derivative is the LR (??)

Thus, given application parameter minTheta, the risk-minimising (fpr,tpr) point has a gradient of exp(minTheta). Visually, we can imagine sliding a segment of gradient exp(minTheta) in the plane and retaining all the (fpr,tpr) pairs tangeant to the segment. In this example, there seems to be one such point because the curve seems concav-sih (more on this later), but in general there could be multiple solutions. The curve closest to the left-hand top corner corresponds to the lowest expect risk as it has lower (Pmiss,Pfa) all else constant.

Demo 15 (CCD and ROC with isocosts)

## Part 2

### Section 1 - The PAV algorithm 
- [Overview] It is a better binning strategy than the current equal-sized bin histogram approach
- The CCD plot implicitly provides the likelihood ratio but we can plot them directly in a graph of scores vs LLR. Adding a horizontal line for min Theta gives both sides of the Bayes rule equation, so we can find the cut-off point for a Bayes optimal classifer. However, it appears that the LLR curve goes up and down i.e. it's not monotonous. From a Bayes rule perspective, this does not seem compatible with the binary criterion. If LLR(s) intersect minTheta at s=c but LLR(s) < minTheta for some scores above c, then we should choose w_0 for these scores, though our classifer will predict w_1.
- This issue is also evident with the ROC curve as it is not convex. That means that the curve can be flat or vertical, in which was one of the two operating points is not optimal because it has the same pmiss but a higher pfa (horizontal) or the same pfa but a higher pmiss (vertical). 
- To enforce monotonicity between scores and likelihood ratios, we can fit monotonic functions like a logit (llr = sigmoid(a+b*s)) or other parameteric models. Non-parametric solutions are also possible and, in fact, one of them called the Pair Adjacent Violators (PAV) has become popular because the resulting ROC curve is convex, which means it includes only optimal (fpr,tpr) points. 
- To summarise, the PAV algorithm creates varying-size score bins that provide two guarantees, a) score-likelihood monotonicity (higher scores have equal or higher likelihood) and b) ROC convex hull (optimal decisions)
- The chart below compares the histogram approach with PAV on the score-LLR and ROC curves. The histogram ROC curve is below or on the PAV curve. I think that most of the PAV points are "lost" inside the histogram bins [Try histogram with thinner bins to check if there's more overlap.]

Demo 16

- In practice, histograms are not used...

### Section 2 - Risk VS AUC use case
- [Overview] Risk-based model selection is better than AUC criteria (use case)
- The first Part of this series introduced ther rank-sum for efficient AUC computation, however, we did not conclude on its applications. When should we use AUC? Is it a good criterion for model selection? How does it compare with risk-based assessments? The following sections shows the limit of AUC as a criterion for model selection through a simulated data example. The conclusion is that a low-AUC recognizer may be preferrable to a high-AUC model because it results in a better expected risk. 
- What follows based on example I found in another blog post, [ML Meets Economics](http://nicolas.kruchten.com/content/2016/01/ml-meets-economics/), which is a great practical introduction to AUC and ROC curves. Footnote: they use opposite labels, where defect products are target/positive classes, hence their ROC isocosts are different than here, but the conclusions are the same.
- To get started, we need some data and two recognizers, a low and high-AUC. In what follows, objects related to the high and low AUC recognizers are prefixed with "hi" and "low" (not "lo" to avoid any confusions with log-odds), respectively. The data generation process directly defines the recogniser scores, instead of generating features and fitting a recognizer. We can check that the AUC of `hiAUCdata` is indeed higher: 

```scala
    def hiAUCdata: Distribution[List[Score]] = HighAUC.normalLLR.repeat(1000)
    def lowAUCdata: Distribution[List[Score]] = LowAUC.normalLLR.repeat(1000)

    def score2Auc(data: List[Score]): Double = {
        val (nonS,tarS) = splitScores(data)
        smartA(nonS,tarS)            
    }

    def simAUC: Distribution[Double] = for {
        h <- hiAUCdata
        l <- lowAUCdata
        hAuc = score2Auc(h)
        lAuc = score2Auc(l)
        diff = (hAuc - lAuc)
    } yield diff

    /* Check that
        P(hiAuc > lowAuc) 
    */
    val p = simAUC.pr(_ > 0)
    val α = 0.05
    println(p) // 0.98
    assert(p > (1-α)) // true
```

- The `hiAUCdata`recogniser is better from an AUC viewpoint, however, it has a worse expected risk for applications that strongly penalise false alarms: 
```scala

    // Define w1HiCnts etc. - see source

    val hiTo = Tradeoff(w1HiCnts,w0HiCnts,hiThresh)
    val lowTo = Tradeoff(w1LowCnts,w0LowCnts,lowThresh)

    // Define an application with Cfa 16x bigger than Cmiss
    val aucPa = AppParameters(0.5,5,80)

    println(hiTo.minRisk(aucPa)) // 2.44
    println(lowTo.minRisk(aucPa)) // 1.44
```

- The LLR and ROC plots help to make sense of these seemingly strange results. The application parameters correponds to a high value of llr threholds, which means that the optimal score cut-off is very high. Thus, it makes sense that the ROC isocost is steep and located towards the left of the curve, as ROC scores are in decreasing order. 
- This is true for both recognisers, though low AUC can achieve high llr on more target instances than high-AUC, and therefore is less penalised with false negatives. In other words, the high-AUC recogniser is very close to an all-w_0 classifer to avoid the high costs of false alarm, which it trades off for false negative costs. 
- The LLR curve tells the same story - the high-AUC reconiser achieves better separation across all instances as can be seen on its steeper LLR curve. The lower-AUC model is not as good in terms of separation, as its LLR curve is flatter around 0. However, this recogniser achieves better separation on high score instances, which improves risk on high Cfa applications.


### Section 3 - Majority-isocost lines

- The recognisers' risks are not miles away from an all-w_0 rule, which is the benchmark we should assess any model against. In many cases, categorising all or no instances as w_1 is the status quo. Would the status quo have a lower risk - and therefore be peferrable to any recognition solution - if the cost of false alarms was higher? This is a legitimate question if we are not certain about outcome costs but still want to assess the benefits of an automated solution.
- We can do a sensitivity analysis by comparing the best recogniser's risk against different application parameter scenarios, but the ROC framework provides a neat way to visualise the scenarios. 
- If we know the expected risk of the relevant majority rule (all-w_0 for a high Cfa application) then we can plot a straight line in the ROC space that achieves the same risk, which I will call a "majority-isocost line" (for lack of better name). Each point on the line is a (fpr,tpr) pair with the same expected risk, and deploying a recogniser only makes economic sense if it lies above the majority-isocost. Any such point will have an isocost closer to (fpr=0,tpr=1) i.e. a lower E(r).

Demo 18

## Conclusion and opening to Part 3 

## Part 3

- Why do we sometimes need a calibrated recognizer?
- How should we assess a calibrated recognizer?
    -> LLR is a calibration plot; relation with reliability diagram
    -> Can we comapre differnt systems against different application parameters?

