---
title: Decisions Under Uncertainty - Part 1
layout: post
---

## Part 1

#### Introduction
- Motivations
- BOSARIS background and notation (borrow a number of concepts: target/non, recognisers; liked to have clearly defined concepts while ML is too often a hodge podge of ill-defined hyped terms)
- Scripting with data simulations
   - Why Scala?
   - probability_monad

#### Section 1 - Concordance Metrics
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



#### Section 2 - Bayes decisions rules
- Correspond to optimal decisions in terms of risk-minimisation; For recognizers, that corresponds to cut-off points that minimize risk given an application context

Imagine a classifiation task using a univariate feature set. A randomly drawn dataset would have two columns, one for the feature, $x$, and one for the variable to predict, $y$, which is either $\omega_0$  or $\omega_0$. We need to decide on a rule, i.e. a way to partition $x$ to assign a label to new instances.

Assume equal prior probabilities i.e. $p(\omega_0) = p(\omega_1) = 0.5$ and that the rule should minimise the number of misclassified instances. The class-conditional histograms of the features, $p(x \vert \omega_i)$, is shown below. What label should we assign to a new instance with $x=0.6$? [answer]

''' Demo 12b'''

To minimise the error rate, we should follow this rule

$$
\text{For any new instance x, choose} 
\begin{cases}
    \omega_1 \text{if } p(x \vert \omega_0) < p(x \vert \omega_1) \\
    \omega_0, & \text{otherwise}
\end{cases}
$$




<div class="mermaid"> 
    graph LR;    
    id11[class ω0]-->id21[α0]; 
    id11[class ω0]-->id22[α1]; 
    id12[class ω1]-->id31[α0]; 
    id12[class ω1]-->id32[α1];
    id21[action α0] --> id41[cost C_00 = 0.0];
    id22[action α1] --> id42[cost C_10 = Cfa];
    id31[action α0] --> id43[cost C_01 = Cmiss];
    id32[action α1] --> id44[cost C_11 = 0.0];
</div>

That's because with equal priors, the likelihood $p(x \vert \omega_i)$ determines whichever of $p(\omega_0)$ or $p(\omega_1)$ is more likely, and we should choose the option with the lower risk of being wrong. If $x=0.6$ then the least risky choice is $\omega_1$.

On the example chart, $\omega_0$'s likelihood is above $\omega_1$ on the left of the vertical bar, so the feature is split into two regions, but in general there could be multiple such regions. What's really important is that any other rule would increase the risk of misclassification.

The same principle applies to more general contexts. If we expect more target instances than non-targets, i.e. $p(\omega_1) > p\omega_1)$, then the criterion becomes  


$$
\text{For any new instance x, choose} 
\begin{cases}
    \omega_1 \text{if } p(x \vert \omega_1)/p(x \vert \omega_0) > p(\omega_0)/p(\omega_1) \\
    \omega_0, & \text{otherwise}
\end{cases}
$$

If the cost associated with missing targets is higher than that of missing non-targets, also called a false alarm, then the criterion becomes "Choose \omega_1" if $p(\omega_1)P(s \vert \omega_0) * Cfa < p(\omega_1)p(s \vert \omega_1) * Cmiss$ i.e. $lr(\omega_1) > p(\omega_0) * Cfa / p(\omega_1) * Cmiss$

Changing priors or misclassification costs changes the decision range. With higher target prevealence (higher p_w1) and/or higher Cmiss, the range for w_1 becomes larger, which means moving the decision cut-off to the left on the CCD plot above. If the cost of misclassifying a w_1 instance is bigger, we should lean more towards w_1. 

The Bayes rule applies to single instances and leads to the smalles total risk possible. If we make the best decision every single time, we'll have made the best decision in aggreagate. 

Given priors and misclassification costs, the expected risk is 
```
E(r|ω1,Cmiss,Cfa) = Cmiss.p(ω1).∫p(x<c|ω1)dx + Cfa.p(ω0).∫p(x>c|ω0)dx = Cmiss.p(ω1).Pmiss + Cfa.p(ω0).Pfa
```
And choosing c with the Bayes rule gives the lowest expected risk. 

Demo 13 (simulation)
Demo 14 (CCD and E(r))

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

