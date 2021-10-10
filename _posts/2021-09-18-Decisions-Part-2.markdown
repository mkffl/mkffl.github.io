---
title: My Machine Learnt... Now What? - Part 2
layout: post
---

## Part 2: Receiving Operator Characteristics

### Section 3 - The ROC Assessment framework
- [Overview] Emphasizes the tradeoff between the two error types by plotting Pfa against (1-Pmiss)

The ROC curve slides through every possible cutoff point in descending order and plots the corresponding (1-Pmiss,Pfa) values at that cutoff. As a reminder, Pmiss is the proportion of targets below the cutoff point, which corresponds to the proportion of targets labeled as false negatives. Thus, $1-\text{Pmiss}$ is called true positive rate (tpr). Pfa is the propotion of non-targets labeled as false positives, also called false positive rate (fpr).

As seen previously, Pmiss and Pfa are the two inputs into $E(r)$ that vary with thresholds, and we would need to add the other four non-varying inputs to provide information about expected risks. That would add another perspective on top of what the CCD and LLR plots provide.

This information can be added via the so-called isocost curve. (Note on E(r) plot which is tied to application parameters). Isocost curves are used extensively in other fields, e.g. in economics to find combinations of inputs that yield the same output. Here, we will find combinations of (tpr,fpr) that yield the same risk.

$$
\begin{equation}
\begin{aligned}

& p(\omega_1)*\text{Pmiss}*\text{Cmiss}+p(\omega_1)*\text{Pfa}*\text{Cfa}=\text{e} \\
& \text{Pmiss} = -\text{Pfa}*\frac{p(\omega_0*Cfa)}{p(\omega_1)*\text{Cmiss}}+\frac{\text{e}}{p(\omega_1)*\text{Cmiss}} \\
& 1-\text{Pmiss} = \text{Pfa}*e^{-\theta}+(1-\frac{\text{e}}{p(\omega_1)*\text{Cmiss}}) \\
& \text{tpr} = \text{fpr}*\text{a}+\text{b}

\end{aligned}
\end{equation}
$$

a and b are determined by the application parameters, while (tpr,fpr) is achieved by the recognizer and is also called an operating point. For given application parameters, an isocost curve with lower fpr and/or higher tpr corresponds to a lower risk, hence the optimal isocost is the most "north west" in the ROC space. That allows the analyst to visually compare two recognizers, and identify the best performer. [Can do comment on combining classifiers].

The ROC curve also includes the Bayes optimal criterion information available in the CCD andd LLR plots via the curve's derivative, which is equal to the likelihood ratio. That's because (1-Pmiss) and Pfa are linear functions of the cumulative density function (cdf) of $\omega_1$ and $\omega_0$, respectively, so the derivative of the ratio is the ratio of their pdf's. That sounds rather abstract, but we can also get the proof from the data.

`lr` gets the likelihood ratio directly from the `Tradeoff` object's counts and `slope` takes the slope of (tpr,fpr). Note that the "function" is actually a line that goes through a set of points, so the slope is the segment that connects a point to the next one. The slope at (1-Pmiss(t),Pfa(t)) is $\frac{ \text{Pmiss(t)}-\text{Pmiss(t-1)} }{ \text{Pfa(t-1)}-\text{Pfa(t)} }$.

```scala
/* lr(w1) = 
        p(s|w1)/p(s|w0)
*/
def lr(to: Tradeoff) = pdf(to.w1Counts).zip(pdf(to.w0Counts)).map(tup => tup._1/tup._2)

/* slope = 
        pmiss(t)-pmiss(t-1) / pfa(t-1)-pfa(t)
*/
def slope(to: Tradeoff) = {
    val pMissD = to.asPmissPfa(0).sliding(2).map { case Seq(x, y, _*) => y - x }.toVector
    val pFaD = to.asPmissPfa(1).sliding(2).map { case Seq(x, y, _*) => x - y }.toVector
    pMissD.zip(pFaD).map(tup => tup._1/tup._2)
}
```

`lr` and `slope`return the same results as expected.

```bash
@ lr(hisTo)
res1: Vector[Double] = Vector(0.0, 0.0, 0.0, 0.0, 0.007871325628334975, 0.0, 0.0, 0.01250151717441437, 0.07484429961857492, 0.19986233803966602, 0.6509123275478416, 1.9127321276853988, 5.6291370950741335, 20.78029965880433, 55.46923170287657, 82.21206052514464, 360.01869158878503, Infinity, Infinity, Infinity, Infinity, Infinity, NaN, Infinity, NaN)
@ slope(hisTo)
res2: Vector[Double] = Vector(0.0, 0.0, 0.0, 0.0, 0.007871325628334973, 0.0, 0.0, 0.012501517174414365, 0.07484429961857492, 0.19986233803966608, 0.6509123275478419, 1.912732127685398, 5.629137095074125, 20.780299658804417, 55.4692317028767, 82.21206052514438, 360.01869158876053, Infinity, Infinity, Infinity, Infinity, Infinity, NaN, Infinity, NaN)
```

The connection to the Bayes decision threshold becomes apparent by going back to the isocost equation. The derivative of the "north west-most" line is $e^{-\theta}$ and is also a tangeant to the ROC curve - if not, we could shift the isocost and get a lower risk. That means that the optimal threshold is at $\text{LR}=e^{-\theta}$ i.e. $\text{LLR}=-\theta$.


{% include demo15-bayesdecisions2-10.html %}

The ROC curve emphasizes that the optimal decision point is determined by a tradeoff between missing errors and false alarm errors. We can get a lower miss rate (Pmiss i.e. 1-tpr) only if we accept a higher false alarm rate (Pfa).

How much Pfa we tolerate is determined by application parameters, captured by the derivative of the isocost line. Given flat priors and high Cmiss, our application requires low Pmiss and thus tolerates high Pfa. Visually, that corresponds to operating points in the upper-hand corner of the ROC space.

Finally, the LLR and ROC frameworks look consistent from an application parameter perspective. On the fraud application, the flat isocost correponds to a low threshold in the LLR graph. A steep isocost would map to a low LLR threshold, as a high $\text{Cfa}*p(\omega_0)$ would require a low false error rate. Sometimes, the priors and costs balance each other and $-\theta$ is close to 0, which means that the isocost slope is parallel to the $tpr=fpr$ line. An example is an application that minimises the error rate when prior probabilities are assumed roughly equal.


## Part 2

### Section 1 - Implementation considerations The PAV algorithm 
- [Overview] It is a better binning strategy than the current equal-sized bin histogram approach

So far the `Tradeoff` class evaluates CCD, LLR and ROC from the histogram counts. Though histograms make sense because evaluations rest on estimated probability functions (pdf, cdf, 1-cdf), software libraries do not use histograms. Popular implementations like R's [ROCR](http://ipa-tys.github.io/ROCR/) or python's [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) construct the ROC curve by counting fpr and tpr for every score threshold. This results in thinner intervals and more evaluation points.

One issue with histograms is to define the bin width. If the bins are too wide, the optimal cutoff may be lost inside a bin, resulting in a higher expected risk. Would it make sense to select small intervals? If I set very thin intervals, the results are on the plot below. The LLR curve goes up and down and the ROC curve now has a steppy pattern.

The binary criterion does not seem compatible with the LLR not being a monotonous function of scores. If LLR(s) intersect $-\theta$ at $s=c$, however, $\text{LLR}(s) < -\theta$ for some scores above $c$, then the Bayes decision rule says that we should choose $\omega_0$ for these scores, though the binary classifer will predict $\omega_1$.

We can enforce monotonicity between scores and likelihood ratios by fitting monotonic functions like a simple linear regression: $\text{LLR}(s) = \beta_0+\beta_1*s$. That would be the same as transforming the scores into $[0,1]$ and fitting a logistic regression on the observed probabilities, a popular method for calibration. There are also non-parametric options to enforce monotonicity and, in fact, one of them called the Pair Adjacent Violators (PAV) is commonly used because it ensures that the resulting ROC curve is [convex](https://en.wikipedia.org/wiki/Convex_set).

The PAV creates groups of scores with representative thresholds, which I can use as input into the `Tradeoff` object. The resulting ROC curve is convex, while the thin-sized bin histogram ROC appears non convex. From a decision perspective, that means that the histogram includes many sub-optimal operating points, i.e. with higher expected risks than what could be achieved by using points on the convex hull. For more information and compelling examples, I would refer to Fawcett and Niculescu-Mizil article "PAV and the ROC Convex Hull".

[Talk about packages not using theta].

Demo 16

{% include demo16-histVSpav-10.html %}

### Section 2 - Risk VS AUC use case
- [Overview] Risk-based model selection is better than AUC criteria (use case)

ROC curves are also popular for providing a visual representation of the concordance/discrimination metric. The Area Under the Curve (AUC) of the ROC is equal to $p(s_{\omega_0} < s_{\omega_0})$ - see this SO question for a proof. TODO. While high discrimination power is good, I wonder what to do with the reported AUC metrics. In particular, should we aim for a minimum AUC before deploying a system? The next example shows that higher concordance is not necessarily better from a risk optimisation viewpoint. Therefore, if one assesses a system through its expected risk, concordance metrics may provide no more than a good sense check - e.g. is it close or well above 0.5?

This use case is based on an example from another blog post, [ML Meets Economics](http://nicolas.kruchten.com/content/2016/01/ml-meets-economics/), which is a great practical introduction to AUC and ROC curves. I only make small adjustments to their numbers to fit my simulated data and I try to dig further into the disagreement between $E(\text{risk})$ and AUC.

The backstory is that a factory makes widgets that may overheat 5% of the time due to faulty gearboxes. Bad gearboxes costs the company £157 owing to wasted labour and inputs. The gearbox supplier, which sells them at £50 a piece, wouldn't improve QC, so the factory decides to use a ML system to detect gearboxes that will overheat. Any item flagged as faulty must be tested, which destroys it. Last, every working widget is sold for a net profit of £40.

If $\omega_1$ ($\omega_0$) represents the defect (working) gearbox and $\alpha_1$ ($\alpha_0$) the decision to test (not test) a gearbox, then the cost/profit structure can be represented as


<div class="mermaid"> 
    graph LR;
    id0([_]) --> id11[class ω0];
    id0([_]) --> id12[class ω1];
    id11[class ω0]-->id21[α0]; 
    id11[class ω0]-->id22[α1]; 
    id12[class ω1]-->id31[α0]; 
    id12[class ω1]-->id32[α1];
    id21[action α0] --> id41[cost c_00 = -40.0];
    id22[action α1] --> id42[cost c_10 = 50.0];
    id31[action α0] --> id43[cost c_01 = 157.0];
    id32[action α1] --> id44[cost c_11 = 50.0];
</div>

Note the negative cost of not testing the item when it's not faulty, which represents the profit of a sale. In the original blog article, the author maximises utility, while we minimise the expected risk, but the two approaches are equivalent (as long as we use signs consistently).

We can use the simplified cost structure implemented in `AppParameters` by identifying the extra terms and offsetting them. This is actually a useful exercise to realise that the Bayes decision threshold $-\theta$ is determined by the ratio between 
$$
p(\omega_0)*u
$$ 
and
$$
p(\omega_1)*v
$$
. We decided that $u$ and $v$ should be Cmiss and Cfa, resp., but we could break them down differently.

The Bayes decision criterion is to choose $\alpha_1$ if $\text{lr}(\omega_1)$ is greater than $\frac{(\text{C10-C00})p(\omega_0)}{(\text{C01-C11})p(\omega_1)}$. Using the values above is equivalent to defining `AppParameters(0.05,107,90)`.

There are two recognizers and, in what follows, objects related to the high and low AUC recognizers are prefixed with "hi" and "low" (not "lo" to avoid any confusions with log-odds), respectively. The data generation process directly defines the recogniser scores, instead of generating features and fitting a recognizer, because it's less work for the same result. We can check that the AUC of `hiAUCdata` is indeed higher: 

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
```

```bash
@ println(p)
res3: Double = 0.98
@ assert(p > (1-α))
res4: Boolean = true
```

- The `hiAUCdata`recogniser is better from an AUC viewpoint, however, it has a worse expected risk: 
```scala
    val aucPa = AppParameters(0.05,107,90)

    // Define w1HiCnts etc. using PAV - see source code
    val hiTo = Tradeoff(w1HiCnts,w0HiCnts,hiThresh)
    val lowTo = Tradeoff(w1LowCnts,w0LowCnts,lowThresh)
```

```bash
@ println(hiTo.minRisk(aucPa))
res3: Double = 2.44
@ println(lowTo.minRisk(aucPa))
res4: Double = 1.44
```

- The LLR and ROC plots help to make sense of these seemingly strange results. The application parameters correponds to a high value of llr threholds, which means that the optimal score cut-off is very high. Thus, it makes sense that the ROC isocost is steep and located towards the left of the curve, as ROC scores are in decreasing order. 
- This is true for both recognisers, though low AUC can achieve high llr on more target instances than high-AUC, and therefore is less penalised with false negatives. In other words, the high-AUC recogniser is very close to an all-w_0 classifer to avoid the high costs of false alarm, which it trades off for false negative costs. 
- The LLR curve tells the same story - the high-AUC reconiser achieves better separation across all instances as can be seen on its steeper LLR curve. The lower-AUC model is not as good in terms of separation, as its LLR curve is flatter around 0. However, this recogniser achieves better separation on high score instances, which improves risk on high Cfa applications.

{% include Demo17-llrRoc4panes-1.html %}

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

