---
title: My Machine Learnt... Now What? - Part 2
layout: post
---

The [previous part]({{ site.baseurl }}{% link _posts/2021-10-18-Decisions-Part-1.markdown %}) introduced the Bayes decision rule as a procedure to find the optimal threshold that transforms a recognizer into a classifier, with "optimal" referrefing to the minimum expected risk achieved on unseen data. The risk at every threshold $c$ was described as a function of $\text{Pmiss}$ and $\text{Pfa}$. 

The Receiving Operator Characteristics (ROC) combines these concepts and provides the same capabilities for threshold selection as the CCD and LLR frameworks introduced earlier. However, it has a focus on error trade-offs rather than likelihood ratios, which allows to run various investigations one one graph.

The first section looks at threshold optimisation using the ROC curve, and shows why doing so is equivalent to using the Bayes decision procedure. The following sections cover  topics on ROC analysis: the monotonicity of a ROC curve and how to enforce it (section 2), the connection between ROC and concordance probability via AUC, and why you may *not* want to optimise for the latter (section 3), and benchmarking a classifier with majority rules using ROC analysis (section 4).

## A. The ROC curve

### Finding an optimal threshold

The ROC curve slides through every possible cut-off point in descending order and plots the corresponding (Pfa, 1−Pmiss) values at that cut-off. Pmiss and Pfa are error rates [TODO: link to part 1] estimated as proportions of errors of type w1 and w0, respectively. (1-Pmiss) is often called the true positive rate (tpr), because Pmiss is estimated as the proportion of targets below the cut-off point, which is also the rate of false negatives and 1 - false negative rate means true positive rate. Similarly, Pfa is the proportion of non-targets labeled as false positives, also called false positive rate (fpr). And so, in the literature, (Pfa, 1-Pmiss) is referred to as (fpr, tpr), also called operating points. 

[source](https://github.com/mkffl/decisions/blob/edc8cf34d8e3d82e7fdd1cdb48914e1bd1bfbbd3/Decisions/src/Evaluations.scala#L183)
```scala
  case class Tradeoff(...) {
        // ...

    val asROC: Matrix = {
      val fpr = (rhsArea andThen decreasing)(w0Counts)
      val tpr = (rhsArea andThen decreasing)(w1Counts)
      Vector(fpr, tpr)
    }
  }
```

Combining error rates with the application-dependent priors and error costs yields expected risks. In the ROC space, the operating points that correspond to the same expected risk are plotted along the so-called isocost lines (iso=same and, here, cost=risk). So, the intersection between a ROC operating point and the isocost line tells us what risk is expected at that cut-off point. Doing this for every ROC operating point and “argmin-ing” returns the optimal threshold.

The bottom graph below plots the ROC curve with the isocost line - red-dotted line with slope 0.2 - for the fraud application of Part 1, `AppParameters(p_w1=0.5,Cmiss=25,Cfa=5)`. The isocost intersects the ROC curve at the optimal operating point. Every other parallel line would also be an isocost but it would intersect at a sub-optimal point because either (1-Pmiss) would be lower and/or Pfa would be higher. That is, the optimal operating point is where the isocost line is tangent to the ROC. 

{% include demo15-bayesdecisions2.html %}

### Connection with the Bayes decision procedure

We can draw a connection to the Bayes decision rule from the observation that "the optimal threshold is where the isocost line is tangent to the ROC". As a reminder, the rule states that we should should targets when defines the cost-weighted prior ratio, 
$$
\delta = \frac{p(\omega_0) \times \text{Cfa}}{p(\omega_1) \times \text{Cmiss}}
$$
is equals to the likelihood ratio. The key is that ROC operating points correspond to likelihood-ratios, and isocost lines have a gradient of $\delta$, so ROC optimization is equivalent to a Bayes decision. Let’s unpick this.

#### Isocost lines have a gradient equal to $\delta$

Starting with equation 1.2 [TODO: link], we get the isocost defined as a linear relationship between (1-Pmiss) and Pfa with $\delta$ as the derivative

$$
\begin{equation}
\begin{aligned}

& p(\omega_1)*\text{Pmiss}*\text{Cmiss}+p(\omega_1)*\text{Pfa}*\text{Cfa}=\text{e} \\
& \text{Pmiss} = -\text{Pfa}*\frac{p(\omega_0*Cfa)}{p(\omega_1)*\text{Cmiss}}+\frac{\text{e}}{p(\omega_1)*\text{Cmiss}} \\
& 1-\text{Pmiss} = \text{Pfa}*e^{-\theta}+(1-\frac{\text{e}}{p(\omega_1)*\text{Cmiss}})

\end{aligned}
\end{equation}
$$

And so,
$$
\text{tpr} = \text{fpr}*\delta+\text{a}
\tag{2.1}
$$

with $\delta = \frac{p(\omega_0) \times \text{Cfa}}{p(\omega_1) \times \text{Cmiss}}$ and $\text{a}=\frac{p(\omega_1) \times \text{Cmiss} - E(\text{risk})}{p(\omega_1) \times \text{Cmiss}}$.

And $\delta$ is the right-hand side in equation 1.1

#### ROC curve gradients are likelihood ratios

First, the derivative of the ratio $\frac{\text{1-Pmiss}}{\text{Pfa}}$ is the ratio of the derivatives: $\frac{\text{(1-Pmiss)’}}{\text{(Pfa)’}}$. Then, Pmiss (Pfa) is based on the cumulative function of omega_1 (omega_0), so the derivative of Pmiss (Pfa) is just the probability of targets (non-targets), and the ROC gradient is $\frac{p(\omega_1)}{p(\omega_0)}$

A numerical example can help hammer that point. The code snippet below calculates two values for every operating point: `lr` gets the likelihood ratio directly from the `Tradeoff` object's counts, while `slope` gets the lr from the slope of (tpr,fpr). Both values are the same, so the derivative of the ROC is the likelihood ratio.

Note that the ROC curve is not a continuous function but it's a line, so its slope is the segment that connects a point to the next one. The slope at $(1-\text{Pmiss(t)},\text{Pfa(t)})$ is $\frac{ \text{Pmiss(t)}-\text{Pmiss(t-1)} }{ \text{Pfa(t-1)}-\text{Pfa(t)} }$.

[source](https://github.com/mkffl/decisions/blob/edc8cf34d8e3d82e7fdd1cdb48914e1bd1bfbbd3/Decisions/src/Recipes.scala#L1163)
```scala
      /* lr(w1) =
                    p(s|w1)/p(s|w0)
       */
      def lr(to: Tradeoff) =
        pdf(to.w1Counts).zip(pdf(to.w0Counts)).map(tup => tup._1 / tup._2)

      /* slope =
                    pmiss(t)-pmiss(t-1) / pfa(t-1)-pfa(t)
       */
      def slope(to: Tradeoff) = {
        val pMissD = to
          .asPmissPfa(0)
          .sliding(2)
          .map { case Seq(x, y, _*) => y - x }
          .toVector
        val pFaD = to
          .asPmissPfa(1)
          .sliding(2)
          .map { case Seq(x, y, _*) => x - y }
          .toVector
        pMissD.zip(pFaD).map(tup => tup._1 / tup._2)
      }
```

And `lr` and `slope`return the same results as expected.

```bash
@ lr(hisTo)
res1: Vector[Double] = Vector(0.0, 0.0, 0.0, 0.0, 0.007871325628334975, 0.0, 0.0, 0.01250151717441437, 0.07484429961857492, 0.19986233803966602, 0.6509123275478416, 1.9127321276853988, 5.6291370950741335, 20.78029965880433, 55.46923170287657, 82.21206052514464, 360.01869158878503, Infinity, Infinity, Infinity, Infinity, Infinity, NaN, Infinity, NaN)
@ slope(hisTo)
res2: Vector[Double] = Vector(0.0, 0.0, 0.0, 0.0, 0.007871325628334973, 0.0, 0.0, 0.012501517174414365, 0.07484429961857492, 0.19986233803966608, 0.6509123275478419, 1.912732127685398, 5.629137095074125, 20.780299658804417, 55.4692317028767, 82.21206052514438, 360.01869158876053, Infinity, Infinity, Infinity, Infinity, Infinity, NaN, Infinity, NaN)
```

That gives the left-hand side of equation 1.1, and so the ROC solution is equivalent to the Bayes decision procedure.

#### Benefits of a ROC-based analysis

The ROC curve emphasizes the tradeoff between missing and false alarm errors in determining the optimal decision. As long as the curve is not flat, we get a lower miss rate $\text{Pmiss}$ if we accept a higher false alarm rate $\text{Pfa}$. How much $\text{Pfa}$ we tolerate is determined by the application type, captured by the derivative of the isocost line. Given flat priors and high Cmiss, our application requires low $\text{Pmiss}$ and thus tolerates high Pfa. Visually, that corresponds to operating points in the right upper-hand corner.

[TODO: rephrase] The graph also shows the connections between LLR and ROC. The fraud application penalizes false negatives heavily, which corresponds to a low $\delta$ in the definition above and therefore to a flattish ROC isocost, which maps to the low threshold in the LLR graph. 

A steep isocost would map to a high LLR threshold. If the ratio of priors and costs balance each other and $-\theta$ is close to 0, then the isocost slope is parallel to the $tpr=fpr$ line. An example of such application would be to minimise the error rate (Cmiss = Cfa = 1) when prior probabilities are assumed to be equal.
### Text bank

The ROC curve slides through every possible cut-off point in descending order and plots the corresponding $(1-\text{Pmiss}, \text{Pfa})$ values at that cut-off. As a reminder, $\text{Pmiss}$ is the proportion of targets below the cut-off point, which is also the rate of false negatives, thus, $1-\text{Pmiss}$ is called true positive rate (tpr). $\text{Pfa}$ is the proportion of non-targets labelled as false positives, also called false positive rate (fpr).



$\text{Pmiss}$ and $\text{Pfa}$ are the two inputs into $E(\text{r})$ that vary with thresholds. We would need to add the other four non-varying inputs to get expected risks, which would add another perspective on top of what the CCD and LLR plots already provide. 

As it turns out, the four application parameters, $\{p(\omega_1), p(\omega_0), \text{Cmiss}, \text{Cfa}\}$, can be visualised with an isocost curve. Combining (tpr,fpr) with isocosts provides a risk assessment given one or more recognizers, which is why ROC graphs are an indispensable part of the analyst's toolbox.

Isocosts are often used to find the best pair of values that correspond to a metric we wish to optimise, for example in economics with combinations of inputs that yield the same output. Here, we will look for combinations of (tpr,fpr) that yield the same risk.

Isocosts are expressed as a linear function between tpr and fpr:



$\delta$ and $\text{a}$ are determined by the application parameters and the risk objective, while (tpr,fpr) is achieved by the recognizer and is also called an operating point. 

Given some application parameters, an isocost curve with lower fpr or higher tpr corresponds to a lower risk, hence the optimal isocost is the closest to the "north-west corner" (fpr=0,tpr=1) in the ROC space.

That allows the analyst to visually identify the Bayes decision cut-off point as the (tpr,fpr) pair that is closest to (0,1). It also allows them to compare[^f1] two recognizers, with the best performer's isocost being above the underperformer - see example in the next section.

Before looking at an example, let's connect the ROC to the Bayes optimal criterion [TODO: link to previous section]. ROC and likelihood ratio are connected because $(1-\text{Pmiss})$ and $\text{Pfa}$ are linear functions of the cumulative density function (cdf) of $\omega_1$ and $\omega_0$, respectively, so the derivative of the ratio is the ratio of their pdf's, i.e. the likelihood ratio. 

Sliding through every operating point on the ROC curve is the same as going through every possible likelihood ratio (LR), which is the left-hand side in the Bayes equation 1.1 [todo: link]. The right-hand side is a function of application parameters, and so the Bayes optimal solution can be identified on the ROC plot.



With likelihood ratios available, the connection to the Bayes decision threshold becomes apparent by going back to the isocost equation. The derivative of the "north west-most" line is $e^{-\theta}$ and is also a tangent to the ROC curve - if not, we could shift the isocost and get a lower risk. That means that the optimal thresholds are at $\text{LR}=e^{-\theta}$, i.e. $\text{LLR}=-\theta$, which is equation 1.1.

So far, we have made no assumptions about the shape of the ROC curve, so there can be many solutions, but the next section will cover monotonicity, which guarantees that there is only one optimal cut-off.




## B. The PAV algorithm 

So far the `Tradeoff` class evaluates CCD, LLR and ROC from the histogram counts. Though histograms make sense to estimate the probability functions that underpin these evaluation frameworks - e.g. pdf, cdf, (1-cdf) - software libraries do not use histograms. Popular implementations like R's [ROCR](http://ipa-tys.github.io/ROCR/) or python's [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) construct the ROC curve by counting fpr and tpr for every score threshold. This results in thinner intervals and more evaluation points[^f2].

One issue with histograms is to define the bin width. If the bins are too wide, the optimal cut-off may be lost inside a bin, resulting in a higher expected risk. Would it make sense to select smaller intervals? I plot the results of very thin intervals below. The LLR curve goes up and down, i.e. it's not a [monotonous function](https://mathworld.wolfram.com/MonotonicFunction.html) of scores, and the ROC curve now has a steppy pattern, i.e. it's not [convex](https://en.wikipedia.org/wiki/Convex_set).

The binary criterion does not seem compatible with the LLR not being a monotonous function of scores. If $\text{LLR}(p_s)$ intersect $-\theta$ at $s=c$, however, $\text{LLR}(p_s) < -\theta$ for some scores above $c$, then the Bayes decision rule says that we should choose $\omega_0$ for these scores, though the binary classifer will predict $\omega_1$.

We can enforce monotonicity between scores and likelihood ratios by fitting monotonic functions like a simple linear regression: $\text{LLR}(p_s) = \beta_0+\beta_1*s$. There are also non-parametric options to enforce monotonicity and, in fact, one of them called the Pair Adjacent Violators (PAV) is commonly used because it ensures that the resulting ROC curve is convex.

The PAV creates groups of scores with representative thresholds, which I can use as input into the `Tradeoff` object. The resulting ROC curve is convex, while the thin-sized bin histogram ROC appears non convex. 

From a decision perspective, that means that the histogram includes many sub-optimal operating points, i.e. with higher expected risks than what could be achieved by using points on the convex hull. For more information and compelling examples, I would refer to T. Fawcett and A. Niculescu-Mizil (2007).

{% include demo16-histVSpav-10.html %}

With monotonicity now enforced, the solution to the Bayes decision rule is unique. As showed at the end of the previous part, this solution is similar to the one we get when sliding through all (Pmiss,Pfa) pairs, computing expected risk and choosing the lowest value. The latter approach is typically used by software packages, with the ROC operating points used to compute risk estimates.

## C. Risk VS AUC use case


ROC curves are also known for their visual representation of the concordance probability covered in Part 1. The Area Under the Curve (AUC) of the ROC is equal to $p(s_{\omega_0} < s_{\omega_0})$ - see [this](https://stats.stackexchange.com/questions/190216/why-is-roc-auc-equivalent-to-the-probability-that-two-randomly-selected-samples) SO question for a proof. What is not clear so far is why we should care about concordance probabilities. In particular, should we aim for a minimum AUC? 

The next example shows that higher concordance is not necessarily better from a risk optimisation viewpoint, and so, if our objective is to minimise risk, we need not pay too much attention to AUC, which may at best be a good sense check - e.g. is it close or well above 0.5?

The example is based on another blog post, [ML Meets Economics](http://nicolas.kruchten.com/content/2016/01/ml-meets-economics/), which is a great practical introduction to AUC and ROC curves. I only make small adjustments to their numbers to fit my simulated data and I emphasize the reasons for the disagreement between $E(\text{risk})$ and AUC.

Imagine a factory making widgets with the following data
- Widgets overheat 5% of the time due to faulty gearboxes
- Bad gearboxes costs the company £157 owing to wasted labour and inputs
- The gearbox supplier, which sells them at £50 apiece, won't improve quality controls, so the factory decides to use an ML system to predict which gearboxes will overheat
- Any gearbox flagged as faulty must be tested, which destroys it, resulting in a loss equivalent to the item's cost (£50)
- Every working widget is sold for a net profit of £40

If $\omega_1$ ($\omega_0$) represents the defect (working) gearbox and $\alpha_1$ ($\alpha_0$) the decision to test (not test) a gearbox, then the cost & profit structure is


<div class="mermaid"> 
    graph LR;
    id0([start]) --> id11[class ω0];
    id0([start]) --> id12[class ω1];
    id11[class ω0]-->id21[α0]; 
    id11[class ω0]-->id22[α1]; 
    id12[class ω1]-->id31[α0]; 
    id12[class ω1]-->id32[α1];
    id21[action α0] --> id41[cost c_00 = -40.0];
    id22[action α1] --> id42[cost c_10 = Cfa = 50.0];
    id31[action α0] --> id43[cost c_01 = Cmiss = 157.0];
    id32[action α1] --> id44[cost c_11 = 50.0];
</div>

Note the negative cost of not testing the item when it's not faulty, which corresponds to a net profit. In the original blog article, the author maximises utility, while we minimise the expected risk, but the two approaches are equivalent (as long as we use signs consistently).

We can use the simplified cost structure implemented in `AppParameters` by identifying the extra terms and offsetting them. This is actually a useful exercise to realise that the Bayes decision threshold $-\theta$ is only determined by the ratio between 
$$
p(\omega_0)*u
$$ 
and
$$
p(\omega_1)*v
$$
. We decided that $u$ and $v$ should be Cmiss and Cfa, resp., but we could break them down differently.

The Bayes decision criterion is to choose $\alpha_1$ if $\text{lr}(\omega_1)$ is greater than $\frac{(\text{C10-C00})p(\omega_0)}{(\text{C01-C11})p(\omega_1)} = \frac{90 \times p(\omega_0)}{107 \times p(\omega_1)}$, and so the cost structure is equivalent to `AppParameters(p_w1=0.05,Cmiss=107,Cfa=90)`.

In what follows, we evaluate the high- and low-AUC recognizers, which are prefixed with "hi" and "low", resp. The data generation process defines the recogniser scores directly instead of generating the data and fitting recognisers to get the scores, which would be a lot of work for no added benefit. I use the `probability_monad` framework to craft and sample from the score distributions - defined [here](https://github.com/mkffl/decisions/blob/e49290f5f01faadef2f4c383d663cfa28c457741/Decisions/src/Data.scala#L75). 

We start by checking that the AUC of the `hiAUCdata` recognizer (highAUC) is truly higher using the `smartA` function described in Part 1. In `probability_monad`, the `pr` method samples from the rv to evaluate a predicate. Here, we check that the difference in AUC is positive, i.e. that `hiAUCdata`'s AUC is bigger than `lowAUCdata`.

[source](https://github.com/mkffl/decisions/blob/edc8cf34d8e3d82e7fdd1cdb48914e1bd1bfbbd3/Decisions/src/Recipes.scala#L1550)
```scala
    def hiAUCdata: Distribution[List[Score]] = HighAUC.normalLLR.repeat(1000)
    def lowAUCdata: Distribution[List[Score]] = LowAUC.normalLLR.repeat(1000)

    def splitScores(data: List[Score]): Tuple2[Row, Row] = {
      val tarS = data.filter(_.label == 1).map(_.s).toVector
      val nonS = data.filter(_.label == 0).map(_.s).toVector
      (nonS, tarS)
    }

    def score2Auc(data: List[Score]): Double = {
      val (nonS, tarS) = splitScores(data)
      smartA(nonS, tarS)
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
res3: Double = 0.9775823313628146
@ assert(p > (1-α))
res4: Boolean = true
```

The `hiAUCdata`recogniser is better from an AUC viewpoint, however, it has a worse expected risk on the sample dataset:

```scala
    val aucPa = AppParameters(0.05,107,90)

    // Define w1HiCnts etc. using PAV - see source code
    val hiTo = Tradeoff(w1HiCnts,w0HiCnts,hiThresh)
    val lowTo = Tradeoff(w1LowCnts,w0LowCnts,lowThresh)
```

```bash
@ println(hiTo.minRisk(aucPa))
res3: Double = 5.16
@ println(lowTo.minRisk(aucPa))
res4: Double = 2.81
```

The LLR and ROC plots help to make sense of the disagremment between AUC and $E(\text{risk})$. 

The high $-\theta$ value penalises false positives more than false negatives, which makes sense because there are many more non-targets for only slightly lower Cfa. This corresponds to a high optimal score cut-off and a steep ROC isocost located in the lower left-hand corner.

The LLR plot confirms the previous AUC results, as can be seen from hiAUC's steeper LLR curve vs lowAUC's line that is flatter around 0.

lowAUC can identify more positives than hiAUC while keeping fpr low, as can be seen by the steep increase of its ROC curve. In other words, hiAUC needs to be closer to an all-$\omega_0$ classifer to keep a low enough fpr, and that means it has a higher expected risk. 

{% include Demo17-llrRoc4panes.html %}

<h2 id="majority-classifier">D. Majority-isocost lines</h2>
The all-$\omega_0$ "classifier" mentioned above is a simple rule that assigns all instances to non-target. Another name may be majority classifier because it assigns the most prevalent class in the dataset. Its expected risk is a benchmark that any recognizer should beat because it is inexpensive, simple and thus it is often the status quo before any automated solution is considered. 

The all-$\omega_0$ expected risk is $p(\omega_1) \times \text{Pmiss} \times \text{Cmiss} = 0.05 \times 1 \times £107 = £5.35$, i.e. close to hiAUC's risk of £5.16, which thus does not add a lot of value. If hiAUC was our only option, it would be a good idea to double check its costs to ensure that it's better than the simple rule.

The status quo labels instance as targets or non-target depending on which option has the lower expected risk. The isocost line of the status quo - the majority-isocost line (naming is mine) - determines if a recogniser can bring any value on top of the status quo. If the recognizer's ROC curve is below the the line, then its optimal risk is higher than the status quo and it does not make economic sense to use it. 

As explained in the aforementioned blog post, the majority-isocost line can inform go/no-go decisions to develop ML solutions. Besides, I think it is useful for sensitivity analyses of different application scenarios. The application parameter inputs are subjective so using a range of inputs is preferrable if it's possible.

For example, an increase in supplier costs, a change in regulation impacting labour wage or an improvement in the supplier's defect rates would impact the operating context, and therefore the expected risk of a recogniser. Visualising these scenarios via the majority-isocost lines can provide reassurance that a recognizer adds value even in the worst case.

The graph below shows the previous parameters and another scenario that penalises false positives  more, which corresponds to the steeper line. It's interesting that lowAUC still has points above the steeper isocost. 


{% include Demo18-ROC-equal-utility.html %}

A note on the calculation - the majority-isocost line is defined as $\text{(tpr-d)} = \text{(fpr-d)} \times \frac{p(\omega_0) \times \text{Cfa}}{p(\omega_1) \times \text{Cmiss}}$, with $d$ being (0,0) for the all-$\omega_0$ rule, or (1,1) for the all-$\omega_1$ rule. See the details in the appendix.

To determine which all-$\omega_i$ is the reference, compute their expected-risk ratio

$$
\text{rr}=\frac{E(risk_{all-\omega_1})}{E(risk_{all-\omega_0})} = \frac{p(\omega_0) \times \text{Cfa}}{p(\omega_1) \times \text{Cmiss}}
$$

And so, the line equation can also be defined as $(\text{tpr}-d) = (\text{fpr}-d) \times \text{rr}$, which has a nice visual interpretation. If the all-non-target rule has the lower risk, the line equation is $\text{tpr} = \text{fpr} \times \text{rr}$ which lies above the $\text{tpr}=\text{fpr}$ line. Otherwise, it is $\text{(tpr-1)} = \text{(fpr-1)} \times \text{rr}$ which also lies above $\text{tpr}=\text{fpr}$.

Furthermore, the majority-isocost is $(\text{tpr}=\text{fpr})$ when $\text{eer} = 1$ i.e. when application parameters don't carry any information for the Bayes decision threshold. In that case, there is no majority classifier, and we may as well flip a coin to decide which label to choose for all instances. 

A recogniser's ROC is on this line if it can't do better than this randomly assigned majority rule, which means that this recogniser has a concordance probability (AUC) of 0 - that is often how people refer to $(\text{tpr}=\text{fpr})$.

## E. Conclusion
ROC analysis teaches us that finding an optimal threshold is about trading off one error for another. We start with a very high Pmiss and we give it away in exchange for more Pfa ; the application type determines when we get the best deal, in which case we stop the trade and find the corresponding threshold.

Up until now, scores had no meaning other than higher values corresponding to higher target probability. The [next part]({{ site.baseurl }}{% link _posts/2022-03-02-Decisions-Part-3.markdown %}) addresses decision-making using calibrated scores, which can be a better alternative to using raw scores.

### Appendix

#### Details of the majority-isocot equation

Rearrange the expected risk equation (2.1)

$$
\text{tpr}=\text{fpr}*\delta+\frac{p(\omega_1) \times \text{Cmiss} - E(\text{risk})}{p(\omega_1) \times \text{Cmiss}}
$$
With $\delta = \frac{p(\omega_0) \times \text{Cfa}}{p(\omega_1) \times \text{Cmiss}}$

At any point (d,d) on $\text{tpr}=\text{fpr}$,
$$
E(\text{risk}) = p(\omega_0)\times\text{Cfa}\times\text{d} + p(\omega_1)\times\text{Cmiss}-p(\omega_1)\times\text{Cmiss}\times\text{d}
$$

Plug the $(d,d)$ risk value into the expected risk equation, rearrange to get

$$
\text{(tpr-d)} = \text{(fpr-d)} \times \delta
$$

[^f1]: The ROC framework makes it easy to not only compare but also combine recognizers, which can be a better option when one recognizer achieves better risk only on some parts of the ROC curve. See following section on the convex hull, and T. Fawcett and F. Provost (2001).

[^f2]: See "Algorithm 2" in T. Fawcett and A. Niculescu-Mizil (2007). This is similar to constructing histograms with varying-size bins, which gets the lowest bin width such that there is at least a positive value in tpr or fpr. So, one could argue semantics and say that implementations actually use histograms. For an example implementation in R, see [ROCR](https://github.com/ipa-tys/ROCR/blob/master/R/prediction.R) `.compute.unnormalized.roc.curve`. My R is rough around the edges but I think it implements Fawcett's algorithm.

### References
- T. Fawcett and F. Provost (2001). Robust Classification for Imprecise Environments.
- T. Fawcett and A. Niculescu-Mizil (2007). PAV And the ROC Convex Hull.

