---
title: Improving Shapley values with Causal Knowledge
layout: post
---

The SHAP explanation method has been widely used because of useful guarantees, availability across  frameworks and a sense of accessibility/flexibility. The most sought-after guarantee is local accuracy, by which Shapley values add up to explain the difference between a prediction and a baseline. There are fast implementations with good documentations across many frameworks, for example [shapr](https://cran.r-project.org/web/packages/shapr/vignettes/understanding_shapr.html), which supports the next parts of this blog. Finally, SHAP does not seem to require assumptions about the underlying data process and some implementations like KernelSHAP apply to any ML model.

In practice, Shap can return misleading results, which I will illustrate in the first part with two examples. The second part looks at Shapley values as a combination of scenarios and identifies unhappy scnearios that we may want to exclude. Scenario exclusion should agree with any causal knowledge of the underlying data. The last part introduces [Asymmetric Shapley values](https://arxiv.org/pdf/1910.06358.pdf) to add causal assumptions to improve Shapley values.

This blog post assumes some familiarity with SHAP as covered in introductions like [the Interpretable ML online book](https://christophm.github.io/interpretable-ml-book/shap.html#definition).


## A. Unexpected Results

Two examples will illustrate when SHAP may return unexpected results
- A variable is not used, also called a non-interventional variable
- A variable is determined by a parent variable, a case of variable mediation

To anchor the examples, I will use a classic loan approval use case. A financial institution is about to launch an automated loan approval product but they ask us to audit the model first to make sure that it complies with regulation. They want to understand what variables are used to predict a loan outcome. We are told that the only predictor used by the model is the applicant’s income, but we also have data about the applicant's race and favourite TV show.

To evaluate Shapley values, assume that we know the generative process that the bank's ML model is trained on. 

### Non-interventional

{:refdef: style="text-align: center;"}
![Non Interventional](/assets/shap/non-interventional-diagram.png){: width="500px"}
{: refdef}

Diagram of the Data Generative Process (DGP) for the non-interventional example: a loan outcome is determined by the applicant’s Income and their Race. Their favourite TV show is caused by Race but isn’t an input into the outcome. If this causal model seems absurdly simplistic or weird, it probably is but the implications we’ll draw are applicable to more realistic scenarios.

```R
g_non_interventional <- function(N){
	income <- 1 + 3*rnorm(N)
	race <- 1 + 5*rnorm(N)
	favourite_show <- race + rnorm(N)/2
	y <- 0.5*income + 0.5*race + rnorm(N)
	return(cbind(income, race, favourite_show, y))
}
```

`g_non_interventional` implements the generative logic with only numeric variables for simplicity. If you prefer thinking about categorical variables, for example “1: Loan application approved ; 0: Loan application not approved”, you can assume variables like `race`and `y` to be described in the log-odds space, then we could apply an inverse logit transform to describe them as probabilities.

Next, we train an ML model and generate 300 obervations from this DGP to apply kernelSHAP and plot the distribution of shapley values for all 3 variables `income`, `race` and `favourite_show`. 

Shapley value distributions span the same range for all three variables, meaning that `favourite_show` gets attributions often as large as the other two variables although it's not an input into the loan outcome. If we didn't know it, would we conclude that Netflix preferences partially determine an applicant's access to credit? Thinking beyond this exammple, Shapley values of non-interventional variables can be different than zero, which can lead analysts to the wrong conclusions.

{:refdef: style="text-align: center;"}
![Non Interventional](/assets/shap/shapley-distribution-non-interventional.svg){: width="500px"}
{: refdef}

### Mediation / Indirect effect

The next example keeps Income and Race and introduces Zip Code as a mediation variable. Race determines an applicant's zip code area but it has no direct effect on the loan approval outcome. Income and Zip code are the inputs into the loan approval outcome.

{:refdef: style="text-align: center;"}
![Non Interventional](/assets/shap/mediation-diagram.png){: width="500px"}
{: refdef}

```R
g_indirect_effect <- function(N){
    income <- 1 + 3*rnorm(N)
    race <- 1 + 5*rnorm(N)
    zip <- race + rnorm(N)/5
    y <- 0.5*income - 0.5*zip + rnorm(N)
    return(cbind(income, race, zip, y))
}
```

`g_indirect_effect` implements the causal model. Higher income and lower race values increase the chances to get the loan approved. Note that the random error of Race is 1.6 times larger than Income, so the range of values spanned is larger. Also, zip code is mostly determined by `race` but it has a small random component, a small effect on the loan outcome that can't be assigned up to an appplicant's race, which explanations should capture.

The same ML model as before is trained and kernelSHAP is run on 300 observations. Shapley value distributions show, again, that all three variables have the same range. This means that the zip code "explains" the loan outcome as much as race does, although it is predominantly determined by race. Here we would expect race to have more explanatory power than zip code overall, so it seems that SHAP dilutes the true effect of race with the zip code. 

The explanations then underestimate the true effect of race, which could hide discrimination. An auditor may wrongly conclude that a model is compliant because sensitive attributes appear to have explanatory power, as it's spread over child variables. We need a way to correct Shapley values for such situations.


## B. Shapley as an overage of scenarios

### Shapley and scenarios

SHAP explains a prediction by taking each feature and estimating what the predicted value from model `f` would have been without this feature. The comparison involves many scenarios corresponding to coalitions of features. Every scenario is like a single experiment that attempts to answer the question "How much does the feature contribute to the prediction?". The Shapley value combines all the answers into a single value, and it does this for every feature.

In other words, the Shapley value averages scenarios to provie a contrastive explanation Δ(i). Δ explains why we shouldn't drop feature i, assuming it's different than zero. The explanation could be worded like  "You can't drop feature i because it changes the prediction by Δ(i) from the baseline". The baseline can be any reference value and is often chosen to be the expectation `E(f(X))`.

The chain of contrastive statements “explain out” the prediction with reference to the baseline value, a property sometimes called local accuracy because SHAP describes exactly how, starting from the baseline expectation, each feature contributes to the final prediction.

The problem is that for every feature, some of the scenarios that Shapley averages over may not be consistent with our knowledge of the data generating process. We would therefore like to ignore inconsistent scenarios. Section C gives an example of how we can be selective about these scenarios to improve Shapley values.

To recap
- A coalition, denoted S, is a group of features used to predict an instance. In-coalition features are part of the model whereas out-of-coalition features are excluded
- The Outcome function or payoff function is a model’s prediction under a coalition, written $v_{f,x}$ where f is the predictive model and x are in-coalition features
- A scenario is an outcome under a particular coalition, i.e. Δ_v(i, S) where i is the target feature for a prediction. For example, Δ_v(zip, {income, race}) assess the change in f when zip code is added to a coalition of income and race
- The Shapley weight of feature i $\phi_i$ is the average over all possible scenarios
$$
\phi_i = \frac{1}{\text{M}}\sum_{S\subseteq M\setminus \{i\}}{M-1 \choose \vert S \vert}^{-1}(v(SU\{i\})-v(S))
$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $v(SU\{i\})-v(S)$ reports a single experiment's result, which is run for all possible coalitions S and averaged so that smaller and very large coalitions get more weight as ${M-1 \choose \vert S \vert}^{-1}$ is a U-shape function of the number of elements $\vert S \vert$.
- Local consistency is the guarantee that the sum of Shapley weights is equal to the difference between a prediction and a baseline value like E(f(X))

### Zip code example

The scenarios are built from all coalitions that exclude zip, of which there 2^(N-1) so 4

$$
\{\varnothing\} , \{\text{income}\}, \{\text{income, race}\}, \{\text{race}\},
$$

The scenario under the empty set evaluates the impact of adding the zip code when there is no features. The no-feature outcome is $f(\mathbf{E(X)})$ estimated as the average prediction over the training set, and the outcome when only zip code is present can be estimated with a sample of the remaining features conditioned on the zip value. We will come back to this example.

Another interesting scenario is built on the 3rd combination that includes income and race. Looking at the effect of the zip code when income and race are known can tell us if zip code has any explanatory power at all because all other features are included. $\Delta $ captures any remaining effect of zip after everything else has been accounted for.
$$
Δ_v(\text{zip}, \{\text{income, race}\})=v_f(\{\text{income, race, zip}\}) - v_f(\{\text{income, race}\})
$$

This is how a prediction breaks down for one observation, using an application with average income, low race values and a high prediction of acceptance. 

Income: 0.9571218
Race: -4.655322
Zip code: -4.646968
f: 2.58386

Compare these values with all test observations to add some perspective.

{:refdef: style="text-align: center;"}
![Non Interventional](/assets/shap/data_distribution.svg){: width="500px"}
{: refdef}

We know that $v_f,x(\{\text{income, race, zip}\})$=$f(x)$=2.584. What would be the outcome without zip, $v_f,x(\{\text{income, race}\})$? If it is the same as $f(x)$ then the difference Δ_v is zero and through this scenario we would conclude that zip makes no contribution. The outcome without zip is approximately 2.687, which is close, and a statistical test does not allow to conclude that the two values are different, which reinforces the view adding the zip code has no effect on the prediction.

Another interesting scenario is the last coaltion above with only rac, S={race}. Here again, the data supports the hypothesis that $\Delta_v$ is zero, so the zip code has no incremental effect when race is already factored in.

The Shapley value for zip is [1.3] though. In the chart below, each bar shows the Shapley values for one of the three features and the baseline ("None"). All values add up to the applicant's predicted value 2.584 thanks to Local accuracy.

{:refdef: style="text-align: center;"}
![Non Interventional](/assets/shap/test_example_shapley_values.svg){: width="500px"}
{: refdef}

Zip and Race together explain most of the prediction, with zip code approximately 1.23. The remaining scenarios from coalitions $\{\varnothing\}$ and $\{\text{income}\}$ fully account for the non-zero Shapley value of Race. Each scenario captures the effect of zip absent race, which passes any effect race may have onto the zip code, which threfore acts as a proxy for race. If we know that race is a causal ancestor of zip code, then we would like the zip Shapley value to only measure any additional effect, not the total effect of race and zip. Asymmetric Shapley value allows to rule out these two scenarios while keeping local accuracy. Let's look at an implementation right after clarifying one point about missing features estimation.

### Estimating Shapley outcome functions $v_{f,x}$

To get $v_{f,x}$ we can retrain the same model keeping in-coalition features, e.g. $v_{f,x}({\text{race}})$ by dropping income and zip code. But training 2^M models where M is the number of features wouldn't be efficient in many cases, so we average the model predictions on a sample of likely values for out-of-coalition features. For example, if only race is known, the "likely" values of income and zip are determined by $p({\text{income, zip}} \vert \text{race})$. 

Shapley has no opinion on how the outcome function is calculated or what the feature distribution is, so we make the best decision possible. Here the features are known to be generated from a normal distribution so I use `shapr`'s multivariate gaussian but this package has other options - parametric and otherwise - available. Worth noting that some packages only provide empirical estimates by independently sampling each feature, which makes samples prone to unrealistic hallucinations. For example, if features include smoking beahviour and age, sampling independently may
include 5-year old chain smokers, which the model is not trained to adequatly repond to - what would an adequate prediction even mean?

To recap, $v_{f,x}(\text{race})$ is approximated with $E_{X_{\text{income, zip}} \vert X_{\text{race}}}f(x_{\text{race}}, X_{\text{income, zip}})$ with lower case x the known value and upper case X the conditionally distributed values. $v_{f,x}(\text{race, zip})$ is also estimated and the two sampling distributions overlap:

{:refdef: style="text-align: center;"}
![sample distributions 010 vs 011](/assets/shap/outcome-samples-v_race-v_race_zip.svg){: width="500px"}
{: refdef}

A t-test testing if the mean difference is zero returns a p-value of c.0.34, supporting the view that $\Delta_v(\text{zip}, \{\text{race}\})$ = 0 i.e. that zip code has not incremental effect on the prediction when only race is known.

Other scenarios that capture meaningful variables show opposite results with clearly different distributions, e.g. $\Delta_v(\text{income}, \{\text{race}\})$:

{:refdef: style="text-align: center;"}
![Non Interventional](/assets/shap/outcome-samples-v_race-v_income_race.svg){: width="500px"}
{: refdef}




## C. Asymmetric Shapley values



## References
- C. Frye, C. Rowat and I. Feige (2020). Asymmetric Shapley values: incorporating causal
knowledge into model-agnostic explainability.
- E. Kumar, S. Venkatasubramanian, C. Scheidegger and S. Friedler (2020). Problems with Shapley-value-based explanations as feature importance measures.