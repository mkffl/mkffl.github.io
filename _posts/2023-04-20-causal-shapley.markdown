---
title: Improved Shapley values with causal Knowledge
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


The diagram of the Data Generative Process (DGP) says that a loan outcome is determined by the applicant’s Income and their Race. Their favourite TV show is partially determined by their Race but isn’t an input into the outcome. If this causal model seems absurdly simplistic or weird, it probably is but the implications we’ll draw are applicable to more realistic scenarios.

```R
g_non_interventional <- function(N){
	income <- 1 + 3*rnorm(N)
	race <- 1 + 5*rnorm(N)
	favourite_show <- race + rnorm(N)/2
	y <- 0.5*income + 0.5*race + rnorm(N)
	return(cbind(income, race, favourite_show, y))
}
```

A structural equation model (SEM) implements the generative logic with only numeric variables for simplicity. If you prefer thinking about categorical variables, for example “1: Loan application approved ; 0: Loan application not approved”, you can assume variables like `race`and `y` to be described in the log-odds space, then we could apply an inverse logit transform to describe them as probabilities.

Next, we generate 300 obervations from this DGP to apply kernelSHAP and plot the distribution of shapley values for all 3 variables `income`, `race` and `favourite_show`. All three variables span the same range, meaning that `favourite_show` gets attributions often as large as the other two variables although it's not an input into the loan outcome. If we didn't know it, would we conclude that Netflix preferences partially determine an applicant's access to credit? Thinking beyond this exammple, Shapley values of non-interventional variables can be different than zero, which can lead analysts to the wrong conclusions.

[Add Shapley diagram]

### Mediation

The next example assumes that Race isn’t used by the model but it determines an applicant’s zip code, which is a model input. The generative model is defined as

## B. Shapley as an overage of scenarios



## C. Asymmetric Shapley values



## References
- C. Frye, C. Rowat and I. Feige (2020). Asymmetric Shapley values: incorporating causal
knowledge into model-agnostic explainability.
- E. Kumar, S. Venkatasubramanian, C. Scheidegger and S. Friedler (2020). Problems with Shapley-value-based explanations as feature importance measures.