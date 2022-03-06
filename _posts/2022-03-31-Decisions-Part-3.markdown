---
title: My Machine Learnt... Now What? - Part 3
layout: post
---

The [previous part]({{ site.baseurl }}{% link _posts/2021-10-28-Decisions-Part-2.markdown %}) looked at the ROC approach to binarise scores while minimising expected risk. The only constraints we have put on scores is that higher values mean higher probability for the target class. Sometimes, however, we need scores to be calibrated i.e. to tell us something about the likelihood of occurrence of the target event. 

To make sound business decisions, calibrated predictions must be evaluated just like with raw scores. NIST SRE have built an evaluation framework that provides answers to key questions that underpin the deployment of an automated solution:
- What calibrated system outperforms others on a range of application types? What if we care about all possible application types?
- What is the expected risk of my calibrated system?
- How does my system fare vs a majority vote approach (see to [previous part]({{ site.baseurl }}{% link _posts/2021-10-28-Decisions-Part-2.markdown %}))
- What would be the answers above if my systems are perfectly calibrated?

Note: Really, the only question left is “Why do more people not use that framework?”

We start by looking at a few scenarios that require score calibration. Then we will review two types of score calibration: as probabilities and as log-likelihood ratios (llr). The last section introduces the NIST SRE solution to evaluate calibrated predictions based on llr-like scores.

A note on external source and the code used for this blog - the NIST SRE literature referred to in this blog is listed at the end. The source code that underpins the examples is available at https://github.com/mkffl/. The financial fraud use case is based on the same data generating process detailed in Part 1. The main recognizer is based on SVM and the competing recognizer is based on Random Forests.

## A. 