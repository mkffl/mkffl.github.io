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

## A. Why use calibrated scores

Calibration is concerned with scores that enable [Bayes decisions]({{ site.baseurl }}{% link _posts/2021-10-18-Decisions-Part-1.markdown %}), whereas row or uncalibrated scores require an extra step before making a risk-minimising decision. For example, ROC threshold optimisation requires to use a separate evaluation dataset to slide through every operating point and a) grab the corresponding (Pmiss, Pfa) b) calculate the expected risk at that point and c) find the threshold associated with the risk-minimised operating point.

While there's nothing wrong with that extra step, it can be inefficient compared to calibrated scores. I can think of 3 scenarios (there may be more):

#### The “Human in the Loop” System

Sometimes, hard (e.g. binary) predictions are not good enough, and the agent using the predictions needs scores that carry more information than simply “this is a negative class prediction”.

For example, a fraud detection system may involve human supervision when it is uncertain about an outcome. Uncertainty typically refers to a likelihood of incidence given by the score. The detection system may send transactions to human reviewers if their predicted incidence is between 30% and 50% because at these levels the cost of miss still outweighs the cost of using expert labellers.

Human labellers often have access to more information than what is in the the training dataset, which leave out valuable information for technical or legal reasons. For example, they may review  online social media data to check if a user has several identities, or they could request access to a restricted PII database available on a case by case basis.


#### Deployment with varying application types

Sometimes, predictive systems must be deployed in different contexts i.e. with varying prevalence rates and/or error costs. Here, calibrated scores can help ease the deployment process. 

If a bank that trialled a binary fraud detection system eventually finds that it saves money and safeguards its reputation, it would naturally roll it out in more contexts, in new regions. It makes sense then to avoid burdening the system with an extra step to optimise raw scores via a separate database of transactions. With calibrated scores, the systems can be managed centrally and local entities just need to apply a threshold that depends on their application types.

#### Combining multiple subsystems

Systems that make decisions based on multiple subcomponent outputs may require calibrated predictions. The motivating example of the NIST SRE literature is identify detection using different systems each built on separate biometric data, e.g. fingerprints, face recognition or voice signals. 

If all prediction outputs speak the same language - that of calibrated probabilities - they can be combined to form a decision such as “this person is who claim they are”.

Another example may be an autonomous vehicle (AV example) that makes a decision to stop at a traffic light using calibrated signals from cameras, Lidars, radars, and other sensors.
