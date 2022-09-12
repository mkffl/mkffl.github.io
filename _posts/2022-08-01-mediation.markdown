---
title: Practical Application of Causal Analysis
layout: post
---

I have recently read [The Book of Why (TBoW)](http://bayes.cs.ucla.edu/WHY/) by Judea Pearl and Dana McKenzie. It is an introduction to causal inference, which aims to answer causal questions using data. The book is written from the perspective of J. Pearl, who has made major contributions to this field. 

Through 10 chapters, the book illustrates key concepts of causal analysis with applications. A number of pages also discuss the differences between traditional stats and causal analysis, and J. Pearl's own experience (and frustrations) sometimes come out from the tone and anecdotes.

Though I think the book could have done with fewer autobiographic passages, it is a useful introduction to causal thinking for beginners like myself. The focus of the book is primarly on the practical benefits of causal theory, and thus it avoids a common pitfall of academic debates - I am looking at you, bayesians vs frequentists.

As I read the book and discovered this new discuples, my main questions have been
- What does causal analysis allow me to achieve beyond what mainstream statistics already does?
    - Does it provide better or faster answers to the same problems?
    - Can it help me solve new problems?
- How do I get started?
    - What is the aboslute minimum theoretical baggage needed to start applying causal analysis to real-world problems?
    - What software packages should I get familiar with?

In this blog article, I will try and answer the first question using a case study on employee racial discrimination. As chapter 9 of TBoW explains, traditional ansalyses have not given a satisfying answer to this important class of problems.

In what follows, I will try and apply causal inference tools to unpick the problems behind a  question like "does organisation XYZ discriminate against its people?". I will be mostly concerend with the intuitions rather than mathematical rigour and the ability to generalise to other problem classes. I write this blog mostly for myself, to check if I really understood what TBoW had to say about mediation analysis.

- minimalist code
- intuitions behind math equations


[
I have recently read Judea Pearl and Dana McKenzie’s [The Book of Why (TBoW)](http://bayes.cs.ucla.edu/WHY/), which is an introduction to causal inference. I found the book to be autobiographical as it describes the tensions and frustrations experienced by J. Peal with traditional statistics, which seeks to learn from data without assuming a particular model. Causal inference, meanwhile, requires a model-based view, i.e. an assumption about the generative mechanism underlying the observed data.

Pearl bases his demonstration of the benefits of causal theory on real-world problems that need solving. In doig so, he avoids a common pitfall of engaging only on a philosophical level, which I have found to be sterile and unconvincing. (For example, see the endless debates between frequentist and bayesian statistics.) Instead, when asked to choose between the scientific status quo and an alternative - here, the so-called Causal Revolution - one should ask “How does the new thing help to solve existing problems better/faster? Does it help to solve new problems?”. In this blog article, I apply causal analysis to a classic problem in statistical inference. I will also play with causal diagrams to assess if they are as useful as J. Pearl claims.

- don't assess the book as a literary commentary but discuss its main conclusions - 
]
### Promotion cycles at BigBankCorp

In Chapter 9, TBoW discusess the Berkeley admission paradox, where the Dean of the Berkeley university asked if the admission process discriminated against women. An analysis of admissions by gender showed that, while total admission rates were lower for female, they were higher or equal to males for each department. In what follows, I apply the same type of problem in a slightly different context where the head of a HR department asks if promotions are fair towards minority candidates. I use synthetic data to playa round with causal analysis tools to provide an answer.

- Note: later realised that The HR admission use case is another popular use case described in various paper by Pearl
- Note: 8572 employees, which comprises of 35% BAME employees ; two departments
- What questions asked?

[This is an example of Simpson’s paradox, which is not really a paradox after we realise that females applied to departments with lower admission rates. However, it's not clear if Berkeley's admissions were unfair towards female applicants.]

In my old company, HR asked our team if minority employees were discriminated against during promotion rounds. This happened around the time of George Floyd’s death in the US, which prompted internal discussions about fairness for blacks and minorities in general. Imagine that BigBankCorp, a large lending company, annually reviews the performance of its junior staff to determine if they are ready to move to middle management positions. HR ask you if, on the basis of the data below, the promotion process is unfair towards minority employees, which in the UK are often identified as Black Asian and Middle Eastern (BAME).

### A. Identifying discrimination

{% include mediation/mediation_analysis11_p1.html %}
{% include mediation/mediation_analysis11_p2.html %}

Overall promotion rates for BAME employees stand at 2.7%, almost half the rate for non-BAME employees. However, when looking at the rates by business unit (Consumer vs business-to-business), the chances of promotion are similar or very close.

“Should we double check the figures?”, asked the HR analyst, looking confused. “Should I cut the data in a different way, for example, using more granular business departments? What to conclude if percentages are still similar between minorities and non? What if they show an inverse trend, as we thought would be the case? Hold on, what's the question, again?”

A causal diagram can help visualise and answer the question. We might agree that the situation can be represented as follows - "might" because there could be multiple ways to model BigBankCorp's promotion process, as will become clear.
- Note: thinking about all the factors influencing an employee's successful promotion and linked to their ethnic category, they vary by business unit, so the model is simple and described as


<div class="mermaid" style="width:180px; margin:0 auto;">
graph TD
E((ethnic_category))
D((business_unit))
P((outcome))

E --> D
E -->|?| P
D --> P
</div>

An employee's ethnic category is either BAME or non-BAME, and that determines the business unit they choose to work in, which is either Consumer of B2B. Consumer has lower promotion rates than B2B. This makes sense because BigBankCorp's B2B department is more recent than Consumer, with more opportunities to grow inside the firm. Besides, the ethnic category may also determine the likelihood of being promoted, in which case the selection process discriminates on a racial basis. But how do we know if there is such an arrow from ethnicity to outcome?

Under this causal model, the solution is to holding the business unit constant. If the effect of ethnicity onto the promotion outcome only happens through the elected business unit, then holdiing the latter constant blocks the arrow from ethnic category to business unit, allowing the analyst to observe the effect from business unit to outcome. If there's also a direct effect, then holding the business unit constant does not block the mediated path, and the direct, discriminatory effect will show in promotion rates by BU.

Here, under these causal assumptions, we conclude that BigBankCorp's managerial review process does not discrimate against minority employees. In fact, the data was generated using a simple model with no direct arrow from E to O. 

[source](https://dev.azure.com/mkiffel/_git/personal-blog?path=/blog-mediator/blog_mediation/model.py&version=GBmain-mediation&line=8&lineEnd=9&lineStartColumn=1&lineEndColumn=1&lineStyle=plain&_a=contents)
```python
model1 = pm.Model()

with model1:
    ethnic_category = pm.Bernoulli("ethnic_category", 50 / 100.0)

    p_b2b = if_else(
        is_equal(ethnic_category, 1.0),
        # BAME
        # p(department=b2b|ethnic_category=BAME) = 21.0%
        21 / 100.0,
        # Non BAME
        67 / 100.0
    )

    business_unit = pm.Bernoulli("business_unit", p_b2b)

    p_promotion = if_else(
        is_equal(business_unit, 1.0),
        # b2b
        # p(outcome=promotion|business_unit=b2b) = 7.27%
        7.27272727 / 100.0,
        # Consumer
        1.34328358 / 100.0
    )

    outcome = pm.Bernoulli("promotion", p_promotion)
```

`model1` represents the outcome of a BigBankCorp employee review process as a random variable, which depends on the employee's minority status only via the business unit they chose to work in. We can check the absence of direct connection by inspecting `model1`'s graph

```python
pymc3.model_to_graphviz(model1)
```

{:refdef: style="text-align: center;"}
![Model 2 Diagram](/assets/analysis12_diagram.html.svg){: width="180"}
{: refdef}

I sampled from `model1` to generate the dataset used for the charts above.

If the promotion process was discriminatory, the analyst would observe unequal promotion rates even when holding the business unit constant. To double check this, we can build a model that emulates the direct effect of ethnic category on the outcome.

TODO: add source
TODO: fix numbers (wrong direction)
```python
model2 = pm.Model()

with model2:
    ethnic_category = pm.Bernoulli("ethnic_category", 65 / 100.0)

    business_unit = if_else(is_equal(ethnic_category, 1.0), 60 / 100.0, 40 / 100.0)

    business_unit = pm.Bernoulli("business_unit", business_unit)

    drop_minority_application = pm.Bernoulli("drop_minority_application", 50 / 100.0)

    p_promotion = if_else(
        # If minority
        is_equal(ethnic_category, 0.0),
        if_else(
            # If is discarded
            is_equal(drop_minority_application, 1.0),
            0.0,
            if_else(
                # If b-to-b
                is_equal(business_unit, 1.0),
                20 / 100.0,
                12 / 100.0,
            ),
        ),
        if_else(is_equal(business_unit, 1.0), 20 / 100.0, 12 / 100.0),
    )

    outcome = pm.Bernoulli("promotion", p_promotion)
```

Discrimination happens through the `drop_minority_application` node, which randomly rejects a BAME employee's application with a probability of 0.5. I picture a panel of case reviewers with a particularly bigoted assessor, who uses their veto power to discard one in two BAME employee application without even looking at it. `model3`'s graph confirms the direct effect of E on O.

```python
pymc3.model_to_graphviz(model2)
```

TODO: tweak CPD to have admission rates in 2-6% range
{:refdef: style="text-align: center;"}
![Model 2 Diagram](/assets/analysis22_diagram.html.svg){: width="300"}
{: refdef}

And running the previous analysis on a random draw from the model confirms that the discriminatory effect is identified.

{% include mediation/mediation_analysis21_p1.html %}
{% include mediation/mediation_analysis21_p2.html %}

Now, BAME candidates have a lower chance of acceptance even when holding the business unit constant.

I have called discrimination the direct effect of E on O, because I focus on the performance review process. There may be discrimination when BigBankCorp hires new employees by influencing which business unit they apply for. It's not impossible, though I find it hard to believe, as BAME individuals more likely self-select the department they choose to work in. I think that one benefit of causal diagrams is to make it very clear what type of effect we want to measure.


### B. Keep your "controlling urges" in check

The previously described approach was also applied by Bikel, U.C. Berkeley's analyst appointed by the dean to report any evidence of discrimination. Bikel saw that admission rates by department (math, biology, etc.) were not lower for females, so, he concluded that the university acceptance procedure did not discrimimnate against women. But the story doesn't stop her, and TBOW reports on a conversation between Bikel and Krushke, another statistician who got interested in the case. Krushkal took note of Bikel's result, though he claimed that it did not prove that there were not other causes of discrmination, and he built a simple numeric example as proof. 

I wanted to reproduce Krushkal's small example because I find minimalist use cases very useful as learning tools. However, I could not access the original document referenced in TBOW because of academic paywalls. So, I have used J. Pearl's notes to cook up a hopefully similar model, and I applied it to my BigBankCorp use case. It will illustrate another type of causal effect called a collider, which J. Pearl uses to debunk the deeply anchored myth that statistical analysis should hold any observed variable constant when measuring an effect.

`model3` 's source of discrimination candidates' citizenship (referred to as C), with values as local" or "expat". Further assume that BAME employees are always rejected if C is "local", and non-BAME employees are always rejected if C is "expat". In the other cases - BAME expats or non-BAME local - the chances of promotion are similar. These strong assumptions may not seem credible, but they make the maths easier. We could use a smoother hypothesis to the same effect. But, remember that the goal is to show that there exists a model with discrimination, which returns the same query results as `model1`.

The model definition is available at [provide link], and its graph below shows a direct link from E and C. 

{:refdef: style="text-align: center;"}
![Model 2 Diagram](/assets/analysis32_diagram.html.svg){: width="300"}
{: refdef}

Running the same queries a before gives the following results.

{% include mediation/mediation_analysis31_p1.html %}
{% include mediation/mediation_analysis31_p2.html %}


The promotion rates look very similar to `model1`'s and that is by design. In fact, the expected values of these probabilities are the same for both models. That means that the small differences are only due to sampling variations or, alternatively, we get the same results if we repeatedly sampled from the models and averaged the query results. The appendix includes the deriation and the results from repeated sampling.

Imagine that we correctly identify that the true generative process is `model3`, i.e. we rightly assume that the diagram just above reflects reality, but we don't know if an arrow's value is zero or not. In particular, we are not sure if E connects directly to O with a value that's not zero. How do we know if BigBankCorp discrimiates against BAME employees? 

The answer depends on the variables observed. If C is not observed, then blocking the mediation effect E->B->P is possible only by observing promotion rates by ethnic groups, i.e. the first query captures only the direct effect. The second query captures both the direct and the mediated effect. The reason is that under `model3`, B becomes a collider, a type of node that blocks information when not held constant. But when it's fixed at a certain value, the collider lets information pass from parent nodes to its children. This is why the second query does not reveal the discriminator effect E->O, as it gets "diluted" with the mediated effect E->B->O.

<details>
    <summary>Colliders</summary>
    Colliders are interesting beasts that seem to play games with our intuitions. My cat occasionally triggers my house alarm when she plays insdie (false positive), and in very rare instances a burglar breaking into my home would also trigger the alarm (true positive). Assume that no other factors would trigger the alarm. My cat's behaviour is independent from a burglar's decision. When the alarm's on, I usually think it's because of my cat so I don't panic. But, if my neighbour calls me because the alarm is on while I am on holiday, and my cat stays at some friends', then suddenly I think of a burglary. Here you have it - if I don't know the state of the alarm (on or off), then the two causal factors are independent, i.e. knowing that the cat's not at home tells me nothing about a potential thief. But, conditioned on the alarm ringing, knowing that the cat's away increases the probability of a burglary. That is, holding the collider at the value "on" allows the "cat" information to flow through and it changes the probability of downstream nodes.
</details>
<b>

If C is observed, then holding both C and B constant allows only the direct effect to propagate, as per the results below.

{% include mediation/mediation_analysis31_p3.html %}

To recap - when looking at the direct effect of C on O, and if C is not observed, then we don't want to control for B. Adding B to the set of variables of, say, a logistic regression, just because B is available, may lead us to the wrong conclusion that there's no direct effect. Causal theory teaches us to . 



### C. Measure mediation effects

We have tools to know if effects exist, and now we would like to estimate their intensity to answer questions like

>Does discrimination account for most of the promotion difference?

The answer to this question, combined with BigBankCorp's goals, can guide the company's response. If discrimination accounts for a tiny part of the promotion gap, the company may still want to remove it for ethical and reputational reasons, but it may spend most of its resources on e.g. raising awareness of B2B career opportunities for BAME graduates. Thus, the two effects command vastly different interventions.

As a side note, causal analysis enables "ationable insights", a term that has became fashionable but that many analytics projects fail to deliver. This situation may sound oddly familiar - a team present their final results to all project parties, everyone agrees that the content is "interesting", but no one knows what to do next. "Actionable" insights is a thing that everyone talks about but most have never experienced. 

At times, I have felt that only controlled experiments could generate data used for real decisions, because observational data is almost always fraught with confounders. Causal analysis provides tools to infer the results from interventions in the real world using obervational data - provided we have a causal model of the world.

Going back to BigBankCorp's use case, remember that the total effect of ethnic group on promotion rates is [0.03912082709307785], as shown on the first chart. Direct and indirect effects are calculated by imagining scenarios where some attributes change, and measuring the impact. It is like simulating and comparing different worlds.

Starting with the direct effect, we can ask

> How many BAME people would have been promoted if the same proportion worked in B2B as for non-BAME employees

B2B has higher promotion rates, so this adjustment would bring promotion rates closer between the two groups. For each business unit, keep BAME employee's chance of promotions but weigh it by non-BAME employee's frequency:
$p(O_p\vert E_{bame}, B_b) \times p(B_b \vert E_{non})$

Doig this effectively neutralises the difference in BAME employees' choice of business unit, which is the indirect effect. 

In that hypothetical scenario, $\sum_b p(O_p \vert E_{bame}, B_b)*p(B_b \vert E_{non}) \times 3000$ BAME individuals would have been promoted if not for the discriminatory nature of BigBankCorp’s performance process. 
We can then compare this number with a baseline scenario where BAME employees get treated exactly like non-BAME employees, resulting in $p(Op \vert E_{non}) \times 3000$ promotions. The difference is the direct effect, expressed as a frequency (not as a number of individuals):

$\text{de} = \sum_b p(Op \vert Eb, BUb)*p(B_b \vert Enon) - p(Op \vert E_non)$

In the literature, the direct effect is called "natural" to refer to the baseline weights of the mediating variable, and it's calculated by `natural_direct_effect` below. For BigBankCorp, the value is [-0.029156843426365042], i.e. ethnic category reduces performance by c. 2.9 percentage points, or equivalently, about 2.9 percent of BAME candidates don't get promoted solely because of discrimination. That's about three quarters of the total effect, so BigBankCorp has every reason to make the fight against discrimination their top priority.

source
```python
class MediationMeasurementBinary:
    def __init__(self, data):
        self.data = data

        # p(Op|E)
        self.p1 = probability1(data)
        
        # p(Op|E,B)
        self.p2 = probability2(data)
        
        # p(Bcons|E)
        self.p5 = probability5(data)

    def total_effect(self):
        """ TE = p(O_p|E_bame) - p(Op|Enon)
        """
        return (
            self.p1[1]-self.p1[0]
        )

    def natural_direct_effect(self):
        """ NDE = \sum_b p(Op|Ebame, BUb)*p(BUb|Enon) - p(Op|Enon)
        """
        return (
            self.p2[Employee.BAME,Business.B2B] * (1-self.p5[Business.B2B])
            + self.p2[Employee.BAME,Business.CONSUMER] * self.p5[Business.B2B]
            - self.p1[Employee.NON_BAME]
        )
```

We may conclude that the indirect effect is the difference between the total and the direct effect, c.1%, and we are all done. However, things don't work exactly like that, so let's compute it from first principles then discuss why effects don't always add up. 

The indirect effect works in a similar manner, by asking

>How many non-BAME employees would have been promoted if they had chosen their BUs with the same frequency as BAME employees. 

Think about it - doing this neutralises the direct effect because non-BAME do not suffer from any discrimination, but it does introduce the bias owed to the choice of business unit. [note on bias and discrimination]. So, the recipe is: keep the observed non-BAME promotion rate, $p(Op \vert E_{non}, B_b)$, and time it by $p(Op \vert E_{bame}, B_b)$.

The natural indirect effect is also expressed as the difference between this simulated probability and the baseline probability $p(B_b \vert E_{non})$:

$\text{nie}=\sum_b p(Op \vert Enon, BUb)*(p(BUb \vert Ebame) - p(BUb \vert Enon))$

which estimates the number of non-BAME promotes in the hypothetical scenario that only the frequency of business units would differ. The method `natural_indirect_effect` computes this quantity, estimated at [0.02165948519028952] for BigBankCorp.

source
```python
class MediationMeasurementBinary:
    # ... 

    def natural_indirect_effect(self):
        """ NIE = \sum_b p(Op|Enon, BUb)*p(BUb|Ebame) - p(Op|Enon)
        """
        return (
            self.p2[0,0]*(1-self.p5[1]) 
            + self.p2[0,1]*self.p5[1]
            - self.p1[0]
        )        
```

That means that the choice of business unit by itself drives a of 2.1% of promotions, i.e. not miles away from the 2.9% direct effect. From an intervention perspective, if BigBankCorp's primary concern was to close the promotion gap, management should still focus on discrimination, but may decide otherwise if the indirect effect had been larger than the direct one.

#### Non additive effects

Direct and indirect effects don't add up if the direct effect varies with the mediator (add source). For BigBankCorp, this means that the intensity  of discrimination varies by business unit. The previous chart  reveals it, as the gap between the pale brown consumer bars is smaller than the gap between the dusky purple B2B bars.

It's easier when looking at the same numbers in a table 


|          | Non-BAME | BAME | CDE  | 
|----------|----------|------|------|
| B2B      | 8.3      | 3.7  | -4.6 | 
| Consumer | 2.1      | 1.0  | -1.1 | 

If the CDE values were approximately the same for each business unit, we could rule out interactions, and effects would add up to the total effect. CDE stands for Controlled Direct Effect and carries the same meaning as the NDE, only holding the business unit constant. It asks - Within each business unit, how many BAME employees would have been promoted if not for their ethnic category? Holding the mediator value constant neutralises any indirect effect, hence allowing discrimination only. About 4.6% of BAME individuals in BAME were not promoted due to discrimination, and this is over 4 times higher than in Consumer, so the direct effect does interact with the mediating effect.

As a side note, that result can be surprising because in `model2` [incl link], discrimination is built in through a flat 50% rejection rule via `drop_minority_application`. This design suggests that discrimation does not vary by business unit, although after more careful inspection this is only true in the log-probability space, as the CDE then becomes

$\log \{0.5 \times p(Op \vert E_{non}, B_{b})\} - \log p(Op \vert E_{non}, B_{b}) = log {0.5}$

for each business unit b. So, the results from the sample are consistent with the data generative process.

[REMOVE] If effects added up, the difference in promotees could be neatly separated beetween BAME inviduals who did not get promoted due to the business unit they chose (more BAME employees work in Consumer compared to non-BAME), and those who were discriminated against. But, effects are more likely intertwined, as discrimination may impact business units differently, e.g. with a higher discrimination in B2B vs Consumer.

TODO: Don't add up but reconciliation
TODO: necessary and sufficient

#### Implications for interventions

If BigBankCorp's HR successfully remove discrimination before next year's round of promotions, only the natural indirect effect will remain. So, if the Head of HR asks if a successful intervention will close the promotion gap between employee types, the analyst should revise their expectaions - ending discrimination will reduce the gap, but there will remain an expected difference of c.2.1% owing to employees' business unit choices.

With no direct effect, all that remain are indirect effects, which by themselves amount to an expected 2.1% of BAME employees. The literature refers to this measurement as the sufficient cause for indirect effects, i.e. excluding any direct mechanism. Similarly, the 2.9% figure referred to the sufficient cause for direct effect (excluding any mediation mechanism). The difference between total effects and sufficient causes are called necessary. For example, the necessary mediation effect here is 3.9-2.9=1.0, which corresponds to what discrimination on its own can't explain, i.e. the part the direct effect that relies on mediation to exist. The concept of Sufficient and Necessary causes have connections to legal [] - see [this] for more details

### More comments
- counterfactuals
- working with causal models
    - more complex models
    - automated identification
- 

### Conclusion


The product of these two terms is a joint probability of a special type, which can be expressed in notation using counterfactuals, but I will stick to traditional probability notation to keep things familiar. So
### Appendix 1

I use pymc3 to craft and sample from imaginary Data Generatin Processes (DGPs). The DGP meet some requirements that Ithen verify with causal inference methods.

In general, simulated data - or fake/dummy/synthetic (that one sounds more scientic) - can be generated using `scipy` or `numpy`'s random variables, which is fine for simple DGPs like a multivariate gaussian, but beyond this I find the code unreadable. Probability graphs are easier to read because they define the process for one observation, and the sampling method takes care of the rest.

`pymc3` might sound overkill for this, because I don't need any bayesian inference functionality just to create DGPs, but the library interface is simple and it comes with useful tools like diagram visualisation. My top choice would have been scala's `probability_monad`, which I used for another article of this blog (TODO: link), but python has causal inference packages like [doWhy](https://github.com/py-why/dowhy).

A bayesian probabilistic graph is a set of conditional probability distributions (CPD) for each node. (starting nodes like Citizenship are just prior probabilities, i.e. not conditioned on any variables). To get a DGP, I need to define each node's probability distribution, conditioned on the depending nodes. For example, Business Unit ($p(B_b \vert C_c, E_b)$) is Bernoulli rv defined for all 4 cases corresponding to the values that (C, E) can jointly take.

Defining a DGP in this way means that constraints can be enforced by expressing them as CPDs defined in the graph. See the next appendix for details.

### Appendix 2

Notation

$O_p$: Outcome=promoted<br/>
$E_b$: Ethnic group=BAME ($E_{non}$ for non-BAMEs)<br/>
$C_{ex}$: Citizenship=expatriate ($C_l$ for locals)<br/>
$B_{cons}$: Business unit=Consumer ($B_{b2b}$ for business-to-business)<br/>

The constraints are

#### Requirement 1
For every b in {Consumer, b2b}<br/>
$$
    p(O_p \vert E_b, B_b) = p(O_p \vert E_{non}, B_b)
$$

For Consumer, this expression implies<br/>

$$
p(O_p \vert D_{cons}, E_{non}, C_l) = \frac{p(O_p \vert B_{cons}, E_b, C_{ex}) \times p(C_{ex} \vert E_b, B_{cons})}{p(C_l \vert E_{non}, B_{cons})}
$$

I skip the details, but it's essentially integrating out and using the fact that we set $p(O_p \vert B, E_{non}, C_{ex})=0$ and $p(O_p \vert B, E_b, C_l)=0$.

Then, $p(C_{ex} \vert E_b, B_b)$ can be expanded into an expression that takes CPDs that we set: $\frac{p(B_b \vert C_{ex},E_b) \times p(C_{ex})}{\sum_c(p(B_b \vert E_b, C_c) \times p(P(C_c))}$. So, I get a value for $p(O_p \vert D, E_{non}, C_l)$ that I can set in the graph.

#### Requirement 2
The overall chances of promotion are lower for BAME employees:<br/>
$p(O_p \vert E_b) < p(O_p \vert E_{non})$

That one's interesting. The expanded form is

$$
\sum_b p(O_p \vert E_b, C_{ex}, B_b)\times p(C_{ex})\times p(B_b \vert C_{ex}, E_b) < \sum_b p(O_p \vert E_{non}, C_l, B_b)\times p(C_l)\times p(B_b \vert C_l, E_{non})
$$

The middle terms, $p(C)$, can be ignored because we assume that there are fewer expatriates, so $p(C_{ex})<p(C_l)$. The remaining two probablilities are that of getting promoted conditioned on ethnicity and business unit, times that of the chosen business unit given E. 

The first probability is already fixed to meet requirement 1, so it's out of my control. I tried some substitutions, but I failed to get an expression that would garantee that the requirement is met, and the best I could do is to guess a value and do the calculation to check if the requirement is met. We can come up with a good guess, though - if b2b has a higher promotion rate than Consumer, then setting a higher probability of Consumer for BAME employees will mechanically reduce their promotion rates compared to non-BAMEs. 

That's essentially Simpson's paradox in full force, because as shown on `model3`'s chart, promotion rates conditioned on E and B are higher for BAME employees than non-BAMEs, and yet, requirement 2 is also true.

#### Requirement 3
$p(O \vert E)$ and $p(O \vert E, B)$ must be similar in `model1` and `model3`.

This is guaranteed by inputing `model3`'s CDP into `model1`, using $p(B \vert E)$ and $p(O \vert E,B)$, which can be easily derived. For example<br/>
$p(B_{b2b} \vert E_{bame})=\sum_{c}{p(B_{b2b} \vert E_{bame}, C_c) \times p(C_c)}$.

#### Empirical checks

It's possible to verify that requirements are met by way of sampling. That approach was not available in 1970s, when all statistical work had be done by hand, so we should be grateful and use our inexpensive computers. 

For the example of Requirement 1, let's sample say, 50 datasets, and compute $p(O \vert E_{bame}, B_b)-p(O \vert E_{non-bame}, B_b)$, for each business unit. `probability4` does it for one sample dataset.

source
```python
def probability4(data):
    """ p(promotion | E_bame, B_b) - p(promotion | E_non, B_b) 
        return
            (pd.Series) The difference in promotion rates for b2b and consumer
    """
    return (
        probability2(data)
        .rename("promotion")
        .reset_index()
        .pivot(
            index="business_unit",
            columns="ethnic_category",
            values="promotion"
        )
        .sort_index()
        .assign(difference=lambda df: df["BAME"]-df["non-BAME"])
        .loc[:, "difference"]
    )
```

Repeating this 50 times and looking at the results confirms that the chances of promotion by business unit are the same for each ethnic category. The sample distributions are clearly centered around 0. If still in doubt, just draw more samples!

{% include mediation/mediation_analysis41.html %}

### References
- TBOW
- Pearl's paper on mediation



