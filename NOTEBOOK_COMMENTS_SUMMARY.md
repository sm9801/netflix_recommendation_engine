# Netflix Recommendation Engine - Notebook Comments Summary

This document contains all module-level comments (encased in `"""` `"""`) from the Results.ipynb notebook, organized by cell and section.

---

## Cell 2: Initial K-means and GMM Comparison

### Comment:
```
We will first compare the performance of K-means and GMM models at varying K values.
Optimal seed values will be chosen based on minimum cost (negative log likelihood for GMM, 
sum of squared distances for K-means).
```

---

## Cell 3: Model Performance Analysis

### Comment 1:
```
Strictly from results of RMSE, K-means and GMM perform similarly. However, based on BIC values, 
GMM outperforms K-means consistently across all K values. 
This suggests that GMM is a better model for this dataset, likely due to its ability to capture 
more complex data distributions through its probabilistic framework.
Let's perform EDA to determine whether our hypothesis is correct.
```

### Comment 2:
```
Elbow Method for Optimal K Selection of K-means Clustering
```

---

## Cell 4: K-means Clustering Setup with t-SNE

### Comment:
```
Based on the above elbow method and silhouette score, we choose K=3 for K-means clustering.
We hypothesized that the data has multiple underlying user distributions with varying preferences 
and behaviors.
Thus, we employ a manifold visualization technique, t-SNE, to visualize the high-dimensional data 
in a 2D space.
This will help us observe the clustering tendencies and validate our hypothesis about the data's 
structure.
To optimize t-SNE performance, we will experiment with different perplexity values ranging from 
30 to 50 and monitor the KL divergence for each setting.
```

---

## Cell 6: t-SNE Results Analysis for K-means

### Comment:
```
From the t-SNE visualizations across different perplexity values, we observe that the data points 
do not form clear clusters. 
This supports our hypothesis of multiple underlying user distributions that are not captured by 
Kmeans.
The clusters remain relatively consistent across varying perplexity settings, indicating stable 
clustering tendencies in the data.
Based on the KL divergence plot, we notice that lower perplexity values tend to yield lower KL 
divergence, suggesting better preservation of local structures in the data.

A perplexity value around 50 seems to strike a good balance, capturing both local and global 
structures effectively.
Numerically, the KL divergence value is also lowest at perplexity 50, indicating that this setting 
provides a more faithful representation of the high-dimensional data in the 2D t-SNE space.
Overall, these visualizations validate our hypothesis and highlight the presence of distinct user 
clusters in the dataset.
Let's now analyze RMSE and BIC values for our chosen K value of 3.
```

---

## Cell 8: K-means Across Different K Values

### Comment:
```
As expected, KMeans do not produce the most ideal callable clusters since it assumes spherical 
clusters and uses hard assignments.
This indicates that the underlying data distribution may be more complex.
We will now proceed how K-means changes per K value.
```

### Nested Comment (inside loop):
```
Choose the best seed for each K based on minimum cost
```

---

## Cell 9: K-means Analysis Summary

### Comment:
```
The above plots show how K-means clusters evolve as K increases.
Although our K-means distortion scores indicated K = 3 as optimal, none of the K values produce 
well-separated clusters.
At best, K=3 seems to provide the most meaningful clustering structure.
However, within the main cluster segments, we can observe sub-clusters forming as K increases, 
indicating more granular user groupings.
This suggests that while K=3 captures the broad user segments, higher K values reveal finer 
distinctions among users.
We want to identify these underlying distributions more effectively, because there are a lot of 
overlap between the sub-clusters.
We will now apply Gaussian Mixture Models (GMM) using the EM algorithm.

We choose the Gaussian Mixture Model (GMM) because it allows for soft assignments of data points 
to clusters, 
meaning each user can belong to multiple clusters with varying degrees of membership. 
GMM also applies soft assignment of data points to clusters based on probability distributions, 
which is more realistic for user behavior data where preferences can overlap across different 
user segments.
This flexibility enables GMM to capture the underlying data distribution more effectively than 
K-means, which assumes hard assignments and spherical clusters.
```

---

## Cell 10: GMM with t-SNE Optimization

### Comment:
```
As before, we will now perform the same EDA for GMM clustering with K=12 and optimize for 
perplexity values ranging from 30 to 50.
```

---

## Cell 12: GMM t-SNE Results Analysis

### Comment:
```
From the t-SNE visualizations for GMM clustering, we observe more distinct and well-separated 
clusters compared to K-means.
Based on the KL divergence plot, we choose a perplexity = 50 for optimal balance.
This setting captures both local and global structures effectively, as indicated both by the 
lowest KL divergence value and our manifold visualization.
```

---

## Cell 14: GMM Performance Analysis

### Comment:
```
As expected, GMM produces more callable clusters with less overlap compared to K-means.
This indicates that GMM is better suited for capturing the complex underlying distributions in 
user behavior data.
As before, we will now proceed how GMM changes per K value.
```

### Nested Comment (inside loop):
```
We use a Gaussian Mixture Model with EM to jointly perform user clustering and matrix completion.
To visualize high-dimensional latent structure, we project completed user embeddings into 2D 
using t-SNE and color by soft GMM assignments.

The GMM is trained in the original high-dimensional space.
t-SNE is used purely as a visualization tool to preserve local neighborhood structure so that soft 
cluster assignments can be interpreted qualitatively.

For our initialization, we choose the best seed for each K based on maximum log likelihood.
```

---

## Cell 16: RMSE and BIC Comparison

### Comment:
```
Let's now analyze RMSE and BIC values across different K values for GMM and compare with K-means.
```

---

## Cell 19: Model Selection Justification

### Comment:
```
Based on these results, we observe that the RMSE values for both K-means and GMM remain 
consistent and similar.
However, GMM consistently outperforms K-means in terms of BIC across all K values by a large 
margin.
This causes a real concern, where K-means may not be capturing the underlying data distribution 
effectively, yet it still successfully achieves a comparable RMSE.
In fact, this indicates an issue with K-means, where it may be overfitting to the data and 
failing to generalize the data distribution effectively.

We will now analyze the uncertainty in cluster assignments by computing the posterior entropy 
for GMM and comparing cluster assignments between K-means and GMM.
Based on BIC, our best K for GMM is 16 with seed = 1. 

Since global minimum of BIC and RMSE corresponds to K = 3 for GMM, we choose K = 3 and 
seed = 3 for GMM.

HOWEVER, we also choose K = 16 and seed = 1 for GMM, despite its BIC being a local minimum 
because the global minimum of BIC value corresponds to the smallest K value.
BIC penalizes model complexity, often favoring simpler models with fewer clusters, yet K = 16 
has the second lowest BIC value, despite K = 16 being far more complex.
This indicates that the large likelihood gains from increasing K to 16, large enough to outweigh 
the complexity penalty imposed by BIC.
Therefore, we believe that K = 16 is more meaningful, and this choice is justified as K = 16 
captures more nuanced user segments while still maintaining a low BIC.

Based on distortion scores and silhouette scores, our best K for K-means is 3 with seed = 0.
```

---

## Cell 20: Entropy and Agreement Analysis

### Comment:
```
Entropy and agreement were computed using model-specific optimal K values (GMM at K = 3 and 
K = 16 selected via BIC).

While GMM model exhibits low entropy, indicating confident assignments, agreement between 
K-means and GMM at K = 3 is approximately 57%.
Agreement between K-means and GMM at K = 16 is almost nonexistent at around 0.9%. 
This reflects fundamental differences in clustering objectives and model assumptions rather than 
instability.

The main cause for differences in cluster assignments is the different cluster settings for 
K-means and GMM (K = 3 vs K = 16).
Furthermore, earlier speculation about the K-means model overfitting and failing to generalize 
likely stems from the cleanliness of the training data.
Despite the apparent inability to generalize, K-means still achieves comparable RMSE to GMM due 
to the sparsity of noise and missing values in the training data.

On this dataset, K-means and GMM converge to similar solutions due to well-separated clusters 
and low posterior uncertainty.
However, GMM provides calibrated uncertainty estimates and posterior-weighted matrix completion, 
which becomes critical in noisier or more ambiguous regimes.

To validate these findings, we will perform a series of stress tests on the two models by 
introducing varying levels of noise and creating missing data in the dataset.
Additionally, we will also force cluster overlap to observe how each model copes under 
challenging conditions.
```

### Nested Comment 1 (after first comment):
```
Stress Test 1: Varying Missing Data Rates
    We will systematically force missing values into the dataset at different rates 
    (10%, 30%, 50%, 70%) 
    and evaluate the performance of both K-means and GMM in terms of RMSE, BIC, posterior 
    entropy, and cluster agreement.
    This will help us understand how each model handles incomplete data and their robustness 
    to missing data.
    Furthermore, this will show how GMM generalizes the underlying user distributions that are 
    missed by K-means.
```

---

## Cell 24: Stress Test 1 Results Analysis

### Comment:
```
As hypothesized, as the missing data rate increases, the performance of K-means degrades 
significantly in terms of RMSE and BIC.

GMM model also experiences performance degradation, but to a much lesser extent.
RMSE values remain robust and more consistent. BIC values for GMM remain significantly better 
than K-means across all missing data rates, with little degradation.
Entropy, although monotonically increasing, remains near zero, indicating confident cluster 
assignments even with substantial missing data.
Cluster agreement between K-means and GMM is effectively nonexistent, reflecting growing 
divergence in clustering solutions under challenging conditions.

Notice that K = 3 for GMM slightly outperforms K = 16 in RMSE at every missing rate, but K = 16 
achieves slightly better BIC values as missing rate increases. 
This suggests that K = 16 captures more nuanced user segments while still maintaining a low 
BIC, which is consistent with our earlier observations.

Agreement levels are relatively high between K-means and GMM at K = 3, but drop significantly 
for K = 16.
This reflects that with more clusters, GMM captures finer distinctions among users that K-means 
fails to identify, especially as data becomes sparser.

This stress test validates our earlier findings that GMM is more robust and better suited for 
capturing the underlying data distribution, especially in the presence of missing data.
We will now visualize these results.
```

---

## Cell 26: BIC Comparison for Missing Rates

### Comment:
```
BIC comparison between K = 3 and K = 16 for GMM are difficult to visualize above, so we plot 
them separately below.
```

---

## Cell 27: t-SNE Visualization for Missing Rates

### Comment:
```
The above plots indicate the consistency in RMSE and BIC values for GMM, even with increasing 
missing data values, 
confirming our hypothesis that our GMM model generalizes the data better than K-means.
We now explore changes in data distribution at different missing rates using t-SNE plots.
```

---

## Cell 28: Stress Test 1 Conclusions

### Comment:
```
From the t-SNE visualizations under varying missing data rates, we observe that K-means 
clusters become increasingly diffuse and overlapping as the missing rate increases.
This indicates that K-means struggles to maintain distinct cluster boundaries in the presence 
of substantial missing data.

In contrast, GMM clusters remain relatively well-defined and separated even at high missing 
rates.
As expected, GMM at K = 16 shows more nuanced cluster structure compared to K = 3, capturing 
finer distinctions among users.

This highlights GMM's robustness and ability to capture the underlying data distribution more 
effectively, despite significant missing information.
Overall, these visualizations corroborate our earlier quantitative findings, demonstrating 
GMM's superior performance and resilience to missing data compared to K-means.

We will now proceed to Stress Test 2: Varying Noise Levels.
```

---

## Cell 29: Stress Test 2 Introduction

### Comment:
```
Stress Test 2: Varying Noise Levels
    We will introduce noise into the dataset at different severity levels
    and evaluate the performance of both K-means and GMM in terms of RMSE, BIC, posterior 
    entropy, and cluster agreement.
    This will help us understand how each model handles noisy data and their robustness to noise.

    We will test five types of noise at varying levels. We first test the baseline realistic 
    noise, where random Gaussian noise is added to all observed entries.
    Next, we will test moderate stress noise with item correlation, strong stress noise with 
    sparse outliers, severe stress noise with heavy tails, and extreme adversarial flips.
    This will help us understand how each model copes under different noise conditions.
    We aim to fully test the robustness of both models under varying noise conditions.
```

---

## Cell 30: Stress Test 2 Execution

### Comment:
```
We now proceed with Stress Test 2: Varying Noise Levels
This comprehensive test evaluates K-means and GMM robustness to three types of noise.
```

---

## Summary of Analysis Flow

1. **Initial Comparison**: Compare K-means vs GMM on clean data
2. **K-means Analysis**: Examine K-means across different K values using t-SNE
3. **GMM Analysis**: Examine GMM across different K values and justify K=16 selection
4. **Model Selection**: Justify choice of K=3 for K-means, K=3 and K=16 for GMM
5. **Stress Test 1**: Test robustness to varying missing data rates (10%-70%)
6. **Stress Test 2**: Test robustness to five types of noise at varying severity levels

### Key Findings Highlighted:
- GMM outperforms K-means on BIC across all configurations
- K-means clusters become diffuse with missing data; GMM remains stable
- GMM provides better uncertainty estimates through posterior entropy
- K=16 for GMM captures more nuanced user segments while maintaining low BIC
- GMM demonstrates superior robustness to both missing data and noise
