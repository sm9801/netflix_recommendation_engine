# Netflix Recommendation Engine

This project is a machine learning-powered recommendation system designed to suggest movies, shows, and other programs to users based on their previous viewing preferences and ratings on Netflix. Using a combination of collaborative filtering techniques and clustering, the system predicts user preferences and tailors recommendations accordingly.

Full results and analysis can be found here: [Netflix Analysis](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Results%20copy.ipynb)

Utilized Tech Stack: Python, NumPy, Pandas, scikit-learn, Seaborn, Matplotlib, Yellowbrick, Cleanlab

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset and Methodology](#dataset-and-methodology)
- [Approach](#approach)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Project Overview
This Netflix recommendation system uses collaborative filtering to generate tailored suggestions by predicting ratings for shows that users haven’t yet watched. By clustering similar users and using probabilistic modeling, the system uncovers and aligns with user preferences, enhancing user engagement with personalized recommendations.

- Problem Overview
  - Recommendation plays a vital role in customer retention on streaming platforms such as Netflix, because recommendation systems feed content that consumers might enjoy, and therefore, prolongs customer retention rates. As such, a proficient recommendation system has a correlating factor in determining sales of Netflix subscriptions. Without it, or with a lackluster version of it, users face negative experience while using the streaming platform which may increase the rate of customer retention.
  - To create a production ready recommendation system, it is imperative to first understand the underlying distribution of user preferences. By modeling user behavior, we can then assign probabilities to contents to determine whether a user may enjoy a content or not.
  - We aim to create the best model for such recommendations by first determining the structure of user data given our dataset. Afterwards, we will define and test algorithms for predicting user ratings for recommendations.

## Dataset and Methodology

The dataset contains information about user ratings. It is a matrix of 1200 * 1200 user rating data and missing data is represented as 0.
To assign ratings for the missing data, we implemented the K-means and Gaussian Mixture Models (GMM) for clustering, and the Expectation-Maximization (EM) algorithm for modeling and rating predictions.

K-means was selected as an initial baseline due to its computational efficiency, simplicity, and interpretability. It performs well on large datasets and provides clear, hard cluster assignments, making it suitable for quickly identifying dominant user segments based on rating behavior. This allowed us to establish a strong baseline for user segmentation and evaluate cluster structure using metrics such as RMSE and silhouette behavior.

However, user preferences in a real-world streaming platform like Netflix are rarely exclusive. Users often belong to multiple preference groups, which hard clustering cannot fully capture. To address this limitation, Gaussian Mixture Models were introduced. GMMs model each cluster as a probability distribution and assign users soft membership probabilities, allowing the recommendation system to reflect uncertainty and overlapping interests. This probabilistic formulation better aligns with real viewing behavior and enables more nuanced recommendations.

After data points are assigned cluster probabilities via GMM, ratings will be predicted via the Expectation-Maximization (EM) algorithm. In this recommendation system, the EM algorithm helps assign users to probabilistic clusters, capturing similarities in their preferences and improving rating predictions.

1. **E-Step (Expectation Step)**: Calculate the expected value of the latent variables based on the observed data and the current parameter estimates.

   Given data  $X$ and current parameters $\theta$, compute the responsibilities (posterior probabilities) for each cluster $k$ :
   
   $\gamma_{z_i} = P(z_i | X; \theta) = \frac{\pi_k \, \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i | \mu_j, \Sigma_j)}$
   
   where:
   - $\gamma_{z_i}$ is the responsibility of cluster \( k \) for data point $x_i$,
   - $\pi_k$ is the prior probability of cluster $k$,
   - $\mathcal{N}(x_i | \mu_k, \Sigma_k)$ is the Gaussian distribution for cluster $k$ with mean $\mu_k$ and covariance $\Sigma_k$.

3. **M-Step (Maximization Step)**: Update the parameters $\theta = (\pi_k, \mu_k, \Sigma_k)$ by maximizing the expected log-likelihood from the E-step.

   The parameters are updated as follows:
   - **Update the mean**:
     
     $\mu_k = \frac{\sum_{i=1}^{N} \gamma_{z_i} \, x_i}{\sum_{i=1}^{N} \gamma_{z_i}}$
     
   - **Update the covariance**:
     
     $\Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{z_i} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{z_i}}$
     
   - **Update the mixture weights**:
     
     $\pi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma_{z_i}$
     
   where $N$ is the total number of data points, and $K$ is the number of clusters.

These E and M steps repeat until convergence, yielding optimized parameters $\theta$ that enhance the model’s accuracy in predicting user ratings.

- Performance Comparison: K-Means vs GMM

| K | K-Means RMSE | K-Means BIC | K-Means LL | GMM RMSE | GMM BIC | GMM LL |
|---|---|---|---|---|---|---|
| 1 | 0.5667 | -2,821,274 | -2,817,016 | 0.4802 | -1,544,758 | -1,540,500 |
| 2 | 0.4805 | -2,707,332 | -2,698,814 | 0.4559 | -1,464,771 | -1,456,252 |
| 3 | 0.4846 | -2,676,802 | -2,664,022 | 0.4566 | -1,468,236 | -1,455,457 |
| 4 | 0.4746 | -2,653,928 | -2,636,887 | 0.4756 | -1,447,234 | -1,430,193 |
| 5 | 0.4873 | -2,642,527 | -2,621,225 | 0.4738 | -1,450,889 | -1,429,587 |
| 6 | 0.4961 | -2,636,149 | -2,610,585 | 0.4725 | -1,453,663 | -1,428,099 |
| 7 | 0.4940 | -2,631,491 | -2,601,667 | 0.4759 | -1,453,335 | -1,423,511 |
| 8 | 0.5025 | -2,626,356 | -2,592,271 | 0.4920 | -1,445,582 | -1,411,497 |
| 9 | 0.4959 | -2,622,983 | -2,584,636 | 0.4967 | -1,449,051 | -1,410,704 |
| 10 | 0.4800 | -2,619,471 | -2,576,864 | 0.4974 | -1,452,497 | -1,409,889 |
| 11 | 0.4873 | -2,618,841 | -2,571,972 | 0.4998 | -1,450,890 | -1,404,022 |
| 12 | 0.4797 | -2,618,970 | -2,567,840 | 0.5020 | -1,454,027 | -1,402,897 |

- Based on user ratings, GMM performs slightly better than K-means, indicating similar performance given the current dataset. However, the gains from GMM are minor and given the current results, both models are suitable for selection. The results hint that GMM generalizes user data better and may perform better in complex data settings; however, these claims need to be substantiated through evidence.

### Hypothesis
- We expect K-means and GMM to perform similarly given the current dataset, but as underlying distribution becomes more complex and as complications are introduced, GMM will generalize better than K-means and will be reflected via test results.
- To test our hypothesis, after EDA is performed and parameters are optimized, we will put our GMM and K-means model through 3 stress tests:
  - Stress Test 1: Missing data at varying rates [0.1, 0.3, 0.5, 0.7]
  - Stress Test 2: Noise introduction [baseline, moderate, strong, adversarial]
  - Stress Test 3: Missing data at 80% and strong noise introduction
- The results of Stress Test 3 will be used to compute percentage improvement of the optimal GMM. 

## Approach

### EDA

To understand the underlying data, we will apply t-SNE plots to both K-means and GMM. To do so, we first need to determine hyperparameter settings for t-SNE plots, namely the perplexity hyperparameter.
We will determine these settings via KL-Divergence computations. For K-means, we also compute distortion and silhouette scores.

| K-means Distortion | K-means Silhouette |
| -- | -- |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20Distortion.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20Silhouette.png) |

| K-means | GMM |
|--|--|
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20Perplexity.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/GMM%20Perplexity.png) |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20KL.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/GMM%20KL.png) |

- Distortion and silhouette scores indicate that K-means performs best at lower K values, despite various user rating distributions present in the data.
- Based on KL-Divergence scores and visualizations of perplexity plots, we choose perplexity = 50 for our t-SNE settings. We now visualize our t-SNE plots for K-means and GMM:

<table>
  <tr>
    <td><strong>K-means</strong></td>
    <td><img src="https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20tSNE.png" /></td>
  </tr>
  <tr>
    <td><strong>GMM</strong></td>
    <td><img src="https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/GMM%20tSNE.png" /></td>
  </tr>
</table>

As we hypothesized, K-means clusters have significant overlap across all K values, where clusters generally clump into a singular blob. This is indicative of K-means inability to generalize underlying data within Netflix user ratings.
On the other hand, GMM clusters have clear and clean separations between each groups. These visualizations indicate that GMM generalizes high dimensional data effectively, as we expected. However, to solidify our findings, we must put both models under our stress tests.

First, we visualize our results for comparison:

| RMSE | GMM BIC | K-means BIC |
| -- | -- | -- |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/RMSE%20plots.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/GMM%20BIC.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Kmeans%20BIC.png) |

### Model Selection

Based on these results, the RMSE values for K-means and GMM are both stable and comparable. Performance alternates across different numbers of clusters, with K-means outperforming GMM in certain ranges and GMM performing better in others.

Whereas both models perform well on this dataset, our manifold analysis suggests that K-means does not generalize effectively to the underlying data distribution. In particular, we observe that K-means BIC values increase monotonically as K increases, indicating deteriorating model quality with additional clusters.
This behavior is notable because it implies that K-means achieves optimal performance at K = 3, consistent with the minimum distortion scores observed earlier. Beyond this point, increasing the number of clusters fails to capture additional meaningful structure and instead degrades model performance.

These findings suggest that the K-means model is limited in its ability to represent nuanced patterns in Netflix user rating distributions. As the number of clusters is forced upward, K-means increasingly overfits local structures while failing to generalize the global manifold, resulting in underperformance.
BIC scores for GMM are worse than K-means across all K values by a large margin. However, at K = 3 and above, GMM shows near monotonic decrease in BIC, where the global minimum appears at K = 20, indicating model improvement and likelihood gains despite BIC penalty. 

However, under current conditions, both K-means and GMM perform well in terms of RMSE. In order to test and compare these models, we will now evaluate them through stress tests.
Based on the above results, we make the following model selection for stress test evaluations:

- **K-means K = 3 (seed = 0):** Best performing K-means settings based on RMSE and BIC, as well as distortion scores
- **GMM K = 3 (seed = 1):** 2nd lowest minimum of BIC. Global minimum for RMSE was achieved by K = 4, but BIC levels also reached the second highest at this point.
Therefore, we test with K = 3 instead of 4.
- **GMM K = 20 (seed = 1):** Global minimum of BIC with large likelihood gains despite BIC penalty of model complexity

## Evaluation
### Stress Test 1 (Missing Data)

For this stress test, we force missing data at varying rates. Namely, we test the models in the following severity:
- [0.1, 0.3, 0.5, 0.7]
- The above missing data rates account for a percentage of observed data, not the entire matrix
- Therefore, total percentage of missing data may be higher than their respective rates after forcing missing data

With the defined model parameters, we have the following results:
| Stress Test 1 Results | Stress Test 1 t-SNE |
| -- | -- |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%201.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%201%20tSNE.png) |

**Summary**
- In terms of RMSE, both GMM models are considerably more robust to missing data compared to K-means. K-means RMSE increase almost exponentially as missing rate increases, while GMM RMSE values increase very slightly.
- BIC values remain ideal for K-means across all missing data rates. However, this is from underfitting due to less data. RMSE performance for K-means is significantly worsened, and as K increases, underfitting increases, resulting in performance decrease. Without enough data, K-means cannot overcome underfitting despite a better BIC score.
- GMM Entropy, although monotonically increasing, remains near zero, indicating confident cluster assignments even with substantial missing data
- Cluster agreement between K-means and GMM at K = 20 is effectively nonexistent, reflecting growing divergence in clustering solutions under challenging conditions, where K-means cannot identify meaningful clusters.

**t-SNE Analysis**
- K-means:
  - Clusters become increasingly diffused and overlapping as the missing rate increases. Eventually, clusters clump into a singular blob.
  - Struggles to maintain distinct cluster boundaries in the presence of substantial missing data
- GMM:
  - Clusters remain well-defined and separated even at high missing rates. Distances between clusters increase as missing rate increases, indicating growing uncertainty between clusters. This is expected, as missing data increases, GMM assigns lower probabilities to each data point for cluster assignment.
  - GMM at K = 20 shows more nuanced cluster structure compared to K = 3, capturing finer distinctions among users.

This stress test validates our earlier findings that GMM is more robust and better suited for capturing the underlying data distribution, especially in the presence of missing data.
We now begin Stress Test 2.

### Stress Test 2 (Noise Levels)

For this stress test, we introduce varying levels of noise and corruption across our data. We test the models in the following severity:
- Baseline (heteroskedastic Gaussian), Moderate (item-correlated noise), Strong (missing not at random), Adversarial (adversarial flips)

With the defined model parameters, we have the following results:
| Stress Test 2 Results | Stress Test 2 t-SNE |
| -- | -- |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%202.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%202%20tSNE.png) |

**Summary**
- GMM at K = 3 maintains lowest RMSE values, while K-means performs slightly better than GMM at K = 20 for initial noise severities
- GMM performs especially better at Strong Noise level in terms of RMSE
- Cluster agreements between GMM at K = 3 and K-means remains between 30 ~ 50% across noise types while for GMM at K = 20 and K-means, cluster agreements range from 0 ~ 10%
- Entropy levels remain lowest for GMM at K = 3, though as noise levels increase, entropy increases as well

**t-SNE Analysis**
- The above visualizations indicate the impact of different noise types on clustering structure. These tests are quite extreme, and both GMM and K-means eventually break down under severe noise conditions. However, K-means break down from the baseline noise levels and worsens as noise severity increases, as indicated by poor cluster separation in t-SNE plots. GMM appears to be more robust in maintaining some clustering structure compared to K-means, especially under adversarial noise conditions, where BIC is the minimum before model failure.
- These results prove the robustness of GMM clustering over K-means given the data structure of Netflix data. However, as observed, GMM is not immune to missing data, and especially not to noise or data corruption. Current GMM model struggles significantly at noise stress tests and further improvement is required to overcome these challenges.

We now perform Stress Test 3.

### Stress Test 3 (Missing data + Sparse / Heavy-Tailed Outliers)
This stress test is designed to simulate real-world data scenarios. Users are very unlikely to rate Netflix programs, and hence the high rate of missing data represents an accurate representation. Furthermore sparse and long tailed outliers represent the most realistic noise data, since only popular contents receive the most ratings. A significant number of Netflix contents receive very few ratings.

With the defined model parameters, we have the following results:
| Stress Test 3 Results | Stress Test 3 t-SNE |
| -- | -- |
| ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%203.png) | ![](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Visualizations/Stress%20Test%203%20tSNE.png) |

## Results
The above results show that GMM is more robust under strenuous conditions of high missing data rate and noise. Findings are similar to our previous tests, with GMM (K=3) outperforming KMeans in RMSE and BIC, while GMM (K = 20) shows higher RMSE and BIC than K-means. Entropy levels are higher for GMM at K = 20 than K = 3, and the poor performance in GMM at K = 20 likely stems from overfitting in cluster assignments. 

Since BIC between the two GMM settings are close, we focus on RMSE and entropy levels for performance comparison, and therefore choose **K = 3** as the preferred model under stress conditions.

We now calculate the overall performance increase of GMM (K = 3) over KMeans in this stress test.

RMSE Improvement = $\frac{RMSE<sub>K-means</sub> - RMSE<sub>GMM</sub>}{RMSE<sub>K-means</sub>} = 29.468950777989367 %

- **Optimal GMM Clusters: 3 (seed = 1)**
- **RMSE: 2.49**
- **RMSE improvement over the baseline: 29.47%**

The achieved RMSE score indicates that GMM at K = 3 accurately predicts user ratings even under real-world data settings, delivering recommendations that align well with user preferences.

## Future Improvements
- **Hybrid Recommendation Approach**: Combine collaborative filtering with content-based filtering to improve recommendation accuracy.
- **Model Improvement**: introduce more sophisticated robustness techniques (especially against noise / data corruption) or apply neural network recommendations