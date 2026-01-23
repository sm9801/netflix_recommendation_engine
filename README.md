# Netflix Recommendation Engine

This project is a machine learning-powered recommendation system designed to suggest movies, shows, and other programs to users based on their previous viewing preferences and ratings on Netflix. Using a combination of collaborative filtering techniques and clustering, the system predicts user preferences and tailors recommendations accordingly.

Full results and analysis can be found here: [Netflix Analysis](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Results%20copy.ipynb)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
  - [Collaborative Filtering with Gaussian Mixture Model (GMM)](#collaborative-filtering-with-gaussian-mixture-model-gmm)
  - [Prediction with Expectation-Maximization (EM) Algorithm](#prediction-with-expectation-maximization-em-algorithm)
  - [K-Means Clustering](#k-means-clustering)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

## Project Overview
This Netflix recommendation system uses collaborative filtering to generate tailored suggestions by predicting ratings for shows that users haven’t yet watched. By clustering similar users and using probabilistic modeling, the system uncovers and aligns with user preferences, enhancing user engagement with personalized recommendations.

- Problem Overview
  - Recommendation plays a vital role in customer retention on streaming platforms such as Netflix, because recommendation systems feed content that consumers might enjoy, and therefore, prolongs customer retention rates. As such, a proficient recommendation system has a correlating factor in determining sales of Netflix subscriptions. Without it, or with a lackluster version of it, users face negative experience while using the streaming platform which may increase the rate of customer retention.
  - To create a production ready recommendation system, it is imperative to first understand the underlying distribution of user preferences. By modeling user behavior, we can then assign probabilities to contents to determine whether a user may enjoy a content or not.
  - We aim to create the best model for such recommendations by first determining the structure of user data given our dataset. Afterwards, we will define and test algorithms for predicting user ratings for recommendations.

## Dataset
The dataset contains information about user ratings. It is a matrix of 1200 * 1200 user rating data and missing data is represented as 0.
To assign ratings for the missing data, we implemented the K-means and Gaussian Mixture Models (GMM) for clustering, and the Expectation-Maximization (EM) algorithm for modeling and rating predictions.

### Model Selection
K-means was selected as an initial baseline due to its computational efficiency, simplicity, and interpretability. It performs well on large datasets and provides clear, hard cluster assignments, making it suitable for quickly identifying dominant user segments based on rating behavior. This allowed us to establish a strong baseline for user segmentation and evaluate cluster structure using metrics such as RMSE and silhouette behavior.

However, user preferences in a real-world streaming platform like Netflix are rarely exclusive. Users often belong to multiple preference groups, which hard clustering cannot fully capture. To address this limitation, Gaussian Mixture Models were introduced. GMMs model each cluster as a probability distribution and assign users soft membership probabilities, allowing the recommendation system to reflect uncertainty and overlapping interests. This probabilistic formulation better aligns with real viewing behavior and enables more nuanced recommendations.

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

### EDA via t-SNE -> 
To understand the underlying data, we will apply t-SNE plots to both K-means and GMM.

### Collaborative Filtering with Gaussian Mixture Model (GMM)
To capture patterns in user behavior and generate personalized recommendations, the project compares K-means and GMM algorithms. GMM’s probabilistic approach enables the system to model complex distributions of user ratings, which will improve the accuracy of predicted ratings.

### Prediction with Expectation-Maximization (EM) Algorithm
The Expectation-Maximization (EM) algorithm is used alongside GMM and K-means for optimizing predictions. EM iteratively refines cluster parameters to maximize the likelihood of the observed data, allowing the model to more precisely assign user preferences and enhance recommendation accuracy.

### Expectation-Maximization (EM) Algorithm

The Expectation-Maximization (EM) algorithm is an iterative method used to estimate parameters in models with latent variables, such as Gaussian Mixture Models (GMMs). In this recommendation system, the EM algorithm helps assign users to probabilistic clusters, capturing similarities in their preferences and improving rating predictions.

The EM algorithm alternates between two main steps until convergence:

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

### K-Means Clustering
K-Means clustering was implemented to segment users into distinct groups based on their viewing preferences. Through experimentation, 3 was identified as the optimal number of clusters, with each cluster representing a unique subset of user tastes. This grouping helps further tailor recommendations by aligning users with others in the same cluster.

## Evaluation
The model’s performance was evaluated using the Root Mean Square Error (RMSE) metric, which compares the predicted ratings with actual user ratings. An RMSE score of 0.48 demonstrates the model’s ability to closely approximate true user preferences.

## Results
- **Optimal GMM Clusters**: 3
- **RMSE**: 0.48

The achieved RMSE score indicates that the model accurately predicts user ratings, allowing it to deliver recommendations that align well with user preferences.

## Future Improvements
- **Hybrid Recommendation Approach**: Combine collaborative filtering with content-based filtering to improve recommendation accuracy.
- **Model Improvement**: introduce more sophisticated robustness techniques or apply neural network recommendations
- **Interactive User Interface**: Create a simple web or desktop interface for users to input ratings and receive recommendations.
- **Real-Time Updates**: Integrate real-time data handling to dynamically update recommendations as users provide new ratings.
