# Documentation - Machine Learning in Practice - Group "The Learneres"

In our project, we focused on the task of refining the pipeline of a machine learning model that learns to predict virality of tweets. The training data set consisted of a pre-given subset of tweets limited mainly to the topics of data science, data analysis and data visualization.
In the following lines, we will outline our design decisions, such as the choice of metrics, features, classifiers, parameters, following their performance and interpretations. Our additional improvements considered preprocessing, feature extraction, classification and evaluation, some analysis of the extracted features was also carried out. 

## Evaluation

### Additional Evaluation Metrics

#### F1 score
The measure is defined as the harmonic mean between the precision (how many of the positively classified cases are indeed positive?) and recall (how many of the true positive cases is the model able to catch?). Since we are working with an imbalanced dataset which consists of only approximately 10% of positive cases, managing this trade-off in the evaluation is especially important. When predicting the tweets’ virality, we try to make sure that the tweets, which our model classifies as viral, are indeed those with the high potential, but also to be able to recognize as many of all the viral tweets as possible.

#### Average Precision
Average Precision score summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight. In simpler terms, average precision score is the average of precision calculated at various probability values of recall. Precision in itself focuses more on the positive values, so it does not get as impacted by the overly large number of negative samples.

### Additional Baseline Classifier

#### Uniform Random Classifier
This classifier treats both classes the same, meaning that even the underrepresented positive class has equal chance of being assigned. If despite the imbalance in the data, our model performs similar to the uniform classifier, we can safely assume that something went wrong. 

### Results

These are the evaluation scores obtained after a run of the baseline classifiers, recorded scores are on the training (top) and validation (bottom) set:


<TABLE HERE>


### Interpretation

It is to be expected that a majority classifier will do exceptionally well with such an unbalanced dataset as ours. Exactly for that reason, it is wise to also consider metrics that can help us determine the reliability and validity of the measures, such as Cohen’s Kappa and the F1 score. Clearly, predicting one class perfectly while completely disregarding the existence of another is not the goal of a reliable classifier. While a random classifier offers some form of balance for the less represented class, if we run it multiple times on different data splits, we can easily see that the behavior is very unpredictable and the scores mostly low.  It could certainly happen, that a random classifier gets many instances correct by chance, but since the class distribution does not correspond to reality, the scores will never get over a certain threshold, no matter how many times we try or even if we multiply the sample to obtain more training data. The frequency distribution can therefore be understood as a middle ground between the majority and the randomness factors. In theory, it is possible to achieve a perfect score, this is however extremely unlikely. Instead, we see that it can prove to be even less reliable than would be expected by chance, as indicated by the negative Cohen’s Kappa for the validation set. 
  
Overall, we can conclude that although the imbalance in the data set may provide some protection against getting good results purely with randomness, we need to keep the imbalance in mind and focus on evening out the representations by other means. We didn’t have the resources to create more positive examples and decided against cutting down the underrepresented class, since our dataset is not overly big. Instead, we tried a number of parameter combinations with our classifiers, which also target to tackle the class imbalances.

## Preprocessing

### Design Decisions

As with just about any natural language data, the raw input tells the model relatively very little. For our purposes and the features that we decided to implement, we first needed to rid the text of the irrelevant features, such as hashtags attached to the tweets, mentions, url links and other typically non-linguistic features. Additionally, we plugged the cleaned tweets into the pre-trained sentiment analyzing tool form the nltk library - [VADER](https://www.nltk.org/_modules/nltk/sentiment/vader.html "Source Code of the VADER tool") (Valence Aware Dictionary for Sentiment Reasoning), in order to gain insights into the polarity and emotional intensity of the tweets. The algorithm returns a dictionary of four values – negative, neutral, positive and compound, with the first three representing ratios of proportions of the corresponding sentiment, thus summing to 1 (the closer to 1, the more dominant is the given sentiment) and the latter provides a normalized weighted score ranging from -1 to 1 (the closer to 1, the more positive the sentiment), representing a balanced value calculated from the other three components.

### Results

#### Text cleaner
  
<TABLE HERE>
  

#### Interpretation of the Text Cleaner output
Admittedly, our current method for extracting the plain text could be further perfected, for example by retaining some of the mentions or hashtags, especially those that also form semantic part of the context in the tweet, or certain punctuation which is occasionally used as regular part of text, such as a colon or a dash. However, as these can equally serve as misleading indicators for our other features, including the sentiment analyser or average character length, we decided to nevertheless exclude them from the clean output completely.

#### Sentiment analyzer

<TABLE HERE>
  
#### Interpretation of the Sentiment Scores
It is highly unlikely that a tweet would obtain a high score on both positive and negative sentiment, hence the simple compound score should be sufficient and also most useful for the overall evaluation, but it is possible that one of the sentimental types has higher informative value when it comes to predicting the popularity of tweets. For that reason we decided to access all four scores at this stage, so that we can have a look at the different data distributions and then make an informed decision about which one(s) to take.

## Feature Extraction

In total, we have decided to implement five features. The following sections will outline the main decisions and explain our reasoning.

#### Average tweet length
The hypothesis behind the creation of this feature is that the length of a tweet can be a factor in deciding its virality. It is created on the basis of the character length. It considers the number of characters in a tweet and compares it with the average length of tweets in the complete dataset. That way we establish a baseline that is typical for the subset of tweets that consider our task at hand. In case the length of character for that particular tweet is more than the average length of a tweet, this flag is tagged 1, else 0. Below is a short demonstrative example, assuming the average length of 76:
  
<TABLE HERE>
  
  
<IMAGE HERE> ....adjust the graph to show proportions better?
*Graph 1: Average tweet length flag, distributions between the non-viral (False) and viral (True) class.*
  
#### Hashtag count
Hashtags are a popular way for users on social media to tag their post to a specific topic. Likewise, it allows users to follow the posts on their topics of interest, including also of users outside one’s network. This leads us to think that the more hashtags a post contains, the more views it is likely to hit and with that, the chance of a tweet being liked or shared also increases.

The original dataset already contained the pre-extracted hashtags, which were organised in a list. Our feature therefore makes use of the count of elements in the given list for each tweet.
  
According to the plotted data distributions below, we can observe some differences between the classes, although the majority of tweets in both classes actually do not use hashtags, so it is clearly not a hard prerequisite. We can probably safely say that the mere increased exposure to content will not guarantee its virality, but in combination with other properties, it can certainly boost its chances of becoming noticed.

<IMAGE HERE>
*Graph 2: Hashtag count distributions between the non-viral (False) and viral (True) class.*
  
#### Mentions count
Similar to hashtags, mentions are also a way to access content from users outside one’s direct following, which in turn increases the exposure of the tweet to wider audiences. They are more social in character, which can theoretically bring more common ground to the author and the viewers – they have at least this one person in common. As such, mentions may have a lot of potential for analysis of popular tweets. 
  
Lists of mentions were also already available in the given data set, where each item consisted of a dictionary that codes the screen name, name and id of the users. For our purposes, however, we once again only counted the number of items - in this case dictionaries, within the given list per tweet.

As it turns out, in our sample, the use of the mentions feature is not as popular as expected (see Graph 3). It can be explained by the nature of our dataset, which is a rather specific subset of tweets expected to be more scientific or educational in nature, rather than social or entertaining. Nevertheless, we considered the feature to be relevant for the general task of deciding the potential virality of a tweet and hypothesized that a different data set would likely benefit from this feature in our model, so we included it anyway.

<IMAGE HERE>
*Graph 3: Mentions count distributions between the non-viral (False) and viral (True) class.*
  

#### Media
The number of characters allowed in a single tweet is limited to 280 unicode characters ([source](https://developer.twitter.com/en/docs/counting-characters)). It is meant to encourage users to be concise and come straight to the point in their messages, however, it can also pose a challenge. According to the common saying “a picture speaks a thousand words”, we think that people often resonate more with visual or multisensory messages rather than simple text. For that reason we also decided to map the presence of media, including photos, videos and urls against purely textual content. The feature is represented as a Boolean, stating either that there has or has not been media content identified in the tweet. 
  
The relative distribution of viral and non-viral tweets is similar around 85% among both types of tweets (Graph 4), with and without media. Nevertheless, the inclusion of media does seem to show a slight increase in the probability of the tweet to be flagged as viral.

<IMAGE HERE>
*Graph 4: Percentual distribution of viral and non-viral tweets among those with and without media.*

#### Sentiment score
Before extracting the sentiment feature, we first analyzed the potential of each score (ie. negative, neutral, positive or compound) to stir our classifier in the right direction. After examining the respective distributions of the four scores between our classes (Graphs 5-8), we concluded that the compound scores best capture some signs of divergence between the viral and the non-viral labels, especially in the mildly positive and mildly negative values, so we decided to only implement this score as our sentiment feature. The feature consists of simply extracting the appropriate value for each tweet. As mentioned above, the ‘compound’ scores range from -1 to 1, with the more positive values associating with more intense positive sentiments and vice versa.

<IMAGE HERE>
*Graph 5: Negative sentiment score distributions between the non-viral (False) and viral (True) class.*

<IMAGE HERE>
*Graph 6: Neutral sentiment score distributions between the non-viral (False) and viral (True) class.*

<IMAGE HERE>
*Graph 7: Positive sentiment score distributions between the non-viral (False) and viral (True) class.*

<IMAGE HERE>
*Graph 8: Compound sentiment score distributions between the non-viral (False) and viral (True) class.*

### Interpretation

At the first glance, it doesn’t seem that any feature would have an obvious direct impact on the final verdict of virality, since in each case, it is only a minority of the sample that shows a trend towards some kind of a pattern. However, it is more likely that certain combinations, rather than a single feature, will have more significance for the classification into one or the other class.

## Dimensionality Reduction

Since we only implemented a handful of features, neither of which is highly dimensional, we decided against using any dimensionality reduction techniques. Nevertheless, we have applied the mutual information filter method in order to obtain the scores for mutual information between our individual features and the target variable. The method had already been implemented in the base code, so we shall omit the design rationale and will only briefly discuss our results. In the end, our k for the k best selector is equal to the number of our features.

### Results and Interpretation

The mutual information scores for our scores and the virality class were as follows (rounded to the decimal places):
  
<TABLE HERE>

According to these scores, we can order our features based on the corresponding information gain: 

media >  character length >  hashtag count >  sentiment score >  mentions > average character length flag

The winning position of the media feature did not come as a surprise for us. Sharing graphics can be quite characteristic for tweets regarding the topics of data science, as well as external links to useful information and sources. Likewise, entertaining memes or videos are probably very common among the most popular tweets in general and we imagine that some portion of the viral tweets in our dataset also falls into this category. In any case, it should be noted that the scores are generally rather low, so our expectations for the classifier performance are not overly optimistic. Since the pre-coded feature of plain character length seems to be more informative than our implemented variation, we decided to also count it into our final feature space.

## Classification

### Classifiers

#### Logistic Regression
Logistic Regression Classifier is a suitable method to explore the association of the independent variables with a dichotomous target class, which serves our current needs quite well. Having initialized the classifier with its penalty parameter set to default, we analyzed the combinations of various *solvers* and the values of *C* (Inverse of Regularization strength) via grid search. We also analyzed the *class_weight*, as the distribution of viral versus non-viral tweets is highly skewed. With ‘None’ equal weights are given to both the classes but in case of ‘balanced’, more weight is given to the less represented class.

#### Random Forest
Random Forest Classifier combines the systematic approach of a single decision tree with robustness of a large number of trees and an element of randomness. Additionally, it provides a lot of flexibility in terms of the implementation. We picked five parameters to explore in more detail: *n_estimators* specifies the number of trees, ie. voters in the forest, we experimented with the values ranging from 50 to 500: *criterion* determines the split quality at the tree formation based on either gini impurity or entropy, we applied both: *max_depth* can be used to limit the expansion of the trees to prevent overfitting, we tried a number of values ranging between 10-200: *bootstrap* parameter determines whether a subset or the whole dataset shall be used for the trees, and finally *class_weight* allows us to balance the weight that is given to the underrepresented class in the dataset.

### Results

The big finale begins: What are the evaluation results you obtained with your
classifiers in the different setups? Do you overfit or underfit? For the best
selected setup: How well does it generalize to the test set?

### Interpretation

Which hyperparameter settings are how important for the results?
How good are we? Can this be used in practice or are we still too bad?
Anything else we may have learned?
