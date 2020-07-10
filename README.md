# Azure-ML-Training

## Contents

1 [Intro to Machine Learning](#10-intro-to-machine-learning)
2 [Defining and Differentiation Machine Learning](#11-defining-and-differentiating-machine-learning)

# 1.0 Intro to Machine Learning


The chapter covers the followign items in the subtopics:

  * The definition of Machine learning and How its different from Traditional Programing
  * Applications of Machine learning
  * The historical context of machine learning
The data science process
The types of data that machine learning deals with
The two main perspectives in ML: the statistical perspective and the computer science perspective
The essential tools needed for designing and training machine learning models
The basics of Azure ML
The distinction between models and algorithms
The basics of a linear regression model
The distinction between parametric vs. non-parametric functions
The distinction between classical machine learning vs. deep learning
The main approaches to machine learning
The trade-offs that come up when making decisions about how to design and training machine learning models

## 1.1 Defining and Differentiating Machine learning

The Definition:

>Machine learning is a data science technique used to extract patterns from data, allowing computers to identify related data, and forecast future outcomes, >behaviors, and trends.

  Data science deals with identifying patterns in data and using it to make predictions or map relations between data points; Machine learning is a technique used by data scientists to forecast future trends using existing data and outcomes implimenting various algorithms which learn and identify data patterns among the give data points.

The following image depicts data science and its multiple desciplines:

![Datascience - Multidisciplinary Image](https://www.simplilearn.com/ice9/free_resources_article_thumb/Data-Science-vs.-Data-Analytics-vs.-Machine-Learning1.jpg)

![Here is a GIF! that highlights various fields of data science and inter-relates it with the timeline of business applications:](https://365datascience.com/wp-content/uploads/Euler-Venn_720p.gif)

## 1.3 History of Machine learning

Artificial intelligence started in 1950s which is all about writing algorithms that mimic human thought process. Machine learning then came to write programs which identify data patterns without explicitly being programmed. Deep learning with the discovery of neural nets is the breakthrough that drove the AI boom and also complex machine learning applications such as language processing and image classification.

Many hidden layers and hidden nodes made the field to be called as deep learning.

![Time line of machine learning](https://blogs.nvidia.com/wp-content/uploads/2016/07/Deep_Learning_Icons_R5_PNG.jpg-672x427.png.webp)

## 1.5 The Data science process

1. Collect Data - From various sources such as, but not limited to:
  * Files
  * Databases
  * Webservices
  * Scraping/Crawling webpages
  
2. Preparing Data - Data cleansing (removing undesirable values and baised data points from the data set) and Data visualistation (visualing data using plots/graphs)

3. Training Model - setting up model pipeline by feature vectorization, feature scaling and tuning machine learning algorithm. evaluating mode performance thorugh variance evaluation matrxi to understand and evaluate the training cycles of the model.

4. Evaluate Model - Testing and comparing the performance of multiple trained versions of the model with the test data.

5. Model Deployment - As a part of Devops which **integrate training, evaluation and deployment scripts for respective builds and realease pipelines**.They make sure all the versions of the model deployments are versioned and artifacts are archived.

6. Retraining - Based on the business need, we might have to re-train our machine learning models going through the processes of Training-Evaluation-Deployment for the new re-trained version.


## 1.6 Types of Data

The form and structure of the data plays a crucial role in deciding the machine learning algorithm to use, values of hyper parameters and problem solving methodology.

> Its all numerical at the end!

* **Numerical Data -** Data that is in the numerical form or that has been converted from other data forms into numerical such as speech or image data coverted into numerical data points.

* **Time-Series Data -** Numerical values that can be ordered. Typically data collected over equally spaced points in time, and can also include data that can be ordered with a non date-time column.
  * **Examples of non-date time column time series data:** Real-time stock performance, demand forecasting and speech data that will be translated into a time-based frequency values.
  
* **Categorical Data -** Discrete and limited set of values which doesnot add any sense of value to the data unless it is categorized/grouped together.

* **Text -** Transforming words and texts into numbers understandable by machine learning algorithms is a challange by itself.

* **Image -** Transforming Image into numeric form is a challange similar to the text for developing machine learning algorithms.

## 1.7 Tabular data

The most common type of data available for most of the machine learning problems is the tabular data.

Defining the elements of tabular data:

**Row:** Item/entity in a table.

**Column:** Property of the item/entitiy. They can be continous or descrete(categorical) data.

**Cell:** A single component of the table describing an item in the x-direction and its property in the y-direction.

> **Vectors-**
> A vector is simply an **array of numbers**, such as (1, 2, 3) or a nested array that contains other arrays of numbers, such as (1, 2, (1, 2, 3)).


