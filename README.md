# CS-6601-Artificial-Intelligence-Assignment-4
CS 6601: Artificial Intelligence Assignment 4


Setup
Clone this repository:

For On-campus students

git clone https://github.gatech.edu/202408A/a4_[YOUR GT GITHUB USER]
And for OMSCS students

git clone https://github.gatech.edu/202408001/a4_[YOUR GT GITHUB USER]
You will be able to use numpy, math, time and collections.Counter for the assignment

You will be able to use the sklearn and graphviz for jupyter notebook visualization only

No other external libraries are allowed for solving this problem. We do check.

Please use the ai_env environment from previous assignments

If you are using an IDE: Ensure that in your preferences, that ai_env is your interpreter and that the assignment_4 directory is in your project structure

Else: From your ai_env Terminal:

conda activate ai_env
The supplementary testing notebooks use jupyter: visualize_tree and unit_testing. From your ai_env Terminal:

jupyter notebook
The supplementary Helper notebook to visualize Decision tree, requires the graphviz library. From your ai_env Terminal:

pip install graphviz==0.17.0
or alternatively
pip install -r requirements.txt
If you have difficulty or errors on graphviz0.17 From your ai_env Terminal:

conda install -c conda-forge python-graphviz 
which installs version 0.19 (compatible)

The previous semester's FAQ can be found under Files

Overview
Machine Learning is a subfield of AI, and Decision Trees are a type of Supervised Machine Learning. In supervised learning an agent will observe sample input and output and learn a function that maps the input to output. The function is the hypothesis ''' y = f(x)'''. To test the hypothesis we give the agent a test set different than the training set. A hypothesis generalizes well if it correctly predicts the y value. If the value is finite, the problem is a classification problem, if it is a real number it is considered a regression problem. When classification problems have exactly two values (+,-) it is Boolean classification. When there are more than two values it is called Multi-class classification. Decision trees are relatively simple but highly successful types of supervised learners. Decision trees take a vector of attribute values as input and return a decision. [Russell, Norvig, AIMA 3rd Ed. Chptr. 18]

Decision Trees

Submission and Due Date
The deliverable for the assignment is to upload a completed submission.py to Gradescope.

All functions must be completed in submission.py for full credit
Important: Submissions to Gradescope are rate limited for this assignment. You can submit two submissions every 60 minutes during the duration of the assignment.

Since we want to see you innovate and imagine new ways to do this, we know this can also cause you to fail (spectacularly in my case) For that reason you will be able to select your strongest submission to Gradescope. In your Gradescope submission history, you will be able to mark your best submission as 'Active'. This is a students responsibility and not faculty.

The Files
You are only required to edit and submit submission.py, but there are a number of important files:

submission.py: Where you will build your decision tree, confusion matrix, performance metrics, forests, and do the vectorization warm up.
decision_trees_submission_tests.py: Sample tests to validate your trees, learning, and vectorization locally.
visualize_tree.ipnb: Helper Notebook to help you understand decision trees of various sizes and complexity
unit_testing.ipynb: Helper Notebook to run through tests sequentially along with the readme
Resources
Canvas Thad's Videos: Lesson 7, Machine Learning
Textbook:Artificial Intelligence Modern Approach (3rd Ed. Chapters may have changed for 4th Ed.)
Chapter 18 Learning from Examples
Chapter 20 Learning Probabilistic Models
Cross-validation
K-Fold Cross-validation
Decision Tree Datasets
hand_binary.csv: 4 features, 8 examples, binary classification (last column)
hand_multi.csv: 4 features, 12 examples, 3 classes, multi-class classification (last column)
simple_binary.csv: 5 features, 100 examples, binary classification (last column)
simple_multi.csv: 6 features, 100 examples, 3 classes, multi-class classification (last column)
mod_complex_binary.csv: 7 features, 1600 examples, binary classification (last column)
mod_complex_multi.csv: 10 features, 2400 examples, 5 classes, multi-class classification (last column)
complex_binary.csv: 10 features, 2800 examples, binary classification (last column)
complex_multi.csv: 16 features, 4800 examples, 9 classes, multi-class classification (last column)
part23_data.csv: 4 features, 1372 example, binary classification (last column)
Not provided, but will have less class separation and more centroids per class. Complex sets given for development
challenge_binary.csv: 10 features, 5400 examples, binary classification (last column)
challenge_multi.csv: 16 features, 10800 examples, 9 classes, multi-class classification (last column)
NOTE: path to the datasets! './data/your_file_name.csv'
Warmup Data
vectorize.csv: data used during the vectorization warmup for Assignment 4

Imports
NOTE: We are only allowing four imports: numpy, math, collections.Counter and time. We will be checking to see if any other libraries are used. You are not allowed to use any outside libraries especially for part 4 (challenge). Please remember that you should not change add or change any input parameters other than in part 4.

Rounding
NOTE: Although your local tests will have some rounding, it is meant to quickly test your work. Overall this assignment follows the CI 6601 norm of rounding to 6 digits. If in doubt, in use round:

x = 0.12345678
round(x, 6)
Out[4]: 0.123457
Part 0: Vectorization!
[10 pts]

File to use to benchmark tests: vectorize.csv
The vectorization portion of this assignment will teach you how to use matrices to significantly increase the speed and decrease processing complexity of your AI problems. Matrix operations, linear algebra and vector space axioms and operations will be challenging to learn. Learn this, and it will benefit you throughout OMSCS courses and your career.

You will not be able to meet time requirements, nor process large datasets without Vectorization Whether one is training a deep neural network on millions of images, building random forests for a U.S. National Insect Collection dataset, or optimizing algorithms, machine learning requires extensive use of vectorization. The NumPy Open Source project provides the numpy python scientific computing package using low-level optimizations written in C. If you find the libraries useful, consider contributing to the body of knowledge (e.g. help update and advance numpy)

You will not meet the time limits on this assignment without vectorization. This small section will introduce you to vectorization and one of the high-demand technologies in python. We encourage you to use any numpy function to do the functions in the warmup section. You will need to beat the non-vectorized code to get full points.

TAs will offer little help on this section. This section was created to help get you ready for this and other assignments; feel free to ask other students on Ed Discussion or use some training resources. (e.g. https://numpy.org/learn/)

How grading works:

We run your vectorized code against non-vectorized code 500 times, as long as you have the correct answer and your average time is significantly less than the average time of the non-vectorized code, you will get the points.
Functions to complete in the Vectorization class:
vectorized_loops()
vectorized_slice()
vectorized_flatten()
vectorized_glue()
vectorized_mask()
The Assignment

[Creative Commons sourced][cc]
E. Thadeus Starner5th is the 5th incarnation of the great innovator and legendary pioneer of Starner Eradicatus Mosquitoes. For centuries the mosquito has imparted only harm on human health, aiding in transmission of malaria, dengue, Zika, chikungunya, CoVid, and countless other diseases impact millions of people and animals every year. The Starner Eradicatus, Anopheles Stephensi laser zapper has obtained the highest level of precision, recall, and accuracy in the industry!

[Creative Commons sourced][cc]
The secret is the classification engine which has compiled an unmatched library of classification data collected from 153 countries. Flying insects from the tiny Dicopomorpha echmepterygis (Parasitic Wasp) to the giant titanus giganteus (Titan Beetle) are carefully catalogued in a comprehensive digital record and indexed to support fast and correct classification. This painstaking attention to detail was ordered by A. Thadeus1st to address a tumultuous backlash from the International Pollinators Association to a high mortality among beneficial pollinators.

[Creative Commons sourced][cc]
E. Thadeus' close friend Skeeter Norvig, a former CMAO (Chief Mosquito Algorithm Officer) and pollinator advocate has approached E.T. with an idea. The agriculture industry has been experiencing terrible losses worldwide due to the diaphorina citri (Asian Citrus Psyllid), drosophila suzuki (spotted wing Drosophila), and the bactrocera tyron (Queensland fruit fly).

[Creative Commons sourced][cc]
Skeeter explains his idea to E.T. to generalize the Starner Eradicatus zapper to handle a variety of these pests. Wonderful! E.T. exclaims, and becomes wildly excited at the opportunity to bring such an important benefit to the World.

[Creative Commons sources below][cc]
The wheels of invention lit up the research Scrum that morning as E.T. and Skeeter storyboard the solution. People are calling out all the adjustments, wing acoustics, laser power and duration, going through xyz positioning, angular velocity and acceleration calculations, speed, occlusion noise and tracking errors. You as the lead DT software engineer are taking it all in, when you realize and speak up..., sir... Sir... SIR... and a hush falls. Sir, we are doing Boolean classification and will need to refactor to multi-class classification. E.T. turns to you and with that look in his eye, gives you and your team two weeks to deliver multi-class classification!

You will build, train and test decision tree models to perform multi-class classification tasks. You will learn how decision trees and random forests work. This will help you develop an intuition for how and why accuracy differs for training and testing data based on different parameters.

Assignment Introduction
For this assignment we need an explicit way to make structured decisions. The DecisionNode class will be used to represent a decision node as some atomic choice in a multi-class decision graph. You must use this implementation for the nodes of the Decision Tree for this assignment to pass the tests and receive credit.

An object of type 'DecisionNode' can represent a

decision node
left: will point to less than or equal values of the split value, type DecisionNode, True evaluations
right: will point to greater than values of the split value, type DecisionNode, False evaluations
decision_function: evaluates an attribute's value and maps each vector to a descendant
class_label: None
leaf node
left: None
right: None
decision_function: None
class_label: A leaf node's class value
Note that in this representation 'True' values for a decision take us to the left.
Part 1: Building a Binary Tree by Hand
Part 1a: Build a Tree
[10 Pts]

In build_decision_tree(), construct a decision tree capable of predicting the class (col y) of each example. Using the columns A0-A3 build the decision tree and nodes in python to classify the data with 100% accuracy. Your tests should use as few attributes as possible, break ties with equal select attributes by selecting the one which classifies the greatest number of examples correctly. For ties in number of attributes and correct classifications use the lower index numbers (e.g. select A1 over A2)


X	A0	A1	A2	A3	y
x01	1.1125	-0.0274	-0.0234	1.3081	1
x02	0.0852	1.2190	-0.7848	-0.7603	2
x03	-1.1357	0.5843	-0.3195	0.8563	0
x04	0.9767	0.8422	0.2276	0.1197	1
x05	0.8904	-1.7606	0.3619	-0.8276	0
x06	2.3822	-0.3122	-2.0307	-0.5065	2
x07	0.7194	-0.4061	-0.7045	-0.0731	2
x08	-2.9350	0.7810	-2.5421	3.0142	0
x09	2.4343	-1.5380	-2.7953	0.3862	2
x10	0.8096	-0.2601	0.5556	0.6288	1
x11	0.8577	-0.2217	-0.6973	-0.1095	1
x12	0.0568	0.0696	1.1153	-1.1753	0
Requirements:
The total number of elements(nodes, leaves) in your tree should be < 10

Hints:
To get started, it might help to draw out the tree by hand with each attribute representing a node.

To create a decision function for DecisionNode, you are allowed to use python lambda expressions:

func = lambda feature : feature[2] <= 0.356
This will choose the left node if the A2 attribute is <= 0.356.

For example, a hand binary tree might look like this:

func = lambda feature : feature[0] <= -0.918
in this example if feature[0] is evaluated as true then it would belong to the leaf for class = 1; else class = 0

Tree Example

You would write your code like this:

func0 = lambda feature : feature[0] <= -0.918
decision_tree_root = DecisionNode(None, None, func0, None)
or
decision_tree_root = DecisionNode(None, None, lambda feature : feature[0] <= -0.918, None)
decision_tree_root.left = DecisionNode(None, None, None, class1)
decision_tree_root.right = DecisionNode(None, None, None, class0)
return decision_tree_root
Functions to complete in the submission module:
build_decision_tree()
Part 1b: Precision, Recall, Accuracy and Confusion Matrix
[12 pts]

To build the Starner Zapper next-gen, we will need to keep the high levels of Precision, Recall, and Accuracy inculcated in the legacy products. In binary/boolean classification we find these metrics in terms of true positives, false positives, true negatives, and false negatives. So it should be simple right?

Helpful Information:
A confusion matrix is a table that counts the number of occurrences between the true/actual classification and the predicted classification. The columns stand for your model prediction and the rows the true label of the feature.
Your confusion matrix will be a K x K matrix, where K = number of classes. Notice that the correct classifier predictions form the diagonal of the matrix. For each class, its diagonal will tell you how many times your classifier predicted correctly (TP), its column (TP omitted) will inform you how many times your classifier predicted a positive value incorrectly (FP), where its row (TP omitted) will inform you how many times your classifier did not correctly predict an actual positive (FN). You will find the referenced material very helpful for your calculations. Recall and Precision return lists whereas Accuracy will return a value. We know that may be illogical, the world of AI has many versions of this so we are trying to limit getting lost in which version. Helpful numpy functions are, diag, diagonal, sum(by axis) for this part of the assignment.

We will use Accuracy, but encourage you to think about the other versions:

Accuracy TP + TN / TP + TN + FP + FN USE THIS
Will tell you overall accuracy but not bias, why?
Balanced Accuracy: Sum of the ratios (accurate divided by sum of its row) divided by number of classes.
How could this skew the results, and why?
Balanced Accuracy Weighted: Balanced Accuracy with weighting added in the numerator and denominator.
Would this be good for the problem?
Precision: TP / TP + FP
Recall: TP / TP + FN

To-do

Confusion Matrix, Accuracy, Precision, Recall
Helpful references:
Wikipedia
Metrics for Multi-Class Classification
Performance Metrics for Activity Recognition Sec 5.

If you want to calculate the example set above by hand, run the following code.

tree = dt.build_decision_tree()
answer = [tree.decide(example) for example in examples]
n_classes = 3
clf_matrix = confusion_matrix(classes, answer, n_classes)
clf_accuracy = accuracy(classes, answer, n_classes )
clf_precision = precision(classes, answer, n_classes)
clf_recall = recall(classes, answer, n_classes)
print(clf_confusion_matrix, clf_accuracy, clf_precision, clf_recall)
Functions to complete in the submission module:
confusion_matrix()
precision()
recall()
accuracy()
Part 2: Decision Tree Learning
Part 2a: Gini
[10 pts]

Purity, we strive for purity, alike Sir Galahad the Pure... Splitting at a decision is all about purity. You are trying to improve information gain which means, you are trying to gain purer divisions of the data. Through purer divisions of the data it is more ordered. This relates to entropy in physics, where ordered motion produces more energy. Through ordered data you gain more information on the defining characteristics (attributes) of something observed.

We will use GINI impurity and the GINI Impurity Index to calculate the gini_impurity and gini_gain() on the splits to calculate Information Gain. The challenge will be to choose the best attribute at each decision with the lowest impurity and the highest index. At each attribute we search for the best value to split on, the hypotheses are compared against what we currently know, because would we want to split if we learn nothing? Hints:

gini impurity
information gain
The Gini Gain follows a similar approach to information gain, replacing entropy with Gini Impurity.
Numpy helpful functions include advanced indexing, and filtering arrays with masks, slicing, stacking and concatenating

Functions to complete in the submission module:
gini_impurity()
gini_gain()
Part 2b: Decision Tree Learning
[25 pts]

Data to train and test with: simple_binary.csv, simple_multi.csv, mod_complex_binary.csv, mod_complex_multi.csv

Grading:

15 pts: average test accuracy over 10 rounds should be >= 50%
20 pts: average test accuracy over 10 rounds should be >= 60%
25 pts: average test accuracy over 10 rounds should be >= 75%

Meanwhile back in the lab... As the size of our flying training set grows, it rapidly becomes impractical to build multiclass trees by hand. We need to add a class with member functions to manage this, it is too much!
To do list:

Initialize the class with useful variables and assignments
Fill out the member function that will fit the data to the tree, using build
Fill out the build function
Fill out the classify function
For starters, consider these helpful hints for the construction of a decision tree from a given set of examples:

Watch your base cases:
If all input vectors have the same class, return a leaf node with the appropriate class label.
If a specified depth limit is reached, return a leaf labeled with the most frequent class.
Splits producing 0, 1 length vectors
Splits producing less or equivalent information
Division by zero
Use the DecisionNode class
For each attribute alpha: evaluate the information gained by splitting on the attribute alpha.
Let alpha_best be the attribute value with the highest information gain.
As you progress in this assignment this is going to be tested against larger and more complex datasets, think about how it will affect your identification and selection of values to test.
Create a decision node that splits on alpha_best and split the data and classes by this value.
When splitting a dataset and classes, they must stay synchronized, do not orphan or shift the indexes independently
Use recursion to build your tree, by using the split lists, remember true goes left using decide
Your features and classify should be in numpy arrays where for dataset of size (m x n) the features would be (m x n-1) and classify would be (m x 1)
The features are real numbers, you will need to split based on a threshold. Consider different approaches for what this threshold might be.
First, in the DecisionTree.__build_tree__() method implement the above algorithm. Next, in DecisionTree.classify(), write a function to produce classifications for a list of features once your decision tree has been built.

How grading works:

We load mod_complex_multi.csv and create our cross-validation training and test set with a k=10 folds. We use our own generate_k_folds() method.
We fit the (folded) training data onto the tree then classify the (folded) testing data with the tree.
We check the accuracy of your results versus the true results and we return the average of this over k=10 iterations.
Functions to complete in the DecisionTree class:
__build_tree__()
fit()
classify()
Part 2c: Validation
[7 pts]

File to use: part23_data.csv
Allowed use of numpy, collections.Counter, and math.log
Grading: average test accuracy over 10 rounds should be >= 80%
Reserving part of your data as a test set can lead to unpredictable performance as you may not have a complete set or proportionately balanced representation of the class labels. We can overcome this limitation by using k-fold cross validation. There are many benefits to using cross-validation. See book 3rd Ed Evaluating and Choosing the Best Hypothesis 18.4

In generate_k_folds(), we'll split the dataset at random into k equal subsections. Then iterating on each of our k samples, we'll reserve that sample for testing and use the other k-1 for training. Averaging the results of each fold should give us a more consistent idea of how the classifier is doing across the data as a whole. For those who are not familiar with k folds cross-validation, please refer the tutorial here: A Gentle Introduction to k-fold Cross-Validation.

How grading works:

The same as 2b however, we use your generate_k_folds() instead of ours.
Functions to complete in the submission module:
generate_k_folds()
Part 3: Random Forests
[25 pts]

File to use: mod_complex_binary.csv, mod_complex_multi.csv
Allowed use of numpy, collections.Counter, and math.log
Allowed to write additional functions to improve your score
Allowed to switch to Entropy and splitting entropy
Grading:
15 pts: average test accuracy over 10 rounds should be >= 50%
20 pts: average test accuracy over 10 rounds should be >= 70%
25 pts: average test accuracy over 10 rounds should be >= 80%
Decision boundaries drawn by decision trees are very sharp, and fitting a decision tree of unbounded depth to a set of training examples almost inevitably leads to overfitting. In an attempt to decrease the variance of your classifier you are going to use a technique called 'Bootstrap Aggregating' (often abbreviated as 'bagging'). Decision stumps are very short decision trees used in Ensemble classification such as Random Forests

They are usually short (depth limited)
They use smaller (but more of them) random datasets for training
They use a subset of attributes drawn randomly from the training set
They fit the tree to the sampled dataset and are considered specialized to the set
Advanced learners may use weighting of their classifier trees to improve performance
They use majority voting (every tree in the forest votes) to classify a sample
A Random Forest is a collection of decision trees, built as follows:

For every tree we're going to build:
Subsample the examples provided us (with replacement) in accordance with a provided example subsampling rate.
From the samples, choose attributes (without replacement) at random to learn on (as specified in attribute subsampling rate)
Fit a decision tree to the subsample of data we've chosen (to a limited depth).
When sampling attributes you choose from the entire set of attributes for every sample drawn (but specify the sampling method does not use replacement)

Complete RandomForest.fit() to fit the decision tree as we describe above, and fill in RandomForest.classify() to classify a given list of examples. You can use your decision tree implementation or create another.

Your features and classify should use numpy arrays datasets of (m x n) features of (m x n-1) and classify of (n x 1).

To test, we will be using a forest with 80 trees, with a depth limit of 5, example subsample rate of 0.3 and attribute subsample rate of 0.3

How grading works:

Similar to 2b but with the call to Random Forest.
Functions to complete in the RandomForest class:
fit()
classify()
Part 4 (Optional) Boosting Competition Challenge (Extra Credit)
Let the games begin! --- 让游戏开始 --- ας ξεκινήσει το παιχνίδι
THIS WILL REQUIRE A SEPARATE SUBMISSION
Files to use: complex_binary.csv, complex_multi.csv
Allowed use of numpy, collections.Counter, and math.log
Allowed to write additional functions to improve your score
Allowed to switch to Entropy and splitting entropy
Ranked by Balanced Accuracy Weighted, Precision, and Recall
Ties broken by efficiency (speed)
Extra Credit Points towards your final grade:
5 pts: 1st place algorithm test over 10 rounds
4 pts: 2nd place algorithm test over 10 rounds
3 pts: 3rd place algorithm test over 10 rounds
2 pts: 4th place algorithm test over 10 rounds
1 pt: 5th place algorithm test over 10 rounds
Decision boundaries drawn by decision trees are very sharp, and fitting a decision tree of unbounded depth to a set of training examples almost inevitably leads to overfitting. In an attempt to decrease the variance of your classifier you are going to use a technique called 'Boosting' implementing one of the boosting algorithms such as, Ada-, Gradient- and XG-, boost or your personal favorite. Similar to RF, the Decision stumps are short decision trees used in these Ensemble classification methods

They are usually short (depth limited)
They use smaller (but more of them) random datasets for training with sampling bias
They use a subset of attributes sampled from the training set
They fit the tree to the sampled dataset and are considered specialized to the set
They use weighting of their sampling and classifiers to reflect the balance or unbalance of the data
They use majority voting (every tree in the forest votes) to classify a sample
Ada-boost Algorithm example [Zhu, et al.]:

N Samples, M classifiers, W weights, C classifications, K classes, I indicator
Initialize the observation weights wi = 1/n, i = 1, 2, . . . , n.
For m = 1 to M:
Fit a classifier T(m)(x) to the training data using weights wi.
Compute err(m) = Sum(i=1..n)wi I(ci != T(m)(xi)) / Sum(i=1..n) wi
Compute α(m) = log (1−err(m)/err(m)) + log(K − 1).
Set wi ← wi · exp (α(m) I(ci != T(m)(xi)), i = 1, . . . , n.
Re-normalize wi.
Output C(x) = argmax(k) sum(m=1..M) α(m) · I(T(m)(x) = k).
Multi-class AdaBoost Zhu, Ji & Rosset, Saharon & Zou, Hui & Hastie, Trevor. (2006). Multi-class AdaBoost. Statistics and its interface. 2. 10.4310/SII.2009.v2.n3.a8.

When sampling attributes you choose from the entire set of attributes without replacement based on the weighted distribution. Notice that you favor or bias towards misclassified samples, which improves your overall accuracy. Visualize how the short trees balance classification bias.

Complete ChallengeClassifier.fit() to fit the decision tree as we describe above, and fill in ChallengeClassifier.classify() to classify examples. Use your decision tree implementation as your classifier or create another.

Your features and classify should use numpy arrays datasets of (m x n) features of (m x n-1) and classes of (m x 1).

How grading works: To test, we will be running 10 rounds, using your boosting with 200 trees, with a depth limit of 3, example subsample rate of 0.1 and attribute subsample rate of 0.2. You will have a time limit.

Functions to complete in the ChallengeClassifier class:
init()
fit()
classify()
Part 5: Return Your name!
[1 pts] Return your name from the function return_your_name()

Helper Notebook
Note: You do not need to implement anything in this notebook. This part is not graded, so you can skip this part if you wish to. This notebook is just for your understanding purpose. It will help you visualize Decision trees on the dataset provided to you.
The notebook Visualize_tree.iypnb can be use to visualize tree on the datasets. Things you can Observe:

How the values are split?
What is the gini value at leaf nodes?
What does internal nodes represents in this DT?
Why all leaf nodes are not at same depth?
Feel free to change and experiment with this notebook. you can look and use Information gain as well instead of gini to see how the DT built based on that.

Video and picture attribution
[cc]: All GT OMSCS materials are copyrighted and retain rights prohibiting redistribution or alteration
Creative Commons sources share and share alike were used for the videos:
[[File:Lucanus Cervus fighting.webm|Lucanus_Cervus_fighting]] Nicolai Urbaniak, CC0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:2013-06-07_13-41-29-insecta-anim.ogv
Thomas Bresson, CC BY 3.0 https://creativecommons.org/licenses/by/3.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Traumatic-insemination-and-female-counter-adaptation-in-Strepsiptera-(Insecta)-srep25052-s2.ogv
Peinert M, Wipfler B, Jetschke G, Kleinteich T, Gorb S, Beutel R, Pohl H, CC BY 4.0 https://creativecommons.org/licenses/by/4.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Broscus_cephalotes.webm
О.Н.П., CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Termites_walking_on_the_floor_of_Eastern_Himalyan_rainforest.webm
Jenis Patel, CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:The_carrot_caterpillar_-.webm
Contributor NamesPathé frères (France)Pathé Frères (U.S.)AFI/Nichol (Donald) Collection (Library of Congress)Created / PublishedUnited States : Pathé Frères, 1911., Public domain, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Gaeana_calling.webm
Shyamal, CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Kluse_-_Tenebrio_molitor_larvae_eating_iceberg_lettuce_leaf_v_02_ies.webm
Frank Vincentz, CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Lispe_tentaculata_male_-_2012-05-31.ogv
Pristurus, CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Xylotrupes_socrates_(Siamese_rhinoceros_beetle)_behavior.webm
Basile Morin, CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0, via Wikimedia Commons laser

https://commons.wikimedia.org/wiki/File:Mosquito_dosing_by_laser_2.webm
https://commons.wikimedia.org/wiki/File:Mosquito_dosing_by_laser_3.webm
https://commons.wikimedia.org/wiki/File:Mosquito_dosing_by_laser_4.webm
Matthew D. Keller et al., CC BY 4.0 https://creativecommons.org/licenses/by/4.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Laser-induced-mortality-of-Anopheles-stephensi-mosquitoes-srep20936-s5.ogv
Keller M, Leahy D, Norton B, Johanson T, Mullen E, Marvit M, Makagon A, CC BY 4.0 https://creativecommons.org/licenses/by/4.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Predicting-Ancestral-Segmentation-Phenotypes-from-Drosophila-to-Anopheles-Using-In-Silico-Evolution-pgen.1006052.s001.ogv
Rothschild J, Tsimiklis P, Siggia E, François P, CC BY 4.0 https://creativecommons.org/licenses/by/4.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Focused_Laguerre-Gaussian_beam.webm
Jack Kingsley-Smith, CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:President_Reagan%27s_Remarks_at_Bowling_Green_State_University,_September_26,_1984.webm Reagan Library, CC BY 3.0 https://creativecommons.org/licenses/by/3.0, via Wikimedia Commons
https://commons.wikimedia.org/wiki/File:NID_Participants_Preparing_Their_Project_-_Workshop_On_Design_And_Development_Of_Digital_Experiencing_Exhibits_-_NCSM_-_Kolkata_2018-08-09_3141.ogv
Biswarup Ganguly, CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0, via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Davos_2017_-_An_Insight,_An_Idea_with_Sergey_Brin.webm World Economic Forum, CC BY 3.0 https://creativecommons.org/licenses/by/3.0, via Wikimedia Commons
Creative Commons Attribution-Share Alike 4.0


Coding Help Email: 📧 xujuncoding@gmail.com
