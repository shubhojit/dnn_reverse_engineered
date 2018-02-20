# dnn_reverse_engineered
Reverse engineering deep learning models

The basic idea was to reimplement CNNs for sentence classification something similar to https://arxiv.org/abs/1408.5882

The project has now slowly reshaped into an endeavour to create a scoring mechanism for publications.
With the wide range of publications in the research domain, I was overwhelmed and so i decided to classify sentences and 
now i intend to classify documents. The idea is as follows:

1) Select a particular document that i have deemed important.
2) Classify it as an important document, and extract essential features from the sentences.
3) Save these features.
4) For every new article/publication, run a matching algorithm to figure out if they have the same context.
5) If yes, assign a score (from 0 to 5)
6) Once, a score has been assigned figure out if it debunks the idea of the original paper or upholds it.
( i.e. assign a negative 5 score if the document is in the same category but the paper completely debunks the claims of the
original paper. Assign a +5 if the document is in the exact same category and espouses a positive "sentiment"

There are three scripts currently:
1) classify_text.py is an attempt to reimplement this paper: https://arxiv.org/abs/1408.5882
This model allows the usage of task-specific and static vectors	both

2) cnn_recurrent.py is an attempt to reimplement this paper: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552
This model has a recurrent structure to capture contextual information while learning word representations.

3) densenet_expts.py : figuring out the densenet architecture. 
The idea being, that for each layer, the feature-maps of all preceding layers will be used as inputs, and its own feature-maps will be the input to all subsequent layers. This could help with improving the contextual information.
