![# A Noise Contrastive Approach to Generative Sequence Model Evaluation](https://github.com/peglegpete/gen-seq-noise/blob/master/git-title.png)

This repository contains the code and results to the experiments detailed in (LINK HERE)

**Abstract:**

# Where to find what
## data
## data-preparation
This folder contains the following scripts that were used to create the experimental data:
- *baseline-output-batcher.ipynb*: uses the CNN/Daily Mail test set outputs of the baseline model provided by See et al. to produce single data points to folder along with a labels and partition dictionary
- *clean1to1-test-only-batcher.ipynb*: uses the clean data points from the CNN/Daily Mail test set to produce single data points to folder along with a labels and partition dictionary
- *generator1to1-batcher.ipynb*: uses the data points from the CNN/Daily Mail train and validation set and the respective outputs for these datapoints from See et. al's PGC network to produce single data points of generator noise to folder along with a labels and partition dictionary
- *generator1to1-test-only-noise-batcher.ipynb*: uses the data points from the CNN/Daily Mail test set and the respective outputs for these datapoints from See et. al's PGC network to produce single data points of generator noise to folder along with a labels and partition dictionary
- *pgc-output-batcher.ipynb*: uses the CNN/Daily Mail test set outputs of the PGC network provided by See et al. to produce single data poins to folder along with a labels and partition dictionary
- *pg-output-batcher.ipynb*: uses the CNN/Daily Mail test set outputs of the PG network provided by See et al. to produce single data points to folder along with a labels and partition dictionary
- *pseudorandom1to1-batcher.ipynb*: uses the data points from the CNN/Daily Mail train and validation set to produce single data points of pseudo-random noise to folder along with a labels and partition dictionary
- *pseudorandom1to1-test-only-noise-batcher.ipynb*: uses the data points from the CNN/Daily Mail test set to produce single data points of pseudo-random to folder along with a labels and partition dictionary
- *sumgan-output-batcher.ipynb*: uses the CNN/Daily Mail test set outputs of the SumGAN model provided by Liu et al. to produce single data points to folder along with a labels and partition dictionary
- *training-stats-maker.ipynb*: uses the CNN/Daily Mail training data to build embedding_matrix and the tokenizer object which are used to transform data points in all experiments to the embedded GloVe representation
## experiments
This folder contains the scripts and results for the experiments presented in the paper. In addition it includes the scripts that were used to do the model evaluation and produce the paper's plots.
- eval-script
  - *eval_template_allmodels.ipynb*: contains the script and results of each model type on the testing data
  - *eval_template_allmodels_threshold-tests-4.ipynb*: contains the script and results of the threshold anaylsis with respect to specificity and sensitivity
  - *eval_template_output-scoring.ipynb*: contains the script and results of the output scoring of different generative models' outputs with the best discriminator
  - *generators.py*: contains the custom batch generator class that creates the batches for evaluation
  - *stats-4.pickle*: a dictionary of the results from *eval_template_allmodels_threshold-tests-4.ipynb*, can be loaded to reproduce the plots or data without the need for recomputation as this takes some time
