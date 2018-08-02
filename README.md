![# A Noise Contrastive Approach to Generative Sequence Model Evaluation](https://github.com/peglegpete/gen-seq-noise/blob/master/git-title.png)

This repository contains the code and results to the experiments detailed in (LINK HERE)

**Abstract:**

# Where to find what
## data
This folder contains the bash script *data.sh* which you can run as `bash data.sh` from the terminal to download the data files used in the experiments and unzip the archive. You can also manually download the zip archive from `https://drive.google.com/open?id=1zo4RDJ6boLHKnHajWMgQ3ty7PsZuJz2U`. The archive contains the following files:
- exp-data/evaluation-data/test-onlyclean/: *only-clean.zip* contains the labels dictionary, partition dictionary and data points of clean text-summary pairs of the CNN/Daily Mail test set
- exp-data/evaluation-data/test-onlynoise/generator-dist/: *only-noise.zip* contains the labels dictionary, partition dictionary and data points of generator noise text-summary pairs of the CNN/Daily Mail test set
- exp-data/evaluation-data/test-onlynoise/pseudorandom-dist/: *only-noise.zip* contains the labels dictionary, partition dictionary and data points of pseudo-random noise text-summary pairs of the CNN/Daily Mail test set
- exp-data/generator-dist-1to1/: *1to1.zip* contains the labels dictionary, partition dictionary and data points of clean and generator noise text-summary pairs of the CNN/Daily Mail train and validation set
- exp-data/pseudorandom-dist-1to1/: *1to1.zip* contains the labels dictionary, partition dictionary and data points of clean and pseudo-random noise text-summary pairs of the CNN/Daily Mail train and validation set
- exp-data/output-scoring/: *baseline.zip*, *pg.zip*, *pgc.zip* and *sumgan.zip* each contain the labels dictionary, partition dictionary and data points of output text-summary pairs of the respective genertaive model on the CNN/Daily Mail test set
- orig-files/: contains the original train (*train.bin*), validation (*val.bin*) and test (*test.bin*) splits of the CNN/Daily Mail dataset, tokenized and lowercased (detailed preprocessing description see paper)
- stats-and-meta-data/400000/training-stats-all/: contains *maxi.npy* and *mini.npy*, the parameters for the uniform noise as computed on the embedded training data
- stats-and-meta-data/400000/: contains *embedding_matrix.npy*, the fixed embedding matrix as constructed on the training data, and *tokenizer.pickle*, a tokenizer object that is used to map natural language strings to the embedded representation via the embedding matrix

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
- ow-on-generator: Contains the scripts and results of the experiments run with the one-way discriminator architecture trained with generator noise. There are ten folders numbered 1 to 10 that correspond to the different trials of the experiment with different random seeds. Each trial folder contains the following files.
  - logs: the tensorboard callback folder with the event file, can be accessed by calling `tensorboad --logdir=/logs`
  - *accs.pickle*: a list of the batch-wise training accuracies of the experiment
  - *best.h5*: the best model as validated on the validation loss
  - *generators.py*: contains the custom batch generator class that feeds data points during training
  - *losses.pickle*: a list of the batch-wise training losses of the experiment
  - *ow_template.ipynb*: the script and results of the experiment run on the one-way discriminator architecture
  - *run_.-tag-val_acc.csv*: epoch-level validation accuracies of the experiment
  - *run_.-tag-val_loss.csv*: epoch-level validation losses of the experiment
- ow-on-pseudorandom: Contains the scripts and results of the experiments run with the one-way discriminator architecture trained with pseudo-random noise. There are ten folders numbered 1 to 10 that correspond to the different trials of the experiment with different random seeds. Each folder has the same structure as described above in ow-on-generator.
- ow-on-uniform: Contains the scripts and results of the experiments run with the one-way discriminator architecture trained with uniform noise. There are ten folders numbered 1 to 10 that correspond to the different trials of the experiment with different random seeds. Each folder has the same structure as described above in ow-on-generator.
- plots: Contains the following folders that were used to produce plots of the paper
  - plotter-all-10-runs: contains *max-notebook.ipynb* (used to validate best performing model) and *plotter-all-10-runs-ipynb* (used to produce the training results averaged plots)
  - see-nll: contains *plotting.ipynb* (used to create the NLL plot of See et al.' PGC network) and *run_.-tag-seq2seq_loss_loss.csv* (the NLL loss history of See et al.' PGC network)
- tw-on-generator: Contains the scripts and results of the experiments run with the two-way discriminator architecture trained with generator noise. There are ten folders numbered 1 to 10 that correspond to the different trials of the experiment with different random seeds. Each folder has the same structure as described above in ow-on-generator.
- tw-on-pseudorandom: Contains the scripts and results of the experiments run with the two-way discriminator architecture trained with pseudo-random noise. There are ten folders numbered 1 to 10 that correspond to the different trials of the experiment with different random seeds. Each folder has the same structure as described above in ow-on-generator.
- tw-on-uniform: Contains the scripts and results of the experiments run with the two-way discriminator architecture trained with uniform noise. There are ten folders numbered 1 to 10 that correspond to the different trials of the experiment with different random seeds. Each folder has the same structure as described above in ow-on-generator.
