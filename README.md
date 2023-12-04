# N-GlycoPred-master

We constructed a hybrid deep learning model N-GlycoPred on the basis of dual-layer convolution, a paired attention mechanism and BiLSTM for accurate identification of N-glycosylation sites. By adopting one-hot encoding or the AAindex, we specifically selected the optimum combination of features and deep learning frameworks for humans and mice to refine the models. 

Explanation: If you want to use N-GlycoPred to predict N-glycosylation sites on a protein, you need to first truncate the protein into a peptide segment of length 21 (10 amino acids before and after "N" as the center), and then use the predict.py file to predict whether the truncated peptide segment is an N-glycosylation site.
