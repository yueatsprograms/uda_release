Code release for [Unsupervised Domain Adaptation through Self-Supervision](https://arxiv.org/abs/1909.11825).
Our code requires pytorch version 1.0 or higher.

1. Modify the scripts in <code>scripts</code>. <code>--outf</code> sets the output folder, and <code>--data_root</code> sets the input data directory.
2. Run the scripts in <code>scripts</code>. Each script corresponds to a column in our results table.
3. <code>show_table.py</code> prints the results and plots the error curves. For ease of comparison, two results are printed: the smallest error and the early stopping error according to our selection heuristics.
4. <code>figure_mmd.py</code> reproduces the two convergence figures shown in our paper.
