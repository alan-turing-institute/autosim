# Stratified sampling

For the Gray-Scott system, there are different 'patterns' which represent different categories of behaviour.
To ensure that training data is representative of all these patterns, we can use stratified sampling to generate a dataset that contains a balanced number of each of these.

Here is a sample command:

```bash
uv run autosim \
    simulator=spatiotemporal/gray_scott \
    stratify.enabled=true \
    stratify.key=simulator.pattern \
    'stratify.values=[gliders,bubbles,maze,worms,spirals,spots]' \
    dataset.n_train=240 dataset.n_valid=24 dataset.n_test=24 \
    dataset.output_dir=outputs/gray_scott_combined
```

When stratification is enabled, the training, validation, and test sets will be divided equally across strata.
That means that, in this example, the training set will contain 40 samples of each pattern, the validation set will contain 4 samples of each pattern, and the test set will contain 4 samples of each pattern.

Results are concatenated in the exact order of `stratify.values`, so the first 40 samples of the training set will be gliders, followed by 40 samples of bubbles, and so on.

If a split size is not divisible by the number of strata, an error is raised.
