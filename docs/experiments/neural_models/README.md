# Training Neural Models Using Generated Nonograms Dataset

Experiments TODO
- Different dataset sizes
- Different dataset shapes
- train on one size, test on another
- ablation
    - model
    - loss function
    - number of layers
    - abstain loss, does it know where it is wrong, does it refine its output
- train model on solving traces
    - see if the model performs better on some trace
    - does it generalise better
    - how similar are traces on unseen puzzles
    - how similar on larger puzzles
- does model know when it is wrong?
    - are incorrect ones on all high abstain cells
    - is abstain value high when p(0) ~= p(1)