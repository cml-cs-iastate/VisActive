Running Sequence

For each iteration [1 .. 10]

1. settings.py
    - Adjust the parameters and hyper-parameters for the model

2. train_baseline.py
    - Train the baseline model on the iteration dataset (start with initial dataset)
    - Test the best trained model
    - Calculate the precision, recall, and f1 measure for evaluation

3. train_hvc.py
    - Train the hvc model to learn the VC for each class
    - The output model is going to be store in the output directory

4. calculate_cov_uniq.py
    - Calculate the coverage and uniqueness for each visual concept
    - Store the output dictionaries
    - These dictionaries are going to use to select the rare class common concepts and rare class rare concepts

5. search.py
    - Search for the unlabeled images in the search folder
    - Load the coverage and uniqueness dictionaries
    - Select the rare class common concepts and rare class rare concepts
    - Select the common class common concepts and common class rare concepts
    - Print and save these four lists
    - Manually label these images and add it to the initial dataset to run the next iteration
