# How to get started
1) Pull the project and make sure you have `poetry` installed
2) run `poetry install` in root
3) if all dependencies are sorted, run `poetry run python main.py`

# How to use
0) The program will run tests! If these are passed, the command prompt will clear and execution starts. 
1) The program will initially ask you for a sample. Input a number between the given intervals. The number signifies the amount of images sample _from each subject_.
2) input ´1´ in order to start the configuration process
3) Press ´y´ unless you want to run the model on random data from ´data/test_data´; in this case give the parameters for ´operations.train_test_split´
4) Choose if you'd like to remove eigenvectors. If it is your first time, choose ´n´, but if you want to see the effects of doing so after trying to run the model once, try a number between 1 and 3 (1 recommended for standard data)
5) Input a parameter for variance. Generally ´0.95' is good. 
6) Choose if you want to save intermediary images into respective folders.
7) Now the model runs!

Ideally you'll run this program in an IDE like Visual Studio Code, as that gives you the best vantage point between different folders.
