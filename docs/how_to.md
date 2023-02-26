# How to get started
1) Pull the project and make sure you have `poetry` installed
2) run `poetry install` in root
3) if all dependencies are sorted, `run poetry run python main.py`

# How to use
1) The program will initially ask you for a sample. The maximum sample is whatever the amount of images in `data/test_data` happens to be. By default there are a 30 yaleface images, so the maximum amount of sample images is 30. 
2) Now you have inserted the sample you want, and can run the main program by pressing `1`. This will start a more or less lengthy calculation that ends at a comparison between images in `test_data` and images in `unknown images`
3) You can also see the eigenfaces in `data/outputs/eigenfaces`

Ideally you'll run this program in an IDE, as that gives you the best vantage point between different folders.
