Installation:
  pip install requirements.txt
  pip uninstall keras-preprocessing
  pip install git+https://github.com/keras-team/keras-preprocessing.git


Modules:
- get_recipes.py
  - `python3 get_recipes.py`
  - Crawls the Epicurious website for recipes and images
  - Generates file food_info.csv
- clean.py
  - `python3 clean.py`
  - Removes dead links and placeholder images from food_info.csv
  - Generates file food_info_cleaned.csv
- foodcnn.py
  - `python3 foodcnn.py`
  - Trains neural network
- predict.py
  - `python3 predict.py`
  - In browser: `http://localhost:5000/`
  - Upload test image, click submit. Will return calorie count
