## Fetch Rewards Coding Exercise - Data Science

### Problem Statement

At fetch, we are monitoring the number of the scanned receipts to our app on a daily base as one of our KPIs. From business standpoint, we sometimes need to predict the possible number of the scanned receipts for a given future month.

The dataset provides the number of the observed scanned receipts each day for the year 2021. Based on this prior knowledge, please develop an algorithm which can predict the approximate number of the scanned receipts for each month of 2022.

## Solution

The solution for the problem statement has been described in detail in the [report.pdf](https://www.github.com/sagar118/) file. Also, please find the app deployed on HuggingFace Spaces [here](https://huggingface.co/spaces/sagar-thacker/fetch-rewards-application).

## Instructions to run the code

The code has been written in Python 3.9. To run the code, please follow the steps below:

### Run the Notebook

1. Clone the repository.
2. Create a virtual environment and activate it.

```bash
pip install pipenv
pipenv install
```

3. Run the notebook with mlflow. In a new terminal, run the following command in the root directory of the repository:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db
```

To view the mlflow UI, open `http://localhost:5000` or `http://127.0.0.1:5000` in your browser.

Open the notebooks to view the EDA and the model training from the notebooks directory.

### Run the Gradio App

The `deployment` and `fetch-rewards-app` directories contain the same code, however the `Dockerfile` is different. For the sake of simplicity, the deployment directory contains the code for the Gradio app locally, while the fetch-rewards-app directory contains the code for the app deployed on HuggingFace Spaces.

To run the Gradio app locally, run the following command:

```bash
cd deployment

docker build -t fetch-rewards-app:0.1 .
docker run -it -p 7860:7860 fetch-rewards-app:0.1
```

Open `http://localhost:7860` in your browser to view the app.

Alternatively, you can view the app deployed on HuggingFace Spaces [here](https://huggingface.co/spaces/sagar-thacker/fetch-rewards-application).

