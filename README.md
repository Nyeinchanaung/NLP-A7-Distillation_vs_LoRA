# NLP-A7-Distillation_vs_LoRA
In this assignment, we will explore the comparison between Odd Layer and Even Layer Student Training Models and LoRA (Low-Rank Adaptation) on a distillation task using BERT from Huggingface.

## GitHub Link:
- https://github.com/Nyeinchanaung/NLP-A7-Distillation_vs_LoRA


## Content
- [Student Information](#student-information)
- [Files Structure](#files-structure)
- [How to run](#how-to-run)
- [Dataset](#dataset)
- [Model Training](#training)
- [Evaluation](#evaluation)
- [Web Application](#application)

## Student Information
 - Name     : Nyein Chan Aung
 - ID       : st125553
 - Program  : DSAI

## Files Structure
1) The Jupytor notebook files
- distilBERT.ipynb (https://github.com/Nyeinchanaung/NLP-A7-Distillation_vs_LoRA/blob/main/distilBERT.ipynb)

2) `app` folder  
- app.py (streamlit)

 
## How to run
 - Clone the repo
 - Open the project
 - Open the `app` folder
 - `streamlit run app.py`
 - app should be up and running on `http://localhost:8501/`

## Dataset
HateXplain is a multilingual hate speech dataset containing over 25,000 social media posts annotated by multiple crowdworkers. Each example includes:
- A text post (from Twitter, Reddit, Gab)
- Multiple annotator labels: Hate Speech (0), Normal (1), Offensive (2)
- Rationales: token-level highlights for why the label was chosen
- Target groups (e.g., race, religion, gender)

To prepare labels for training, we apply majority voting over the annotators’ labels in each example. From the annotators`["label"]` field, we extract valid labels (0, 1, or 2) and assign the most frequent one as the final label. If no valid label exists, we default to label 1 (normal). This ensures each input has a single, consistent label for supervised training.

### Source: https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain 

## Training
Base model: `bert-base-uncased`
### Student variants:
- 6-layer student (odd/even)
- 12-layer student with LoRA

### Hyper Parameter
| Hyperparameter    | Value         |
|-------------------|---------------|
|Epochs	            |5              |
|Learning Rate	    |5e-5           |
|Optimizer	        |AdamW          |
|Weight Decay	    |0.01           |
|Scheduler	        |linear         |
|Warmup Steps	    |10% of total steps|
|Loss Functions	    |Classification Loss (CE), KL Divergence, Cosine Similarity|
|Loss Weights	    |avg: (cls + div + cos) / 3 or weighted (0.2, 0.5, 0.3)|
|Evaluation Metric	|F1-score (macro), plus Accuracy, Precision, Recall|
|Device	            |cuda (GPU)  |

## Evaluation
Three student models—Odd Layer, Even Layer, and LoRA—were trained and evaluated on the hate speech classification task using the HateXplain dataset.
- The Odd Layer model achieved the best overall performance, with the lowest training loss (0.2746) and a test F1-score of 65.21%, indicating effective learning and generalization.
- The Even Layer model, despite a much higher training loss (0.8450), matched the Odd Layer model in all test metrics, suggesting convergence to a similar solution.
- The LoRA model underperformed in both training and testing, with the highest loss and the lowest F1-score (38.33%), indicating difficulty in learning from the data under the current setup.

### Summary result

| Model Type    |Training Loss| Test Loss | Accuracy| Precision | Recall |F1-score|
|---------------|-------------|-----------|---------|-----------|--------|--------|
| Odd Layer     | **0.2746**   |0.8413 |0.6620 |0.6513 | 0.6532    |0.6521  | 
| Even Layer    | 0.8450   |0.8413 |0.6620 |0.6513 | 0.6532    |0.6521  |
| LoRA          | 1.0008   |1.0147 |0.4890 |0.3996 | 0.4553    |0.3833  |

In conclusion, Odd Layer distillation yielded the most efficient training and strong evaluation performance, while LoRA requires further tuning to be competitive in this task.

#### Accuracy Comparison
![accuracy](rs1.png)
#### F1 Score
![f1score](rs1.png)


## Application
### Application Development
The web application is a Streamlit-based toxic comment classifier that allows users to input any text and receive real-time predictions. It uses the best-trained BERT student model (e.g., Odd-Layer) to classify the text as Hate Speech, Offensive, or Normal, and displays the result with color-coded alerts and confidence scores for transparency and clarity.
### How to use web app
1) **Run the App:** (In your terminal, navigate to the project folder and run:)
2) **Enter Text:** In the input box, type any sentence or comment you want to analyze.
3) **Click "Classify":** The model will predict whether the text is Hate Speech, Offensive, or Normal.
4) **View Results:** The prediction is shown with a colored alert and confidence scores for each class.

### Screenshots
#### Hate Speech
![Webapp1](ss1.png)
#### Offensive speech
![Webapp1](ss2.png)
#### Normal input
![Webapp1](ss3.png)
#### How to use webapp
![Webapp2](ss4.gif)
## Conclusion