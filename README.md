# GymIQ – Workout Intelligence System

GymIQ is a **data-driven workout analysis and performance intelligence system** designed to help gym-goers understand their training patterns, fatigue levels, and strength progress using **data science and statistical modeling**.

The system analyzes historical workout data and generates **actionable insights** such as optimal rest days, fatigue warnings, personal records, and strength progression trends.

Unlike basic workout trackers, GymIQ focuses on **performance analytics and training intelligence**.

---

# Features

## 1. Workout Analytics

GymIQ analyzes workout logs and computes key training metrics including:

* Total training volume
* Weekly workload trends
* Rolling workload averages
* Session tracking using Session IDs

These analytics help visualize **training consistency and workload fluctuation over time**.

---

## 2. Strength Modeling (1RM Prediction)

The system estimates **One Rep Max (1RM)** using historical workout data and applies modeling techniques to forecast strength progression.

Capabilities include:

* Predicted 1RM progression
* Target strength estimation
* Forecasting future strength levels
* Performance comparison between predicted vs actual strength

---

## 3. Fatigue Monitoring System

GymIQ calculates a **Fatigue Ratio** from workout data to detect accumulated fatigue.

The system provides:

* Fatigue trend analysis
* High fatigue warnings
* Deload recommendations

Thresholds are determined using statistical methods such as:

* Mean fatigue level
* Standard deviation thresholds
* Rolling fatigue averages

This helps prevent **overtraining and performance decline**.

---

## 4. Optimal Rest Day Analysis

Using residual analysis from performance models, GymIQ determines which rest durations produce the best results.

Example insight:

> “Better performance observed with 3 days of recovery compared to 2 days.”

This helps optimize recovery strategies for consistent progress.

---

## 5. Personal Records Detection

GymIQ automatically detects new **Personal Records (PRs)** across exercises.

It tracks:

* Historical maximum 1RM
* Exercise-specific PRs
* Session-based strength improvements

This allows athletes to monitor **true strength progression** rather than just lifted weight.

---

## 6. Performance Statistics

The system computes additional statistics such as:

* Rolling performance averages
* Historical comparison metrics
* Progress compared to past sessions

Example metric:

> Average performance in the last 8 sessions vs performance 30 sessions ago.

---

## Tech Stack

* **Python**
* **Pandas** – Data analysis
* **NumPy** – Numerical computations
* **Scikit-learn** – Modeling
* **Plotly** – Interactive visualizations
* **Streamlit** – Application interface

---

## Project Structure

```
GymIQ
│
├── app.py                     # Streamlit application
│
├── models
│   ├── modeling.ipynb         # Strength prediction models
│   ├── personal_stats.py      # Performance statistics
│   └── solid_analytics.ipynb  # Workout analytics
│
├── notebooks
│   ├── GymIQ - Modeling.ipynb
│   ├── GymIQ - Stats.ipynb
│   └── GymIQ - Workout Intelligence System.ipynb
│
├── README.md
└── LICENSE
```
---
## Installation & Dependencies

### 1. Clone the Repository

```bash
git clone https://github.com/AmanPhadke/GymIQ.git
cd GymIQ
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

### 3. Install Required Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Application

GymIQ uses **Streamlit** to provide an interactive interface.

Run the application with:

```bash
streamlit run app.py
```

After running the command, Streamlit will open the application in your browser.

Default local address:

```
http://localhost:8501
```

---

## Deployment

### Deploy on Streamlit Cloud (Recommended)

1. Push the project to a GitHub repository
2. Go to Streamlit Cloud
   https://streamlit.io/cloud
3. Connect your GitHub account
4. Select the repository
5. Choose the entry file:

```
app.py
```

6. Deploy the application

Streamlit will automatically install dependencies and host your app.

---

### Alternative Deployment Options

GymIQ can also be deployed using:

* **Docker**
* **AWS EC2**
* **Render**
* **Heroku**

---

## requirements.txt

The requirements file contains the following dependencies:

```bash
pandas
numpy
plotly
scikit-learn
streamlit
```

they can be installed using

```bash
pip install -r requirements.txt
```

---

## Example Insights Generated

GymIQ can generate insights such as:

* "You are expected to reach a 150kg squat by session 95."
* "High fatigue detected. Consider reducing training intensity."
* "Performance improves when taking 3 rest days between squat sessions."
* "New Personal Record detected for Bench Press."

---

## Future Improvements

Planned improvements include:

* Database integration for workout storage
* User profile and weight-class comparisons
* More Advanced Machine learning models for improved prediction
* AI-based workout recommendations
* Advanced visualization dashboards

---

## Goal of the Project

The goal of GymIQ is to transform raw workout logs into **meaningful performance intelligence**, allowing gym-goers to train smarter using data.

---

## Author

**Aman Phadke**

Computer Science student interested in:

* Machine Learning
* Data Science
* Performance analytics
* AI-driven applications

---

