# 💰 LendingClub Policy Optimization
*A comparative study of supervised default prediction and offline reinforcement learning for credit approval.*

---

## 🧩 Project Structure

```text
lendingclub-policy-optimization/
│
├── notebooks/
│   ├── 01_eda.ipynb              # Data exploration and visualization
│   ├── 02_supervised_model.ipynb # Deep learning classifier (AUC/F1)
│   └── 03_offline_rl.ipynb       # Conservative Q-Learning (CQL) + FQE evaluation
│
├── src/
│   ├── data_utils.py             # Data loading, cleaning, splitting
│   ├── features.py               # Preprocessing & transformers
│   ├── profit_threshold.py       # Reward functions and threshold logic
│   └── rl_cql.py                 # RL agent, MDP dataset, and evaluation
│
├── requirements.txt
├── README.md
└── reports/
    └── LendingClub_Policy_Optimization_Report.pdf


---

## ⚙️ Environment Setup
### 1️⃣ Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # (Mac/Linux)
````

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:

* `pandas`, `numpy`, `scikit-learn`
* `matplotlib`, `seaborn`
* `d3rlpy==2.x`, `joblib`, `pathlib`

---

## 🧠 Running the Notebooks

1. **EDA:** Explore & preprocess LendingClub dataset (`01_eda.ipynb`).
2. **Supervised Model:** Train classifier, compute AUC/F1 (`02_supervised_model.ipynb`).
3. **Offline RL:** Build MDP dataset, train CQL agent, estimate policy value (`03_offline_rl.ipynb`).

All relative paths are consistent; update `DATA_PATH` if CSV is elsewhere.

---

## 📊 Key Results

| Metric                     | Model          | Value                 |
| -------------------------- | -------------- | --------------------- |
| **AUC**                    | Deep Learning  | ~0.83                 |
| **F1-Score**               | Deep Learning  | ~0.71                 |
| **Estimated Policy Value** | RL Agent (CQL) | ≈ –3.3 × 10³ per loan |

---

## 🧩 Insights

* DL model → accurate default risk ranking.
* RL agent → directly optimizes for profit but mimics approve-all policy due to missing denied samples.
* Reward design and action coverage are crucial for successful offline RL.

---

## 🚀 Future Work

* Collect denied-loan data for counterfactual learning.
* Improve reward realism (recovery, costs).
* Test newer offline RL algorithms (IQL, BCQ).
* Combine DL scoring + RL policy layers.

```



