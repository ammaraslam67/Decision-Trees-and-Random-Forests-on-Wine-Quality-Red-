# Decision Trees and Random Forests on Wine Quality (Red)
**Student:** Ammar Aslam (ID: 24077434)  
**Module:** Machine Learning and Neural Networks — University of Hertfordshire  
**Notebook:** `rf_dt_wine_red_ammar.ipynb`  

---

## 1. Abstract
This tutorial demonstrates a complete, reproducible classification workflow on the **Wine Quality (Red)** dataset. We frame wine quality as a binary task (good vs not good), compare a **Decision Tree** to a **Random Forest**, and explain model behaviour using **feature importance**. The Random Forest achieves higher discrimination (ROC AUC ≈ **0.907**) and better generalisation (test accuracy ≈ **0.913**), while the single tree serves as an interpretable baseline (AUC ≈ **0.726**). All results are generated deterministically with a fixed seed.

---

## 2. Learning outcomes
After working through this notebook, a new learner will be able to:
- Prepare tabular data for binary classification (target creation, train/test split).
- Train and evaluate Decision Trees and Random Forests with scikit-learn.
- Interpret confusion matrices, accuracy, precision/recall/F1 and ROC-AUC.
- Tune key hyperparameters (e.g., `max_depth`, `n_estimators`) via cross-validation.
- Read and communicate **feature importance** and model trade-offs.

---

## 3. Dataset
- **Wine Quality (Red)** — UCI Machine Learning Repository.  
  Direct URL used in the notebook:  
  https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv  
  Harvard citation: Cortez, P., Cerdeira, A., Almeida, F., Matos, T. and Reis, J. (2009) ‘Modeling wine preferences by data mining’, Decision Support Systems, 47(4), pp. 547–553.

If offline access is required, place a copy at `data/winequality-red.csv`. The notebook first looks there and falls back to the URL.

---

## 4. Repository structure
```
repo/
├─ rf_dt_wine_red_ammar.ipynb      # main, fully reproducible notebook
├─ requirements.txt                 # pinned versions for replication
├─ data/
│   └─ winequality-red.csv          # optional local copy (otherwise fetched)
├─ figures_ammar/                   # auto-saved figures on run
│   ├─ 01_class_distribution.png
│   ├─ 02_dt_confusion_matrix.png
│   ├─ 03_dt_roc.png
│   ├─ 04_depth_sweep.png
│   ├─ 05_rf_confusion_matrix.png
│   ├─ 06_rf_roc.png
│   └─ 07_rf_feature_importance.png
└─ docs/
    └─ UH_Cover_Page_Ammar_Aslam_24077434.docx
```

---

## 5. Quick start

### A) Google Colab (fastest)
```python
!git clone https://github.com/ammaraslam67/Decision-Trees-and-Random-Forests-on-Wine-Quality-Red.git
%cd repo
!pip install -r requirements.txt
```
Open `rf_dt_wine_red_ammar.ipynb` in the left pane and **Run all**.

### B) Local (macOS/Windows/Linux)
```bash
git clone https://github.com/ammaraslam67/Decision-Trees-and-Random-Forests-on-Wine-Quality-Red.git
cd repo
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab   # or: jupyter notebook
```
Open `rf_dt_wine_red_ammar.ipynb` → **Kernel → Restart & Run All**.

---

## 6. Reproducibility
- **Seed:** `SEED = 24077434` (student ID) sets NumPy, Python, and hash seeds.
- **Environment:** package versions pinned in `requirements.txt`.
- **Determinism:** scikit-learn’s tree/forest randomness controlled via `random_state=SEED`.

---

## 7. Methods (what the notebook does)
1. **EDA:** shape check, descriptive stats, and **class-imbalance** visual.  
   *Figure placeholder:* `01_class_distribution.png` — “Class Distribution for Binary Wine Quality.”
2. **Baseline model:** **Decision Tree** with `criterion='gini'`.  
   Confusion matrix and ROC curve on the **test set**.  
   *Figures:* `02_dt_confusion_matrix.png`, `03_dt_roc.png`.
3. **Depth study:** sweep `max_depth ∈ {2,3,4,6,8,None}` to show overfitting.  
   *Figure:* `04_depth_sweep.png` (train vs test accuracy curves).
4. **Random Forest:** tuned with **GridSearchCV** on `n_estimators` and `max_depth`.  
   *Figures:* `05_rf_confusion_matrix.png`, `06_rf_roc.png`.
5. **Model explanation:** **feature importance** (Gini importances) with ranked table/bar plot.  
   *Figure:* `07_rf_feature_importance.png`.

---

## 8. Results (exact numbers from this run)
**A. Decision Tree (test set)**  
- Accuracy ≈ **0.889**  
- ROC AUC ≈ **0.726**  
- Confusion matrix (rows = true, cols = pred):

|               | Not good | Good |
|---------------|----------|------|
| **Not good**  | 254      | 23   |
| **Good**      | 20       | 23   |

**B. Depth sweep (train/test accuracy)**

| max_depth | train_acc | test_acc |
|-----------|-----------|----------|
| 2 | 0.8827 | 0.8875 |
| 3 | 0.8890 | 0.8938 |
| 4 | 0.9179 | 0.8906 |
| 6 | 0.9515 | 0.8813 |
| 8 | 0.9750 | 0.8875 |
| None | 1.0000 | 0.8656 |

**C. Random Forest (GridSearchCV)**  
- Best params: `{'max_depth': 10, 'n_estimators': 200}`  
- CV accuracy ≈ **0.906**  
- Test accuracy ≈ **0.9125**  
- ROC AUC ≈ **0.907**  
- Confusion matrix:

|               | Not good | Good |
|---------------|----------|------|
| **Not good**  | 269      | 8    |
| **Good**      | 20       | 23   |

**D. Top-10 feature importances (RF)**

| feature | importance |
|---|---:|
| alcohol | 0.1635 |
| volatile acidity | 0.1188 |
| sulphates | 0.1107 |
| citric acid | 0.0891 |
| density | 0.0865 |
| fixed acidity | 0.0832 |
| total sulfur dioxide | 0.0832 |
| chlorides | 0.0744 |
| residual sugar | 0.0699 |
| free sulfur dioxide | 0.0611 |

---

## 9. Interpretation
The single tree is transparent but unstable; increasing depth fits training data perfectly while test accuracy drops—classic overfitting. The ensemble stabilises variance by averaging many decorrelated trees, hence the **higher AUC and accuracy** and the tighter confusion matrix. **Alcohol**, **volatile acidity**, and **sulphates** emerge as the strongest discriminators of “good” wine.

---

## 10. Accessibility and marking alignment
- High-contrast, colour-blind-safe palettes; labelled axes/legends; text alternatives for each figure.
- The examiner can **Run All** without manual edits; data fetched automatically if not present.
- Clear narrative and code comments; results tables and saved figures for the report.

---

## 11. References
- Breiman, L. (2001) ‘Random forests’, *Machine Learning*, 45(1), pp. 5–32. https://doi.org/10.1023/A:1010933404324  
- Breiman, L., Friedman, J.H., Olshen, R.A. and Stone, C.J. (1984) *Classification and Regression Trees*. New York: Chapman & Hall. https://link.springer.com/book/10.1007/978-1-4899-0537-1  
- Cortez, P., Cerdeira, A., Almeida, F., Matos, T. and Reis, J. (2009) ‘Modeling wine preferences by data mining’, *Decision Support Systems*, 47(4), pp. 547–553. UCI: https://archive.ics.uci.edu/dataset/186/wine+quality  
---

## 12. Licence and academic integrity
Code is MIT-licensed unless your module requires otherwise. Data licensing follows UCI’s terms. This tutorial is original work; all third-party sources are cited.
