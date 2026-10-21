# EduPredict AI

**Building AI course project** — Early Warning System for University Student Dropout Prevention

## Summary

EduPredict AI is a prototype that uses machine learning to identify patterns associated with university dropout risk in synthetic student records and support timely, human-led academic interventions.

## Background

Universities may identify disengagement only after students have missed classes, failed assessments, or stopped submitting assignments. This limits opportunities to provide support early.

EduPredict AI explores how an early-warning tool could help academic tutors prioritize outreach. It addresses:

- Delayed detection of academic difficulties and disengagement.
- Academic monitoring based on fragmented indicators.
- Limited capacity to prioritize personalized tutoring at scale.

**Why this matters:** timely, supportive conversations may help institutions understand barriers to participation before they become harder to address. A predicted risk is not evidence that a student will drop out.

## Data and AI techniques

The demonstration uses **synthetic data only**. It generates illustrative student-level features:

| Feature | Meaning |
| --- | --- |
| `attendance_pct` | Class attendance, from 0 to 100% |
| `average_grade` | Average grade on a 0–20 scale |
| `assignment_completion_pct` | Submitted assignments, from 0 to 100% |
| `lms_activity_count` | Illustrative number of learning-platform interactions |
| `previous_grade` | Prior average grade on a 0–20 scale |

The simulated binary label (`dropout`) is generated from a hypothetical relationship between these measures and a random component. **It is not drawn from real student outcomes.**

The prototype compares two supervised classifiers: **logistic regression** and **k-nearest neighbors (k-NN)**. It uses a stratified training/test split and reports accuracy, precision, recall, F1 score, ROC AUC and confusion matrices where those metrics are defined. Classifiers are evaluated on held-out synthetic cases, not on real students.

The data-generating assumptions intentionally influence the results. Consequently, performance metrics on synthetic data **must not** be interpreted as evidence of effectiveness, generalizability or fairness in a university.

## How is it used?

1. Run `python src/demo.py` after installing dependencies.
2. The script creates a reproducible illustrative dataset, trains and evaluates the two models, and prints a small set of example predictions.
3. An academic tutor could use a future validated deployment as **decision support**, not as an automated decision system: staff would review indicators, contact students respectfully and offer resources.
4. Students should have appropriate notice, safeguards, and a way to contest or correct their information before any real-world deployment.

**Users:** academic tutors, student-support teams, learning analytics researchers and, through appropriate reporting, students themselves.

**Beneficiaries:** students who might benefit from early, voluntary academic support.

## Challenges

- **Data protection:** student records are sensitive and require a lawful basis, access controls, minimization and retention rules.
- **Bias and fairness:** attendance and LMS activity can reflect disability, employment, connectivity and other circumstances rather than motivation.
- **Prediction errors:** false positives could stigmatize students; false negatives could miss people needing help.
- **Data leakage and drift:** training features must be available at the time of prediction and checked across academic terms.
- **Causal limitations:** predictions alone do not establish why dropout occurs or whether an intervention would help.
- **Deployment gap:** the demo has no real-data integration, dashboard, external validation, user study or demonstrated intervention impact.

No automated admission, grading, discipline or exclusion decisions should be based on the model.

## What next?

1. Collaborate with academic and student stakeholders to define a responsible use case and evaluation protocol.
2. Obtain institutional approvals and an appropriate privacy and ethical review before working with real data.
3. Validate data quality, representation, calibration and subgroup performance on carefully governed historical records.
4. Test interpretable baselines and assess whether alerts improve supportive interventions in a prospective study.
5. Consider an opt-in LMS integration and an accessible tutor dashboard only after validation.

## Acknowledgments

- **Building AI**, an educational course by the University of Helsinki and MinnaLearn, motivated the project and its methods.
- The prototype uses Python, NumPy, pandas and scikit-learn (see `requirements.txt`).
- All demonstration records are artificially generated. No real student data, institutional endorsement or measured dropout-prevention effect is claimed.

## Run the demo

Requires Python 3.10+.

```bash
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/demo.py
```

The script uses a fixed random seed for reproducibility. It does not connect to an LMS or upload any data.

## Repository structure

```text
EduPredict-AI/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── demo.py
```

## Responsible-use note

This is a **concept demonstration and educational prototype**, not a validated tool for screening individual students. Results should never be used to label, penalize or deny services to a person.
