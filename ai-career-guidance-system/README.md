# AI Career Guidance Platform

An interactive AI-powered career guidance platform that helps students explore suitable careers, educational pathways, skill gaps, and actionable learning plans. The application combines machine-learning recommendations, occupational data, and explainability features in a Streamlit interface.

## Features

- Career recommendations using a hybrid machine-learning ensemble.
- Career exploration for students after Class 10, Class 12, and higher education.
- Interest and aptitude assessment with explainable recommendation factors.
- Skill-gap analysis and personalized learning roadmaps.
- Occupational information based on O*NET data.
- Model metrics, comparisons, and retraining tools.
- Downloadable assessment reports.

## Requirements

- Python 3.10 or newer.
- pip.
- The Python packages listed in `requirements.txt`.

The ZIP submission intentionally does not include the large pre-trained model bundle or generated synthetic training CSV. This keeps the submission compact and avoids shipping redundant generated artifacts. Both are reproducible using the training command below.

## Run locally

### 1. Extract the project and open its folder

```bash
cd ai-career-guidance-system
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Generate the training dataset and train the ML models

```bash
python train_models.py
```

If `data/synthetic_student_profiles.csv` is not present, the script automatically generates 5,000 synthetic student profiles. It then trains and evaluates the Random Forest, Gradient Boosting, MLP Neural Network, calibrated SVM, and hybrid soft-voting ensemble. The trained model bundle is saved to `models/career_models.joblib`.

Training is required to enable the supervised multi-model predictions and local model explanations. If you skip training, the application can still start and use its O*NET KNN plus profile-scoring fallback, but the trained ensemble will not be available.

### 5. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal (normally http://localhost:8501).

Convenience launchers are also provided: `start.bat` for Windows and `start.sh` for macOS/Linux.

## Dataset details and download links

### O*NET occupational dataset

The project includes the selected O*NET CSV tables required by the current application in `data/onet/`. The official source and the latest downloadable CSV/ZIP database are available here:

- **O*NET Database downloads:** https://www.onetcenter.org/database.html
- **O*NET database releases archive:** https://www.onetcenter.org/db_releases.html
- **O*NET database license:** https://www.onetcenter.org/license_db.html

The included files are selected tables from the O*NET 31.0 database. They are retained in the submission so the application can run immediately without a separate occupational-data download. If you replace them with a newer release, preserve the filenames and CSV column headers expected by `data_loader.py` and `career_engine.py`.

### Synthetic student-profile dataset

The generated training CSV is not included because it can be recreated from source code. Run:

```bash
python train_models.py
```

The script calls `generate_synthetic_student_dataset(n_samples=5000, ...)` automatically when `data/synthetic_student_profiles.csv` is missing. The generator source is `dataset_generator.py`.

## Run tests

From the project root, with the virtual environment activated:

```bash
python -m pytest -q
```

## Project structure

```text
ai-career-guidance-system/
├── app.py                  # Streamlit application
├── assessment.py           # Assessment and student profile logic
├── career_engine.py        # Career scoring and recommendation engine
├── data_loader.py          # Dataset loading and validation
├── dataset_generator.py    # Synthetic dataset generation
├── explainability.py       # Recommendation explanations
├── ml_models.py            # Machine-learning models and inference
├── ml_recommender.py       # Occupational similarity recommender
├── pathways_engine.py      # Education and skill-gap pathways
├── train_models.py         # Model training and evaluation
├── requirements.txt        # Python dependencies
├── models/                 # Model metrics; trained bundle generated locally
├── data/                   # Career catalogs and selected O*NET tables
└── tests/                  # Automated tests
```

## Docker (optional)

If Docker is installed:

```bash
docker build -t ai-career-guidance .
docker run --rm -p 8501:8501 ai-career-guidance
```

Then open http://localhost:8501.

## Troubleshooting

- **`streamlit` is not recognized:** activate the virtual environment and run `python -m streamlit run app.py`.
- **Model file is missing:** run `python train_models.py` from the project root. This generates the synthetic training data (if needed) and saves the model bundle.
- **Missing O*NET data:** verify that the CSV files in `data/onet/` were extracted alongside the application. The included data is sufficient for the submitted version.
- **Model/scikit-learn compatibility:** the generated model is created using the scikit-learn version installed from `requirements.txt`.

## License and attribution

The included O*NET 31.0 occupational tables are from the National Center for O*NET Development. O*NET database content is provided under the Creative Commons Attribution 4.0 International license. See https://www.onetcenter.org/license_db.html and `data/onet/README.md`. The synthetic student profiles are generated by this project's `dataset_generator.py` code.
