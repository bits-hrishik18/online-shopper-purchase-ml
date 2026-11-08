# Help Doc

Quick reference for running, deploying and committing this project.

## 1. Set up

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run

| Goal | Command |
|---|---|
| Retrain all five models and rewrite `model/`, `test_data.csv`, `metrics.csv` | `python train_models.py` |
| Start the Streamlit app | `streamlit run app.py` |

Retrain in the same environment that will run the app. Saved models are
pickled scikit-learn pipelines, so a very different scikit-learn version can
fail to load them. If the deployed app reports that it cannot load a model,
retrain and commit the new `model/` folder.

## 3. Using the app

1. The bundled test set loads by default. Upload a CSV in the sidebar to
   replace it.
2. Pick a model from the dropdown.
3. Read the six metrics, then the confusion matrix, classification report and
   ROC curve tabs. **Compare all models** shows the five models side by side.

The CSV must contain these 17 feature columns plus `Revenue`:

```text
Administrative, Administrative_Duration, Informational, Informational_Duration,
ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues,
SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType,
VisitorType, Weekend, Revenue
```

## 4. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Missing required column(s)` | The CSV lacks a feature or `Revenue`. Use `test_data.csv` as the template. |
| `ModuleNotFoundError` on Streamlit Cloud | A package is missing from `requirements.txt`. Add it, commit, push. |
| Model fails to load after deployment | scikit-learn version mismatch. Retrain with `python train_models.py`, commit `model/`. |
| App shows "n/a" for AUC | The uploaded target has only one class, so AUC is undefined. |
| Push rejected for file size | GitHub limits files to 100 MB. The largest file here (`model/knn.joblib`) is about 6 MB. |
| `git push` asks for a password and fails | GitHub no longer accepts account passwords. Create a personal access token (GitHub → Settings → Developer settings) and paste it as the password. |

## 5. BITS Virtual Lab (Prayogshala)

The lab is a Rocky Linux VM. These are the problems classmates reported and
how this project handles them. The lab is worth 1 mark (one screenshot), so
develop and test locally and use the lab only for the final run.

| Problem | What to do |
|---|---|
| Lab time is limited (about 21 hours in total) and can show *Quota Exhausted* or *capacity full* | Keep sessions short. Run everything in one go with `bash setup_vm.sh`. If blocked, retry later, or email the lab contact named in the lab instructions to request more time. |
| Installed packages and workspace are reset when the VM restarts | `setup_vm.sh` reinstalls packages every time. Save your work in GitHub, not only in the lab. |
| `git` is not installed | `sudo dnf install git -y` (Rocky Linux 9; `sudo yum install git` also works). |
| Hard to get code into the VM | Easiest: `git clone https://github.com/<your-username>/<repo-name>.git` (public repo). Otherwise email or Drive the zip to yourself, download it in the lab browser and unzip. |
| No `unzip` command | `python3 -c "import zipfile; zipfile.ZipFile('ML_Assignment_2_Project.zip').extractall('.')"` |
| Jupyter kernel stays idle or will not start | Not needed. The code is plain `.py` files, so run `bash setup_vm.sh` in the lab terminal. |
| The deployed Streamlit Cloud app will not load inside the lab browser | Do not rely on it there. Run the app locally in the lab with `bash setup_vm.sh app` and open `http://localhost:8501`. |
| Pip fails on an old lab Python | `setup_vm.sh` installs only what training needs, using `--user`, so pip picks versions that match the lab's Python. |

Steps for the lab run:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
bash setup_vm.sh
```

`setup_vm.sh` writes its output to `~/ml_assignment_run`, so the models you
committed to GitHub are not overwritten. The terminal prints the lab's OS,
host, Python and library versions before the metrics table, so one screenshot
of the terminal shows both the environment and the results. Do not commit
anything generated in the lab.

## 6. Suggested commit plan

Commit as you actually finish each step, in your own words. Run the command
for a step only when you have done and checked that step. Set your own Git
identity first and do not backdate commits.

```bash
git init
git branch -M main
git config user.name  "<your name>"
git config user.email "<your personal GitHub email>"
```

| # | Step | Files | Example commit message |
|---:|---|---|---|
| 1 | Project skeleton | `.gitignore`, `.gitattributes`, `requirements.txt` | `Add gitignore, line-ending rule and dependency list` |
| 2 | Dataset | `data/online_shoppers_intention.csv` | `Add UCI online shoppers dataset` |
| 3 | Shared helpers | `utils.py` | `Add feature lists, validation and metric helpers` |
| 4 | Training scripts | `train_models.py`, `setup_vm.sh` | `Add training script and lab setup script` |
| 5 | Trained artefacts | `model/` (including `model_info.json`), `test_data.csv`, `metrics.csv` | `Add trained models, test split and metrics` |
| 6 | Streamlit app | `app.py` | `Add Streamlit app for model evaluation` |
| 7 | Documentation | `README.md`, `help_doc.md` | `Add README and help doc` |
| 8 | After deployment | `README.md` | `Add repository and live app links` |

Example for step 1 (repeat the pattern for the other steps):

```bash
git add .gitignore requirements.txt
git commit -m "Add gitignore and dependency list"
```

Between steps, run the code you just added (for example, run
`python train_models.py` before commit 5, and `streamlit run app.py` before
commit 6) so each commit reflects work that actually runs.

Connect and push to your own GitHub repository:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## 7. Deploy on Streamlit Community Cloud

These are the problems classmates hit and their fixes:

- **Build hangs or the app never loads.** Streamlit Cloud defaults to a very
  new Python (3.14 in recent reports). Under **Advanced settings**, choose
  **Python 3.12** when you create the app. Keep `requirements.txt` free of
  pinned versions, as it is here; a pinned scikit-learn can force a slow
  compile from source.
- **Sign-in with GitHub is blocked or loads forever.** Use a personal GitHub
  account, not a work or BITS one. Create the Streamlit account with an email
  address first, then link GitHub. Try a private/incognito window, and turn
  off ad blockers.
- **You cannot see the logs.** The *Manage app* button appears only when you
  are signed in; open the deployment log from there.
- **A blank page for 20-30 seconds, or a sleeping app.** This is normal on the
  free tier. Opening the link wakes the app, and the bundled test set shows
  results straight away.
- **Models fail to load online.** The warning banner in the app compares the
  scikit-learn version used for training with the one running online. If they
  differ, retrain locally under Python 3.12, commit `model/`, and redeploy.

Steps:

1. Open <https://streamlit.io/cloud> and sign in.
2. Choose **New app**, select your repository and the `main` branch.
3. Set the main file to `app.py`, open **Advanced settings**, pick **Python 3.12**, and click **Deploy**.
4. Open the live link and check that every model loads. Then paste both links
   into `README.md` (commit 8).

## 8. Submission checklist

- [ ] GitHub repository link opens (repo is public)
- [ ] Streamlit app opens and shows the interactive frontend
- [ ] Exactly one screenshot of the run on the BITS Virtual Lab
- [ ] `README.md` content is included in the submitted PDF
- [ ] Single PDF in this order: GitHub link, Streamlit link, screenshot, README content
- [ ] Your name and BITS ID appear in the PDF only, not in the public repo or app
