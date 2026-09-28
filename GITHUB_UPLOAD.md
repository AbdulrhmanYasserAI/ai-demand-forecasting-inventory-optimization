# Upload to GitHub

Repository name:

`ai-demand-forecasting-inventory-optimization`

Your GitHub profile:

https://github.com/AbdulrhmanYasserAI

## Terminal upload

From this project folder:

```bash
git init
git add .
git commit -m "Build AI demand forecasting and inventory optimization project"
git branch -M main
git remote add origin https://github.com/AbdulrhmanYasserAI/ai-demand-forecasting-inventory-optimization.git
git push -u origin main
```

If GitHub asks for authentication, use your GitHub credentials/token through Git Credential Manager.

## Before publishing

- Keep `data/demo/demo_demand.csv` in the repository.
- Do not upload the full Walmart competition dataset into `data/raw/`.
- Run `python run_project.py` once and keep the generated report files.
- After downloading the real Walmart data, you can run `python run_project.py --real-data` locally.
