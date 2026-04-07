# Carbon Risk Premium — Bolton & Kacperczyk Replication

## Research Question
Do investors demand higher returns for high-carbon stocks?

## Methodology
**Language:** Python  
**Methods:** Fama-MacBeth regressions, portfolio sorts

## Data
Yahoo Finance (39 firms, monthly 2018–2025), sector emission proxies

## Key Findings
Directional evidence of carbon premium; high-carbon quintile shows different return patterns vs low-carbon.

## How to Run
```bash
pip install -r requirements.txt
python code/project19_*.py
```

## Repository Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── code/          ← Analysis scripts
├── data/          ← Raw and processed data
└── output/
    ├── figures/   ← Charts and visualizations
    └── tables/    ← Summary statistics and regression results
```

## Author
Alfred Bimha

## License
MIT

---
*Part of a 20-project sustainable finance research portfolio.*
