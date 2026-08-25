# Raw experimental data

* `HPLC_data.xlsx` — HPLC calibration curves, method validation (linear
  range, recovery, LOD = 1.25–1.31 mg/L, LOQ = 3.78–3.96 mg/L), replicate peak
  areas, and control experiments extracted from the wet-lab supplementary
  material. Regenerate with `build_hplc_xlsx.py` from the source document.
* `kinetics_raw_data.xlsx` — raw initial rates (three replicates) for the
  eight active candidates at seven p-aminophenol concentrations, Michaelis–
  Menten fits (Origin, Levenberg–Marquardt), kinetic constants (Km, Vmax,
  kcat) with 95% CIs, residuals, and fitting methodology.

Fermentation product concentrations (p-aminophenol consumption and
acetaminophen production per strain) are reported in the manuscript Fig. 5
and the supplementary HPLC tables; the underlying replicate values are in
`HPLC_data.xlsx`.
