Put the WFP file here as:  wfp_food_prices_ken.csv

Download "Kenya - Food Prices" (the 3.5M CSV) from
https://data.humdata.org/dataset/wfp-food-prices-for-kenya

samples/cbk_sample_2month.csv is a two-month CBK USD/KES sample (Oct-Nov
2026), retained to reproduce the CBK-versus-WFP rate validation
(python -m src.drivers --inspect --path data/samples/cbk_sample_2month.csv).
Do not point the loader at it: config.FX_CSV must be a full CBK export.
