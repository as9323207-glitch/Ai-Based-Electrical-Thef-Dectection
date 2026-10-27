# Ai-Based-Electrical-Thef-Dectection
# (This  Project Will Help To Identify Therf Who Take The Electricity From Unwanted Wires)



# Electricity Theft Detection Demo

A small C program that trains a logistic regression model from labeled CSV data.

## CSV format

The CSV must contain a header followed by rows with six columns:

`label,feature1,feature2,feature3,feature4,feature5`

- `label`: `0` for normal or `1` for confirmed theft
- `feature1`–`feature5`: numeric measurements

## Compile

```sh
gcc theft_model.c -o theft_model -lm
