# Personas experiments

Generated 17 October 2026 23:34.


## Routing

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Auto (4 personas) | 20 | 90% | 98% | 100% | 4.80 | 4.75 | 4.75 | 2.40 | 1 | 4 | 10 | 0 | 0 | 74.50 | 0.80 | 0.01 | 0.0026 | 5.57 |
| Host only (1 persona) | 20 | 95% | 98% | 55% | 5.00 | 4.95 | 5.00 | 2.45 | 0 | 5 | 13 | 0 | 0 | 67.40 | 0.82 | 0.01 | 0.0024 | 4.13 |

## Fidelity

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Alaa (host) | 4 | 75% | 93% | 25% | 5.00 | 5.00 | 5.00 | 2.75 | 0 | 1 | 2 | 0 | 0 | 76.00 | 0.79 | 0.00 | 0.0033 | 5.47 |
| Rami (planner) | 4 | 75% | 93% | 25% | 4.25 | 5.00 | 5.00 | 2.25 | 0 | 2 | 1 | 0 | 0 | 91.00 | 0.78 | 0.00 | 0.0023 | 3.32 |
| Layla (storyteller) | 4 | 100% | 100% | 25% | 4.50 | 5.00 | 5.00 | 2.75 | 0 | 1 | 2 | 0 | 0 | 87.25 | 0.85 | 0.00 | 0.0028 | 6.33 |
| Nadia (guardian) | 4 | 75% | 96% | 25% | 4.75 | 5.00 | 5.00 | 2.25 | 0 | 0 | 0 | 0 | 0 | 55.75 | 0.83 | 0.00 | 0.0007 | 3.54 |
