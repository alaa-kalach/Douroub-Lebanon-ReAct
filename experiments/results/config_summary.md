# Config experiments

Generated 17 October 2026 23:35.


## Temperature

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) | answer diversity | same numbers every time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| temperature 0.0 | 9 | 100% | 100% | 100% | 4.33 | 4.78 | 4.89 | 3.33 | 0 | 3 | 6 | 0 | 0 | 173 | 0.65 | 0.05 | 0.0058 | 11.52 | 0.64 | 67% |
| temperature 0.6 | 9 | 100% | 100% | 100% | 4.56 | 4.89 | 4.89 | 3.33 | 0 | 1 | 6 | 0 | 0 | 155 | 0.68 | 0.04 | 0.0051 | 10.49 | 0.67 | 67% |
| temperature 1.1 | 9 | 100% | 100% | 100% | 4.67 | 5.00 | 5.00 | 3.33 | 0 | 2 | 6 | 0 | 0 | 173 | 0.68 | 0.05 | 0.0046 | 10.72 | 0.68 | 67% |
| temperature 1.6 | 9 | 100% | 100% | 100% | 4.67 | 5.00 | 5.00 | 3.33 | 0 | 1 | 6 | 0 | 0 | 163 | 0.67 | 0.05 | 0.0046 | 10.40 | 0.62 | 33% |

## Top_P

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) | answer diversity | same numbers every time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top_p 0.3 | 4 | 100% | 100% | 100% | 5.00 | 5.00 | 5.00 | 3.00 | 0 | 1 | 4 | 0 | 0 | 184 | 0.65 | 0.05 | 0.0047 | 11.45 | 0.74 | 50% |
| top_p 0.7 | 4 | 100% | 100% | 100% | 4.75 | 5.00 | 5.00 | 3.00 | 0 | 2 | 4 | 0 | 0 | 197 | 0.64 | 0.05 | 0.0052 | 13.64 | 0.77 | 50% |
| top_p 1.0 | 4 | 100% | 100% | 100% | 5.00 | 5.00 | 5.00 | 3.00 | 0 | 1 | 4 | 0 | 0 | 218 | 0.65 | 0.06 | 0.0055 | 14.07 | 0.68 | 100% |

## Penalties

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) | answer diversity | same numbers every time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no penalties | 4 | 100% | 100% | 100% | 5.00 | 5.00 | 5.00 | 3.25 | 1 | 0 | 4 | 0 | 0 | 235 | 0.63 | 0.09 | 0.0060 | 7.42 | 0.73 | 100% |
| Layla's penalties (0.6 / 0.3) | 4 | 100% | 100% | 100% | 4.00 | 5.00 | 5.00 | 2.75 | 0 | 1 | 3 | 0 | 0 | 212 | 0.63 | 0.05 | 0.0048 | 11.63 | 0.80 | 50% |
| high penalties (1.5 / 1.0) | 4 | 100% | 100% | 100% | 4.50 | 4.75 | 4.75 | 3.25 | 1 | 1 | 4 | 0 | 0 | 160 | 0.72 | 0.02 | 0.0050 | 12.20 | 0.78 | 50% |

## Max Tokens

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max_completion_tokens 150 | 3 | 100% | 100% | 100% | 3.67 | 2.33 | 4.67 | 3.00 | 0 | 0 | 1 | 0 | 3 | 67.33 | 0.79 | 0.00 | 0.0029 | 6.00 |
| max_completion_tokens 400 | 3 | 100% | 100% | 100% | 4.00 | 5.00 | 5.00 | 3.00 | 0 | 0 | 1 | 0 | 0 | 130 | 0.72 | 0.01 | 0.0034 | 6.43 |
| max_completion_tokens 900 | 3 | 100% | 100% | 100% | 4.33 | 5.00 | 5.00 | 3.33 | 1 | 0 | 1 | 0 | 0 | 126 | 0.74 | 0.01 | 0.0039 | 6.53 |

## Model

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-5.4-nano | 6 | 83% | 98% | 100% | 5.00 | 4.83 | 5.00 | 2.17 | 0 | 0 | 0 | 0 | 0 | 65.00 | 0.83 | 0.03 | 0.0010 | 3.81 |
| gpt-5.4-mini | 6 | 83% | 98% | 100% | 4.83 | 4.83 | 4.83 | 2.83 | 0 | 1 | 4 | 0 | 0 | 77.83 | 0.80 | 0.01 | 0.0035 | 4.64 |
| gpt-6-luna | 6 | 83% | 95% | 100% | 4.67 | 4.67 | 5.00 | 2.83 | 0 | 1 | 5 | 0 | 0 | 50.33 | 0.88 | 0.00 | 0.0005 | 5.52 |
| gpt-4.1-mini | 6 | 83% | 98% | 100% | 4.83 | 4.83 | 5.00 | 2.33 | 0 | 1 | 0 | 0 | 0 | 69.00 | 0.85 | 0.00 | 0.0023 | 3.11 |
| gpt-4o-mini | 6 | 50% | 86% | 100% | 4.33 | 4.33 | 4.67 | 1.83 | 0 | 4 | 1 | 0 | 0 | 73.00 | 0.85 | 0.00 | 0.0009 | 2.70 |
| gpt-6.1-sol | 6 | 83% | 98% | 100% | 4.83 | 4.83 | 4.83 | 2.50 | 0 | 0 | 1 | 0 | 0 | 64.00 | 0.85 | 0.00 | 0.0091 | 12.81 |

## Reasoning Effort

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| effort none (sampling sent) | 4 | 100% | 100% | 100% | 4.50 | 5.00 | 5.00 | 2.75 | 0 | 0 | 1 | 0 | 0 | 85.00 | 0.72 | 0.01 | 0.0027 | 4.38 |
| effort low (sampling dropped) | 4 | 100% | 100% | 100% | 4.75 | 5.00 | 5.00 | 3.50 | 4 | 0 | 0 | 0 | 0 | 88.50 | 0.70 | 0.03 | 0.0075 | 6.49 |

## Stop Sequence

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini, parser only | 4 | 75% | 96% | 100% | 4.25 | 5.00 | 5.00 | 2.50 | 0 | 0 | 0 | 0 | 0 | 66.00 | 0.83 | 0.00 | 0.0017 | 3.38 |
| gpt-4.1-mini + stop 'Observation:' | 4 | 75% | 96% | 100% | 4.75 | 5.00 | 5.00 | 2.75 | 0 | 0 | 1 | 0 | 0 | 71.25 | 0.81 | 0.00 | 0.0022 | 4.04 |
