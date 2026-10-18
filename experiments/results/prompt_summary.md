# Prompt experiments

Generated 18 October 2026 16:17.


## Chain Of Thought

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) | answer diversity | same numbers every time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline: CoT, one-shot, system | 16 | 94% | 99% | 100% | 4.69 | 4.88 | 5.00 | 2.69 | 0 | 2 | 9 | 0 | 0 | 68.62 | 0.79 | 0.00 | 0.0027 | 5.71 | 0.49 | 88% |
| no CoT (actions only) | 16 | 94% | 99% | 100% | 4.75 | 4.88 | 5.00 | 2.75 | 0 | 10 | 10 | 0 | 0 | 70.12 | 0.81 | 0.00 | 0.0024 | 4.62 | 0.58 | 75% |

## Examples

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) | answer diversity | same numbers every time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline: CoT, one-shot, system | 16 | 94% | 99% | 100% | 4.69 | 4.88 | 5.00 | 2.69 | 0 | 2 | 9 | 0 | 0 | 68.62 | 0.79 | 0.00 | 0.0027 | 5.71 | 0.49 | 88% |
| zero-shot | 16 | 88% | 98% | 100% | 4.88 | 4.88 | 4.88 | 2.38 | 0 | 2 | 4 | 0 | 0 | 72.38 | 0.79 | 0.01 | 0.0030 | 4.30 | 0.54 | 100% |
| few-shot (4 examples) | 16 | 100% | 100% | 100% | 4.81 | 4.81 | 4.75 | 2.62 | 0 | 2 | 9 | 0 | 0 | 66.38 | 0.81 | 0.01 | 0.0035 | 4.01 | 0.50 | 88% |

## Placement

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) | answer diversity | same numbers every time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline: CoT, one-shot, system | 16 | 94% | 99% | 100% | 4.69 | 4.88 | 5.00 | 2.69 | 0 | 2 | 9 | 0 | 0 | 68.62 | 0.79 | 0.00 | 0.0027 | 5.71 | 0.49 | 88% |
| instructions in the message | 16 | 94% | 99% | 100% | 4.75 | 4.94 | 4.94 | 2.12 | 0 | 3 | 2 | 0 | 0 | 65.81 | 0.81 | 0.01 | 0.0030 | 3.57 | 0.62 | 75% |

## Instructions

| arm | runs | success rate | checks passed | persona as expected | judge: persona | judge: helpful | judge: grounded | model calls | format errors | format slips | grounding checks | invented observations | cut off by max tokens | words | lexical variety | repeated trigrams | cost per run ($) | latency (s) | answer diversity | same numbers every time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline: CoT, one-shot, system | 16 | 94% | 99% | 100% | 4.69 | 4.88 | 5.00 | 2.69 | 0 | 2 | 9 | 0 | 0 | 68.62 | 0.79 | 0.00 | 0.0027 | 5.71 | 0.49 | 88% |
| no instructions block | 16 | 62% | 93% | 100% | 4.88 | 4.75 | 4.75 | 2.50 | 0 | 12 | 10 | 0 | 0 | 70.06 | 0.79 | 0.00 | 0.0027 | 3.98 | 0.53 | 88% |
