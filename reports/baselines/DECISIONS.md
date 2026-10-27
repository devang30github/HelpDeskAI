# V1 baseline decisions

## 2026-10-27: final test run
- r1 (`20261025-164815-v1-baseline`): one generation (q009) failed because the 2,048-token completion
  budget was consumed by reasoning (`finish_reason=length`), and the RAGAS stage first failed on a code
  error that has since been fixed.
- Decision, made before seeing any new numbers: re-run the complete test split once as
  `v1-baseline-r2` with `LLM_MAX_TOKENS=4096`. **r2 replaces r1 as the V1 baseline whatever its
  numbers are.** r1 is kept for the record.
- Nothing else changes: same dataset labels, KB pin, retriever, prompt (v1), generator model and judge.
- V2 must run with the same `LLM_MAX_TOKENS`.
- After r2 there are no further V1 test runs. The only allowed exception is `--rescore --ragas` to
  recover from an infrastructure error, and any use of it is noted here.
