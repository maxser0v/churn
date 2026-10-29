# Design decisions worksheet

1. Binary labels and plausible churn prevalence prevent invalid or collapsed targets from producing a misleadingly accurate model. Shuffled labels can still pass validation, so model quality checks are also necessary.

2. Clean recall is 0.754 versus shuffled recall 0.497; the 0.65 floor separates them. F1 sampling interval is 0.389–0.483, half-width 0.047, so tolerance 0.06 allows noise while staying below the clean/broken F1 gap of 0.195; 0.005 overreacts to noise, while 0.30 could admit a useless model.

3. Skipping slices with fewer than 20 churners avoids unstable decisions but can hide harm to underrepresented groups. Report every skip, collect more labels, monitor group outcomes and confidence intervals over longer windows, and require manual review if evidence remains insufficient.

4. Lint and data validation could run independently, and validator/gate tests could overlap training; the two Python versions already run in parallel. Sequential checks stop cheaply on bad code/data, while the gate and service tests require trained outputs and the image waits for both test jobs.

5. Customer behavior or input data can drift; the live service can have outages, latency or resource exhaustion; retention offers can fail to improve business outcomes or harm a customer group despite acceptable offline scores. Monitor service, model, group and business outcomes.

Local acceptance: 28 tests passed, local pipeline GREEN, instructor self-check 40/40. GitHub execution, package publication, failing pull request and branch protection remain pending.
