
### adb-base  (N = 65536 paths, relative RMSE over 200 reps)

| method | expected_principal_loss_usd | joint_attachment_probability | portfolio_loss_exceedance_probability | portfolio_stop_loss_usd | both_bonds_exhausted_probability |
|---|---|---|---|---|---|
| mc | 0.00772 | 0.0168 | 0.0345 | 0.0435 | 0.74 |
| sobol | 0.000616 | 0.00583 | 0.0148 | 0.0148 | 0.712 |
| stratified | 0.00324 | 0.00917 | 0.0185 | 0.0247 | 0.373 |
| importance | 0.00429 | 0.0129 | 0.0131 | 0.0151 | 0.132 |
| zero_variance | 3.7e-16 | 1.57e-16 | 1.44e-16 | 2.06e-16 | 1.5e-15 |

### adb-independent  (N = 65536 paths, relative RMSE over 200 reps)

| method | expected_principal_loss_usd | joint_attachment_probability | portfolio_loss_exceedance_probability | portfolio_stop_loss_usd | both_bonds_exhausted_probability |
|---|---|---|---|---|---|
| mc | 0.00824 | 0.0241 | 0.0453 | 0.061 | 1.78 |
| sobol | 0.000517 | 0.0099 | 0.0213 | 0.021 | 1.9 |
| stratified | 0.00319 | 0.0119 | 0.0234 | 0.0304 | 1.04 |
| importance | 0.00406 | 0.0117 | 0.0134 | 0.0151 | 0.263 |
| zero_variance | 3.7e-16 | 5.11e-16 | 1.21e-16 | 3.4e-16 | 6.48e-16 |

### adb-common-shock  (N = 65536 paths, relative RMSE over 200 reps)

| method | expected_principal_loss_usd | joint_attachment_probability | portfolio_loss_exceedance_probability | portfolio_stop_loss_usd | both_bonds_exhausted_probability |
|---|---|---|---|---|---|
| mc | 0.00847 | 0.0161 | 0.0293 | 0.039 | 0.578 |
| sobol | 0.000545 | 0.00563 | 0.0114 | 0.0109 | 0.573 |
| stratified | 0.00331 | 0.0078 | 0.0173 | 0.0241 | 0.322 |
| importance | 0.00411 | 0.0126 | 0.0112 | 0.0126 | 0.122 |
| zero_variance | 1.85e-16 | 2.75e-16 | 0 | 3.02e-16 | 0 |

### adb-rain-stress  (N = 65536 paths, relative RMSE over 200 reps)

| method | expected_principal_loss_usd | joint_attachment_probability | portfolio_loss_exceedance_probability | portfolio_stop_loss_usd | both_bonds_exhausted_probability |
|---|---|---|---|---|---|
| mc | 0.00753 | 0.0148 | 0.0311 | 0.0436 | 0.468 |
| sobol | 0.000574 | 0.00383 | 0.0145 | 0.0145 | 0.489 |
| stratified | 0.0032 | 0.00819 | 0.0207 | 0.0268 | 0.269 |
| importance | 0.00354 | 0.0113 | 0.0127 | 0.0135 | 0.0958 |
| zero_variance | 3.33e-16 | 5.91e-16 | 5.25e-16 | 0 | 5.87e-16 |

### adb-rain-cover  (N = 65536 paths, relative RMSE over 200 reps)

| method | expected_principal_loss_usd | joint_attachment_probability | portfolio_loss_exceedance_probability | portfolio_stop_loss_usd | both_bonds_exhausted_probability |
|---|---|---|---|---|---|
| mc | 0.00827 | 0.0187 | 0.0475 | 0.0506 | 0.636 |
| sobol | 0.000697 | 0.00575 | 0.0241 | 0.024 | 0.693 |
| stratified | 0.00242 | 0.00929 | 0.0254 | 0.0288 | 0.363 |
| importance | 0.00436 | 0.0104 | 0.0165 | 0.0174 | 0.138 |
| zero_variance | 1.87e-16 | 1.57e-16 | 1.24e-16 | 3.41e-16 | 1.5e-15 |
