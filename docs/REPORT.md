# AAS bench report

Generated 2026-10-26T18:54:47.309Z from bench run 2026-10-26T10:27:44.043Z (with the ListenBrainz map). 13 people, each split by time: trained on everything before their last 8 weeks, scored on those weeks. Doors and traits use training weeks only.

How to read it. Each line pairs a model with a baseline over the same people. Medians come with a 95% percentile bootstrap interval over people (4,000 resamples). The difference is model minus baseline, signed so that positive means the model is better. The sign test is exact and two-sided, with ties dropped. Holm adjusts the sign-test p-values across every comparison in the table. A result survives when the Holm-adjusted p is below 0.05 and the bootstrap interval for the mean difference sits above zero. With 13 people the sign test cannot reach 0.05 unless at least 10 of 13 people agree (10 to 3 gives p = 0.092; 11 to 2 gives p = 0.022), so a small consistent effect can fail here and still be real.

## Paired comparisons

| comparison | n | model median [95%] | baseline median [95%] | median difference [95%] | mean difference [95%] | wins/losses/ties | sign p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **1 finish or skip (AUC)** | | | | | | | | | |
| skip model vs artist skip rate | 13 | 0.848 [0.741, 0.861] | 0.617 [0.584, 0.649] | 0.205 [0.169, 0.268] | 0.204 [0.161, 0.241] | 13/0/0 | <0.001 | 0.009 | survives |
| **2 next 8 weeks' artists (NDCG@10)** | | | | | | | | | |
| decay vs all-time | 13 | 0.813 [0.684, 0.919] | 0.801 [0.600, 0.847] | 0.040 [-0.000, 0.123] | 0.060 [-0.002, 0.129] | 7/4/2 | 0.549 | 1.000 | no difference |
| decay vs last 4 weeks | 13 | 0.813 [0.684, 0.919] | 0.711 [0.627, 0.879] | 0.026 [0.000, 0.071] | 0.042 [-0.001, 0.090] | 9/3/1 | 0.146 | 1.000 | no difference |
| **3 the game (share correct)** | | | | | | | | | |
| old scorer vs chance | 13 | 0.687 [0.635, 0.820] | 0.200 | 0.487 [0.435, 0.620] | 0.491 [0.382, 0.581] | 12/1/0 | 0.003 | 0.103 | survives before multiple-test adjustment |
| co-listening map vs old scorer | 13 | 0.733 [0.673, 0.820] | 0.687 [0.635, 0.820] | 0.000 [-0.003, 0.040] | 0.016 [-0.001, 0.034] | 6/4/3 | 0.754 | 1.000 | no difference |
| blend map vs old scorer | 13 | 0.742 [0.642, 0.820] | 0.687 [0.635, 0.820] | 0.003 [-0.013, 0.018] | 0.003 [-0.012, 0.019] | 7/4/2 | 0.549 | 1.000 | no difference |
| new-artist rounds, map vs chance | 9 | 0.179 [0.109, 0.280] | 0.200 | -0.021 [-0.091, 0.080] | -0.018 [-0.062, 0.035] | 2/7/0 | 0.180 | 1.000 | no difference |
| new-artist rounds, blend vs chance | 9 | 0.179 [0.053, 0.313] | 0.200 | -0.021 [-0.147, 0.112] | -0.031 [-0.103, 0.045] | 3/6/0 | 0.508 | 1.000 | no difference |
| **4 when you listen (bits per play)** | | | | | | | | | |
| local clock vs uniform | 13 | 0.123 [-0.065, 0.259] | 0.000 | 0.123 [-0.065, 0.259] | 0.098 [-0.037, 0.236] | 8/5/0 | 0.581 | 1.000 | no difference |
| tuned clock vs all-time clock | 13 | 0.123 [-0.065, 0.259] | 0.135 [-0.039, 0.290] | 0.000 [-0.032, 0.010] | -0.049 [-0.157, 0.030] | 5/5/3 | 1.000 | 1.000 | no difference |
| local hours vs UTC hours | 13 | 0.123 [-0.065, 0.259] | 0.117 [-0.035, 0.191] | 0.001 [-0.020, 0.084] | 0.015 [-0.028, 0.054] | 7/5/1 | 0.774 | 1.000 | no difference |
| **5 new or familiar (AUC)** | | | | | | | | | |
| novelty model vs chance | 12 | 0.757 [0.650, 0.890] | 0.500 | 0.257 [0.150, 0.390] | 0.267 [0.196, 0.341] | 12/0/0 | <0.001 | 0.017 | survives |
| **6 new loves, rolling windows (recall@50)** | | | | | | | | | |
| v0 ranker vs popular | 10 | 0.033 [0.000, 0.117] | 0.012 [0.000, 0.067] | 0.000 [0.000, 0.037] | 0.109 [-0.002, 0.315] | 4/1/5 | 0.375 | 1.000 | no difference |
| blend ranker vs popular | 10 | 0.008 [0.000, 0.034] | 0.012 [0.000, 0.062] | 0.000 [-0.050, 0.017] | 0.026 [-0.052, 0.141] | 2/4/4 | 0.688 | 1.000 | no difference |
| plain lists vs popular | 10 | 0.038 [0.021, 0.117] | 0.012 [0.000, 0.062] | 0.000 [-0.008, 0.066] | 0.063 [0.000, 0.170] | 4/2/4 | 0.688 | 1.000 | weak |
| two-stage ranker vs popular | 10 | 0.051 [0.008, 0.140] | 0.012 [0.000, 0.069] | 0.027 [0.000, 0.070] | 0.117 [-0.005, 0.325] | 7/2/1 | 0.180 | 1.000 | no difference |
| two-stage ranker vs plain lists | 10 | 0.051 [0.008, 0.140] | 0.048 [0.016, 0.117] | 0.004 [-0.017, 0.067] | 0.054 [-0.019, 0.162] | 5/2/3 | 0.453 | 1.000 | no difference |
| **6 new loves, rolling windows (recall@20)** | | | | | | | | | |
| two-stage ranker vs popular | 10 | 0.019 [0.000, 0.072] | 0.000 [0.000, 0.024] | 0.000 [-0.008, 0.067] | 0.064 [-0.000, 0.172] | 4/2/4 | 0.688 | 1.000 | no difference |
| **6 new loves, held-out 8 weeks (recall@50)** | | | | | | | | | |
| two-stage ranker vs popular | 9 | 0.000 [0.000, 0.167] | 0.000 [0.000, 0.040] | 0.000 [0.000, 0.000] | 0.014 [-0.013, 0.056] | 1/1/7 | 1.000 | 1.000 | no difference |
| **7 selves from context** | | | | | | | | | |
| accuracy vs majority self | 13 | 0.482 [0.341, 0.573] | 0.256 [0.129, 0.503] | 0.000 [0.000, 0.113] | 0.109 [0.013, 0.219] | 6/3/4 | 0.508 | 1.000 | weak |
| log-loss vs self frequencies (lower is better) | 13 | 1.450 [0.964, 1.633] | 1.529 [1.098, 1.756] | 0.036 [-0.000, 0.129] | 0.075 [0.019, 0.140] | 9/4/0 | 0.267 | 1.000 | weak |
| **8 eras (NDCG@10)** | | | | | | | | | |
| era-aware forecast vs decay | 13 | 0.813 [0.684, 0.919] | 0.813 [0.684, 0.919] | 0.000 [0.000, 0.000] | -0.003 [-0.006, 0.000] | 1/3/9 | 0.625 | 1.000 | no difference |
| current era only vs decay | 13 | 0.824 [0.643, 0.895] | 0.813 [0.684, 0.919] | -0.013 [-0.041, 0.000] | -0.029 [-0.085, 0.013] | 3/8/2 | 0.227 | 1.000 | no difference |
| **9 will it stick, after the first session (AUC, training period)** | | | | | | | | | |
| model vs own base rate | 12 | 0.773 [0.721, 0.812] | 0.484 [0.437, 0.510] | 0.272 [0.211, 0.350] | 0.281 [0.239, 0.325] | 12/0/0 | <0.001 | 0.017 | survives |
| model vs first-play skip | 12 | 0.773 [0.721, 0.812] | 0.593 [0.553, 0.667] | 0.129 [0.113, 0.218] | 0.146 [0.091, 0.192] | 11/1/0 | 0.006 | 0.184 | survives before multiple-test adjustment |
| model vs popularity | 12 | 0.773 [0.721, 0.812] | 0.628 [0.577, 0.662] | 0.125 [0.092, 0.219] | 0.142 [0.089, 0.194] | 11/1/0 | 0.006 | 0.184 | survives before multiple-test adjustment |
| model vs the same model without skip, start, shuffle and hour features | 12 | 0.773 [0.721, 0.812] | 0.762 [0.714, 0.781] | 0.014 [0.001, 0.048] | 0.023 [0.003, 0.044] | 10/2/0 | 0.039 | 0.964 | survives before multiple-test adjustment |
| model without skip, start, shuffle and hour features vs own base rate | 12 | 0.762 [0.714, 0.781] | 0.484 [0.437, 0.510] | 0.271 [0.204, 0.322] | 0.258 [0.213, 0.301] | 12/0/0 | <0.001 | 0.017 | survives |
| **9 will it stick, after the first session (average precision, training period)** | | | | | | | | | |
| model vs popularity | 12 | 0.287 [0.158, 0.407] | 0.091 [0.067, 0.194] | 0.162 [0.056, 0.186] | 0.130 [0.076, 0.181] | 11/1/0 | 0.006 | 0.184 | survives before multiple-test adjustment |
| **9 will it stick, after the first session (Brier, training period)** | | | | | | | | | |
| model vs own base rate (lower is better) | 13 | 0.039 [0.029, 0.094] | 0.044 [0.029, 0.105] | 0.003 [0.001, 0.008] | 0.017 [0.002, 0.041] | 10/3/0 | 0.092 | 1.000 | weak |
| **9 will it stick, after the first session (AUC, held-out 8 weeks)** | | | | | | | | | |
| model vs popularity | 6 | 0.831 [0.766, 0.918] | 0.634 [0.518, 0.705] | 0.224 [0.060, 0.373] | 0.219 [0.113, 0.331] | 6/0/0 | 0.031 | 0.813 | survives before multiple-test adjustment |
| **10 will it stick, after 7 days (AUC, training period)** | | | | | | | | | |
| model vs first-week play counts | 12 | 0.899 [0.868, 0.922] | 0.880 [0.840, 0.904] | 0.019 [0.002, 0.031] | 0.016 [0.006, 0.026] | 9/2/1 | 0.065 | 1.000 | weak |
| model vs popularity | 12 | 0.899 [0.868, 0.922] | 0.632 [0.597, 0.677] | 0.263 [0.214, 0.322] | 0.269 [0.232, 0.306] | 12/0/0 | <0.001 | 0.017 | survives |
| model vs the same model without skip, start, shuffle and hour features | 12 | 0.899 [0.868, 0.922] | 0.898 [0.871, 0.920] | -0.001 [-0.005, 0.003] | -0.003 [-0.012, 0.003] | 4/7/1 | 0.549 | 1.000 | no difference |
| **10 will it stick, after 7 days (AUC, held-out 8 weeks)** | | | | | | | | | |
| model vs first-week play counts | 6 | 0.934 [0.725, 0.980] | 0.874 [0.755, 0.972] | 0.023 [-0.048, 0.062] | 0.013 [-0.036, 0.052] | 5/1/0 | 0.219 | 1.000 | no difference |

## Skip model calibration

Held-out plays pooled over everyone (48694). Expected calibration error (ECE) is the play-weighted mean gap between predicted and observed skip rate over 10 equal-width bins. Brier is the mean squared error of the probability.

| | Brier | ECE |
|---|---|---|
| skip model | 0.109 | 0.010 |
| artist skip rate (baseline) | 0.176 | 0.020 |

Per person: median Brier 0.134 [0.100, 0.147] against 0.180 for the baseline; the model has the lower Brier for 13 of 13 people (sign p <0.001). Median ECE 0.030 [0.019, 0.038] against 0.044; lower for the model in 11 of 13 (sign p 0.022).

Reliability, skip model (pooled):

| predicted | plays | mean predicted | observed skip rate |
|---|---|---|---|
| 0.0 to 0.1 | 19770 | 0.037 | 0.038 |
| 0.1 to 0.2 | 4984 | 0.141 | 0.171 |
| 0.2 to 0.3 | 2839 | 0.247 | 0.277 |
| 0.3 to 0.4 | 2210 | 0.349 | 0.351 |
| 0.4 to 0.5 | 1882 | 0.450 | 0.428 |
| 0.5 to 0.6 | 1909 | 0.550 | 0.514 |
| 0.6 to 0.7 | 2139 | 0.651 | 0.640 |
| 0.7 to 0.8 | 2603 | 0.752 | 0.767 |
| 0.8 to 0.9 | 4179 | 0.857 | 0.855 |
| 0.9 to 1.0 | 6179 | 0.939 | 0.931 |

Reliability, artist skip rate (pooled):

| predicted | plays | mean predicted | observed skip rate |
|---|---|---|---|
| 0.0 to 0.1 | 5788 | 0.068 | 0.082 |
| 0.1 to 0.2 | 12572 | 0.164 | 0.160 |
| 0.2 to 0.3 | 7230 | 0.252 | 0.259 |
| 0.3 to 0.4 | 5013 | 0.344 | 0.338 |
| 0.4 to 0.5 | 4092 | 0.451 | 0.484 |
| 0.5 to 0.6 | 5132 | 0.552 | 0.579 |
| 0.6 to 0.7 | 4712 | 0.645 | 0.725 |
| 0.7 to 0.8 | 3071 | 0.745 | 0.768 |
| 0.8 to 0.9 | 1011 | 0.837 | 0.796 |
| 0.9 to 1.0 | 73 | 0.911 | 0.877 |

## Doors: how new loves came in

Each new love (an artist played properly 10+ times on 3+ days within 8 weeks of first hearing it) is compared with up to 5 artists the same person tried and dropped, first heard nearest in time. The list fact is also compared with never-played ListenBrainz artists named on as many lists. The ratio is the pooled rate for new loves over the pooled rate for controls, with a bootstrap interval over people. Wins count people (with 3+ new loves) whose new-love rate is higher; the sign test is over them.

| door | new loves | controls | ratio [95%] | wins/losses/ties | sign p | verdict |
|---|---|---|---|---|---|---|
| picked | 0.423 | 0.377 | 1.12 [0.88, 1.51] | 9/2/0 | 0.065 | no clear difference |
| came on after a similar artist | 0.084 | 0.035 | 2.40 [1.11, 3.55] | 8/3/0 | 0.227 | more common for new loves |
| next to a favourite | 0.129 | 0.052 | 2.47 [1.56, 3.04] | 9/2/0 | 0.065 | more common for new loves |
| came on by itself | 0.355 | 0.514 | 0.69 [0.57, 0.80] | 1/10/0 | 0.012 | no clear difference |
| out of nowhere | 0.008 | 0.022 | 0.37 [0.09, 0.78] | 1/9/1 | 0.021 | less common for new loves |

Single facts at the first play (an artist can have several):

| fact | new loves | controls | ratio [95%] | wins/losses/ties | sign p | verdict |
|---|---|---|---|---|---|---|
| picked | 0.423 | 0.377 | 1.12 [0.89, 1.50] | 9/2/0 | 0.065 | no clear difference |
| cameOn | 0.554 | 0.598 | 0.93 [0.75, 1.10] | 1/10/0 | 0.012 | no clear difference |
| prevLists | 0.191 | 0.106 | 1.80 [1.15, 2.33] | 7/3/0 | 0.344 | more common for new loves |
| cameOnAfterSimilar | 0.111 | 0.077 | 1.44 [0.65, 2.11] | 6/4/1 | 0.754 | no clear difference |
| pickedAfterFav | 0.114 | 0.080 | 1.43 [1.22, 1.80] | 10/1/0 | 0.012 | no clear difference |
| favLists | 0.422 | 0.237 | 1.78 [1.30, 2.17] | 8/3/0 | 0.227 | more common for new loves |
| favListsTop25 | 0.224 | 0.091 | 2.47 [1.77, 3.44] | 9/2/0 | 0.065 | more common for new loves |
| prevIsFav | 0.285 | 0.234 | 1.22 [1.13, 1.40] | 11/0/0 | <0.001 | no clear difference |
| shuffle | 0.432 | 0.383 | 1.13 [0.94, 1.37] | 6/5/0 | 1.000 | no clear difference |
| firstWasSkip | 0.193 | 0.495 | 0.39 [0.33, 0.44] | 0/11/0 | <0.001 | less common for new loves |
| a recent favourite lists it (vs never-played) | 0.423 | 0.084 | 5.04 [4.08, 6.54] | 12/0/0 | <0.001 | more common for new loves |

1821 new loves and 17374 tried-and-dropped artists. Median days from first play to the 10th proper play: 15.139.

## Trait stability

Each trait is measured on the two halves of every person's training time and correlated across people (test-retest). Intervals are a bootstrap over people. The bench rule for showing a trait (both correlations at least 0.5 as point estimates) is in the status column; the last column asks the stricter question of whether the interval itself clears 0.5.

| group | trait | n | Pearson r [95%] | Spearman rho [95%] | bench status | interval reading |
|---|---|---|---|---|---|---|
| adventurousness | newShare | 10 | 0.723 [0.394, 0.929] | 0.758 [0.113, 0.987] | stable | above zero, not shown to reach 0.5 |
| adventurousness | spread | 12 | 0.628 [0.016, 0.938] | 0.476 [-0.256, 0.884] | stable | not shown to be stable |
| loyalty | oldFriends | 10 | 0.680 [-0.422, 0.926] | 0.479 [-0.355, 0.889] | stable | not shown to be stable |
| patience | finished | 11 | 0.864 [0.668, 0.972] | 0.782 [0.315, 0.972] | stable | above zero, not shown to reach 0.5 |
| patience | skipSeconds | 12 | 0.988 [0.623, 0.999] | 0.902 [0.634, 0.979] | stable | stable, interval above 0.5 |
| clock | clockHour | 11 | 0.531 [-0.014, 0.908] | 0.700 [0.009, 0.990] | stable | not shown to be stable |
| clock | nightShare | 11 | 0.348 [-0.380, 0.953] | 0.364 [-0.406, 0.972] | dropped | not shown to be stable |
| clock | clockSpread | 12 | 0.608 [0.091, 0.943] | 0.538 [-0.164, 0.915] | stable | not shown to be stable |
| clock | clockRegular | 12 | 0.808 [0.562, 0.939] | 0.832 [0.376, 0.993] | stable | above zero, not shown to reach 0.5 |
| era turnover | turnover | 10 | 0.883 [0.468, 0.981] | 0.855 [0.373, 1.000] | stable | above zero, not shown to reach 0.5 |

## New loves: the two-stage ranker, pooled

Recall pools every new love across people and windows (each love counts once). Coverage is the share of new loves that stage one proposes at all; a love outside the candidates is a miss for the ranker and the lists baseline, while popular ranks every unheard artist other people played.

| windows | new loves | candidate coverage | ranker @20 | lists @20 | popular @20 | ranker @50 | lists @50 | popular @50 |
|---|---|---|---|---|---|---|---|---|
| rolling, inside training | 407 | 0.221 | 0.025 | 0.017 | 0.022 | 0.054 | 0.042 | 0.032 |
| held-out 8 weeks | 52 | 0.115 | 0.000 | 0.000 | 0.019 | 0.038 | 0.000 | 0.038 |

Ranker weights fitted on everyone, standardised features (reading only): bias -6.05, logListers 0.63, rrSum -0.06, rrMax -0.26, logBestRank -0.72, scoreSum -0.82, logListed -0.41, logPeople 0.51, logPlays 0.28, maxSimFav 0.17, fromMapOnly 0.56, propList -0.56, propList_x_rrSum 0.39.

## Will a new artist stick

Every artist a person hears for the first time (after their first 12 weeks) is one row; it sticks when it becomes a new love (10+ proper plays on 3+ days within 8 weeks). The model is a logistic regression trained on other people only and reads nothing after the horizon: the end of the first session, or the end of the first 7 days (rows already past the bar by day 7 are left out there). The leakage check re-read 614 sampled rows from the history cut at the horizon; 0 differed. Per-person numbers are the primary reading. Pooled numbers rank people against each other as well, so a pooled AUC can sit below 0.5 under shuffled labels.

| horizon | rows | new artists | stuck | model | median AUC per person [95%] | median average precision [95%] | median Brier [95%] | median ECE [95%] | pooled AUC | pooled AP |
|---|---|---|---|---|---|---|---|---|---|---|
| first session | training period | 26401 | 1771 | model | 0.773 [0.721, 0.812] | 0.287 [0.158, 0.407] | 0.039 [0.029, 0.094] | 0.029 [0.010, 0.056] | 0.816 | 0.314 |
| first session | training period | 26401 | 1771 | model without skip, start, shuffle and hour features | 0.762 [0.714, 0.781] | 0.231 [0.145, 0.387] | 0.040 [0.030, 0.098] | 0.028 [0.016, 0.053] | 0.805 | 0.300 |
| first session | training period | 26401 | 1771 | own base rate | 0.484 [0.437, 0.510] | 0.051 [0.035, 0.145] | 0.044 [0.029, 0.105] | 0.018 [0.013, 0.038] | 0.679 | 0.131 |
| first session | training period | 26401 | 1771 | first-play skip | 0.593 [0.553, 0.667] | 0.037 [0.023, 0.095] | 0.045 [0.030, 0.108] | 0.039 [0.037, 0.069] | 0.491 | 0.057 |
| first session | training period | 26401 | 1771 | popularity | 0.628 [0.577, 0.662] | 0.091 [0.067, 0.194] | 0.044 [0.031, 0.109] | 0.060 [0.036, 0.069] | 0.603 | 0.094 |
| first session | training period | 26401 | 1771 | boosted trees (check) | 0.764 [0.730, 0.792] | 0.262 [0.131, 0.338] | 0.039 [0.030, 0.102] | 0.016 [0.014, 0.066] | 0.793 | 0.256 |
| first session | training period | 26401 | 1771 | labels shuffled (null) | 0.529 [0.463, 0.583] | 0.130 [0.044, 0.239] | 0.044 [0.029, 0.108] | 0.055 [0.030, 0.067] | 0.377 | 0.057 |
| first session | training period | 26401 | 1771 | labels shuffled inside each person | 0.582 [0.506, 0.613] | 0.110 [0.049, 0.188] | 0.044 [0.029, 0.104] | 0.015 [0.013, 0.033] | 0.689 | 0.149 |
| first session | held-out 8 weeks | 1166 | 50 | model | 0.831 [0.766, 0.918] | 0.389 [0.247, 0.578] | 0.030 [0.019, 0.071] | 0.021 [0.016, 0.052] | 0.850 | 0.240 |
| first session | held-out 8 weeks | 1166 | 50 | model without skip, start, shuffle and hour features | 0.812 [0.754, 0.881] | 0.368 [0.251, 0.587] | 0.030 [0.019, 0.072] | 0.025 [0.016, 0.045] | 0.849 | 0.232 |
| first session | held-out 8 weeks | 1166 | 50 | own base rate | 0.478 [0.374, 0.572] | 0.103 [0.049, 0.266] | 0.038 [0.024, 0.083] | 0.021 [0.006, 0.034] | 0.665 | 0.092 |
| first session | held-out 8 weeks | 1166 | 50 | first-play skip | 0.624 [0.506, 0.782] | 0.069 [0.041, 0.146] | 0.041 [0.026, 0.084] | 0.042 [0.014, 0.051] | 0.553 | 0.043 |
| first session | held-out 8 weeks | 1166 | 50 | popularity | 0.634 [0.518, 0.705] | 0.168 [0.051, 0.198] | 0.040 [0.023, 0.084] | 0.040 [0.013, 0.052] | 0.522 | 0.045 |
| first session | held-out 8 weeks | 1166 | 50 | boosted trees (check) | 0.807 [0.771, 0.930] | 0.338 [0.235, 0.586] | 0.033 [0.018, 0.066] | 0.025 [0.016, 0.040] | 0.858 | 0.271 |
| first session | held-out 8 weeks | 1166 | 50 | labels shuffled (null) | 0.390 [0.226, 0.735] | 0.132 [0.042, 0.290] | 0.040 [0.025, 0.084] | 0.041 [0.012, 0.048] | 0.374 | 0.037 |
| first session | held-out 8 weeks | 1166 | 50 | labels shuffled inside each person | 0.564 [0.465, 0.826] | 0.185 [0.085, 0.550] | 0.038 [0.024, 0.083] | 0.019 [0.005, 0.035] | 0.713 | 0.093 |
| first 7 days | training period | 25887 | 1257 | model | 0.899 [0.868, 0.922] | 0.538 [0.458, 0.602] | 0.020 [0.015, 0.052] | 0.011 [0.009, 0.019] | 0.912 | 0.525 |
| first 7 days | training period | 25887 | 1257 | model without skip, start, shuffle and hour features | 0.898 [0.871, 0.920] | 0.540 [0.445, 0.609] | 0.021 [0.015, 0.051] | 0.012 [0.008, 0.018] | 0.913 | 0.527 |
| first 7 days | training period | 25887 | 1257 | own base rate | 0.481 [0.411, 0.528] | 0.040 [0.020, 0.105] | 0.031 [0.023, 0.080] | 0.012 [0.010, 0.022] | 0.660 | 0.094 |
| first 7 days | training period | 25887 | 1257 | first-play skip | 0.581 [0.550, 0.666] | 0.028 [0.017, 0.070] | 0.030 [0.024, 0.082] | 0.032 [0.026, 0.051] | 0.498 | 0.042 |
| first 7 days | training period | 25887 | 1257 | popularity | 0.632 [0.597, 0.677] | 0.078 [0.041, 0.159] | 0.031 [0.025, 0.082] | 0.039 [0.023, 0.052] | 0.617 | 0.073 |
| first 7 days | training period | 25887 | 1257 | first-week play counts | 0.880 [0.840, 0.904] | 0.513 [0.394, 0.575] | 0.021 [0.013, 0.054] | 0.021 [0.014, 0.027] | 0.865 | 0.486 |
| first 7 days | training period | 25887 | 1257 | boosted trees (check) | 0.895 [0.852, 0.914] | 0.497 [0.402, 0.523] | 0.021 [0.015, 0.057] | 0.013 [0.010, 0.023] | 0.903 | 0.439 |
| first 7 days | training period | 25887 | 1257 | labels shuffled (null) | 0.450 [0.358, 0.511] | 0.064 [0.036, 0.134] | 0.031 [0.023, 0.084] | 0.030 [0.025, 0.048] | 0.321 | 0.042 |
| first 7 days | training period | 25887 | 1257 | labels shuffled inside each person | 0.643 [0.568, 0.689] | 0.124 [0.045, 0.294] | 0.031 [0.023, 0.078] | 0.017 [0.013, 0.031] | 0.704 | 0.156 |
| first 7 days | held-out 8 weeks | 1148 | 32 | model | 0.934 [0.725, 0.980] | 0.638 [0.331, 0.832] | 0.025 [0.003, 0.043] | 0.012 [0.007, 0.035] | 0.919 | 0.387 |
| first 7 days | held-out 8 weeks | 1148 | 32 | model without skip, start, shuffle and hour features | 0.940 [0.741, 0.977] | 0.595 [0.341, 0.793] | 0.025 [0.003, 0.042] | 0.011 [0.008, 0.033] | 0.922 | 0.373 |
| first 7 days | held-out 8 weeks | 1148 | 32 | own base rate | 0.387 [0.374, 0.496] | 0.052 [0.037, 0.147] | 0.033 [0.007, 0.046] | 0.019 [0.004, 0.027] | 0.610 | 0.041 |
| first 7 days | held-out 8 weeks | 1148 | 32 | first-play skip | 0.672 [0.490, 0.782] | 0.037 [0.029, 0.144] | 0.032 [0.007, 0.046] | 0.028 [0.013, 0.043] | 0.586 | 0.031 |
| first 7 days | held-out 8 weeks | 1148 | 32 | popularity | 0.650 [0.550, 0.767] | 0.148 [0.048, 0.215] | 0.032 [0.010, 0.044] | 0.024 [0.007, 0.035] | 0.580 | 0.034 |
| first 7 days | held-out 8 weeks | 1148 | 32 | first-week play counts | 0.874 [0.755, 0.972] | 0.544 [0.214, 0.757] | 0.025 [0.003, 0.041] | 0.023 [0.011, 0.029] | 0.882 | 0.393 |
| first 7 days | held-out 8 weeks | 1148 | 32 | boosted trees (check) | 0.907 [0.736, 0.980] | 0.479 [0.255, 0.854] | 0.026 [0.003, 0.050] | 0.018 [0.008, 0.037] | 0.908 | 0.287 |
| first 7 days | held-out 8 weeks | 1148 | 32 | labels shuffled (null) | 0.369 [0.170, 0.485] | 0.050 [0.029, 0.156] | 0.033 [0.008, 0.046] | 0.024 [0.006, 0.048] | 0.273 | 0.018 |
| first 7 days | held-out 8 weeks | 1148 | 32 | labels shuffled inside each person | 0.726 [0.602, 0.864] | 0.223 [0.105, 0.485] | 0.032 [0.007, 0.044] | 0.020 [0.007, 0.030] | 0.709 | 0.092 |

Feature groups, training period. "Without" drops the group from the model; "alone" is a model on that group only. AUC is the median per person; the change is the median of each person's difference from the full model, with its interval.

| horizon | group | AUC without | change without [95%] | AUC alone |
|---|---|---|---|---|
| first session | first play (skip, seconds) | 0.771 | -0.004 [-0.013, 0.000] | 0.636 |
| first session | how it started | 0.771 | -0.006 [-0.025, -0.001] | 0.573 |
| first session | ListenBrainz lists | 0.764 | -0.009 [-0.016, -0.001] | 0.645 |
| first session | favourites in the session | 0.774 | 0.000 [-0.000, 0.000] | 0.526 |
| first session | taste map | 0.771 | -0.001 [-0.003, 0.000] | 0.589 |
| first session | popularity | 0.772 | -0.000 [-0.002, 0.002] | 0.628 |
| first session | the first session | 0.730 | -0.051 [-0.068, -0.034] | 0.736 |
| first session | time of day | 0.775 | -0.000 [-0.003, 0.004] | 0.512 |
| first session | own base rate | 0.779 | 0.013 [-0.008, 0.027] | 0.484 |
| first 7 days | first play (skip, seconds) | 0.900 | -0.000 [-0.001, 0.001] | 0.635 |
| first 7 days | how it started | 0.903 | 0.001 [0.000, 0.002] | 0.531 |
| first 7 days | ListenBrainz lists | 0.893 | -0.002 [-0.006, 0.003] | 0.646 |
| first 7 days | favourites in the session | 0.899 | -0.000 [-0.000, 0.000] | 0.518 |
| first 7 days | taste map | 0.899 | -0.000 [-0.001, 0.000] | 0.567 |
| first 7 days | popularity | 0.899 | 0.000 [-0.001, 0.001] | 0.632 |
| first 7 days | the first session | 0.900 | -0.000 [-0.003, 0.002] | 0.696 |
| first 7 days | time of day | 0.901 | 0.000 [-0.001, 0.002] | 0.523 |
| first 7 days | own base rate | 0.905 | 0.001 [-0.003, 0.008] | 0.481 |
| first 7 days | the first week | 0.759 | -0.149 [-0.198, -0.135] | 0.884 |

Session-horizon weights fitted on everyone, standardised (reading only): bias -3.38, firstSkip 0.17, logFirstSec 0.53, picked 0.28, cameOn 0.01, shuffle 0.03, prevLists 0.02, prevRankScore 0.00, favLists 0.08, favRankScore 0.12, onLb 0.51, favShare 0.02, mapNear 0.08, onMap -0.19, logPeople 0.11, logListed -0.10, sessPlays 0.01, sessProper 0.64, sessTracks -0.01, sessRepeat 0.21, sessReturn 0.01, hourSin -0.11, hourCos 0.02, sessLen -0.21, sessSkipRate 0.34, baseLogit 0.63.

Examples from the held-out weeks, scored after the first session, written so the person can check them against their own history:

- First heard Subsonica in April 2024 (4 plays of it that session, 4 played past 30 seconds; it came on after Fugazi; one of your recent favourites lists it on ListenBrainz). After that session the model gave it 66% to stick. It stuck: 10 proper plays on 3 days by day 2.
- First heard SpaceGhostPurrp in August 2025 (4 plays of it that session, 4 played past 30 seconds; it came on after Lil Ugly Mane; Lil Ugly Mane lists it on ListenBrainz). After that session the model gave it 50% to stick. It stuck: 10 proper plays on 3 days by day 35.
- First heard Cutting Crew in April 2024 (10 plays of it that session, 10 played past 30 seconds; you picked it yourself). After that session the model gave it 95% to stick. It did not stick within 8 weeks.
- First heard Penguin Piano in October 2024 (2 plays of it that session, 2 played past 30 seconds; it came on by itself). After that session the model gave it 2% to stick. It stuck: 10 proper plays on 3 days by day 13.
- First heard Gregory Alan Isakov in September 2025 (1 play of it that session, 0 played past 30 seconds; it came on after Taylor Swift). After that session the model gave it 2% to stick. It did not stick within 8 weeks.

## What survives

Survive the adjustment: 1 finish or skip (AUC): skip model vs artist skip rate; 5 new or familiar (AUC): novelty model vs chance; 9 will it stick, after the first session (AUC, training period): model vs own base rate; 9 will it stick, after the first session (AUC, training period): model without skip, start, shuffle and hour features vs own base rate; 10 will it stick, after 7 days (AUC, training period): model vs popularity.

Significant only before adjustment: 3 the game (share correct): old scorer vs chance; 9 will it stick, after the first session (AUC, training period): model vs first-play skip; 9 will it stick, after the first session (AUC, training period): model vs popularity; 9 will it stick, after the first session (AUC, training period): model vs the same model without skip, start, shuffle and hour features; 9 will it stick, after the first session (average precision, training period): model vs popularity; 9 will it stick, after the first session (AUC, held-out 8 weeks): model vs popularity.

Weak (one test passes, the other does not): 6 new loves, rolling windows (recall@50): plain lists vs popular; 7 selves from context: accuracy vs majority self; 7 selves from context: log-loss vs self frequencies (lower is better); 9 will it stick, after the first session (Brier, training period): model vs own base rate (lower is better); 10 will it stick, after 7 days (AUC, training period): model vs first-week play counts.

No difference or worse: 2 next 8 weeks' artists (NDCG@10): decay vs all-time; 2 next 8 weeks' artists (NDCG@10): decay vs last 4 weeks; 3 the game (share correct): co-listening map vs old scorer; 3 the game (share correct): blend map vs old scorer; 3 the game (share correct): new-artist rounds, map vs chance; 3 the game (share correct): new-artist rounds, blend vs chance; 4 when you listen (bits per play): local clock vs uniform; 4 when you listen (bits per play): tuned clock vs all-time clock; 4 when you listen (bits per play): local hours vs UTC hours; 6 new loves, rolling windows (recall@50): v0 ranker vs popular; 6 new loves, rolling windows (recall@50): blend ranker vs popular; 6 new loves, rolling windows (recall@50): two-stage ranker vs popular; 6 new loves, rolling windows (recall@50): two-stage ranker vs plain lists; 6 new loves, rolling windows (recall@20): two-stage ranker vs popular; 6 new loves, held-out 8 weeks (recall@50): two-stage ranker vs popular; 8 eras (NDCG@10): era-aware forecast vs decay; 8 eras (NDCG@10): current era only vs decay; 10 will it stick, after 7 days (AUC, training period): model vs the same model without skip, start, shuffle and hour features; 10 will it stick, after 7 days (AUC, held-out 8 weeks): model vs first-week play counts.


## Round 3: 343 ListenBrainz listeners

304 of the 343 people kept in bench/lbusers have at least 1,000 listens before their last 8 weeks and 200 in them, so they get the same time split as the Spotify bench. Median history 4.125 years, median 59057 listens. ListenBrainz histories have no play lengths, skips, start reasons, shuffle, device or country, so finish-or-skip, the selves, the game against matched decoys and the local clock cannot be run here; the clock below uses UTC hours. Everything learned (taste map, stick model, ranker) is fitted on the other four of five folds of people. Holm runs over all 26 sign tests of this round.

One caveat runs through every ListenBrainz list feature here: ListenBrainz computes its similar-artist lists from its users' listening sessions, and these people are ListenBrainz users. Their own sessions are a small part of the data behind the lists, but not zero, so list features can carry a trace of the very listening they are used to explain. The stick model is reported with and without them, and the door section checks whether heavy listeners (more sessions in the lists) show a stronger list door than light ones.

### Paired comparisons

| comparison | n | model median [95%] | baseline median [95%] | median difference [95%] | mean difference [95%] | wins/losses/ties | sign p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **2 next 8 weeks' artists (NDCG@10)** | | | | | | | | | |
| decay vs all-time | 304 | 0.706 [0.677, 0.736] | 0.604 [0.551, 0.650] | 0.066 [0.048, 0.083] | 0.091 [0.072, 0.109] | 218/72/14 | <0.001 | <0.001 | survives |
| decay vs last 4 weeks | 304 | 0.706 [0.677, 0.735] | 0.617 [0.587, 0.652] | 0.050 [0.031, 0.061] | 0.059 [0.043, 0.075] | 212/87/5 | <0.001 | <0.001 | survives |
| **4 when you listen (bits per play, UTC hour of the week)** | | | | | | | | | |
| tuned clock vs uniform | 304 | 0.456 [0.405, 0.501] | 0.000 | 0.456 [0.406, 0.506] | 0.454 [0.398, 0.508] | 268/36/0 | <0.001 | <0.001 | survives |
| tuned clock vs all-time clock | 304 | 0.456 [0.404, 0.495] | 0.414 [0.357, 0.460] | 0.001 [0.000, 0.012] | 0.054 [0.027, 0.079] | 153/67/84 | <0.001 | <0.001 | survives |
| **6 new loves, rolling windows (recall@50)** | | | | | | | | | |
| two-stage ranker vs popular | 293 | 0.032 [0.022, 0.041] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.010] | 0.033 [0.023, 0.044] | 132/24/137 | <0.001 | <0.001 | survives |
| two-stage ranker vs plain lists | 293 | 0.032 [0.022, 0.039] | 0.000 [0.000, 0.010] | 0.000 [0.000, 0.000] | 0.019 [0.011, 0.027] | 109/30/154 | <0.001 | <0.001 | survives |
| plain lists vs popular | 293 | 0.000 [0.000, 0.009] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.014 [0.005, 0.023] | 83/49/161 | 0.004 | 0.015 | survives |
| **6 new loves, rolling windows (recall@20)** | | | | | | | | | |
| two-stage ranker vs popular | 293 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.023 [0.016, 0.031] | 95/19/179 | <0.001 | <0.001 | survives |
| **6 new loves, held-out 8 weeks (recall@50)** | | | | | | | | | |
| two-stage ranker vs popular | 184 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.017 [-0.008, 0.041] | 22/6/156 | 0.004 | 0.015 | weak |
| **8 eras (NDCG@10)** | | | | | | | | | |
| era-aware forecast vs decay | 304 | 0.706 [0.676, 0.734] | 0.706 [0.677, 0.735] | 0.000 [0.000, 0.000] | -0.003 [-0.008, 0.002] | 67/76/161 | 0.504 | 0.504 | no difference |
| current era only vs decay | 304 | 0.688 [0.655, 0.714] | 0.706 [0.677, 0.738] | -0.019 [-0.029, -0.012] | -0.026 [-0.039, -0.014] | 107/192/5 | <0.001 | <0.001 | worse than baseline |
| **9 will it stick, after the first session (AUC, training period)** | | | | | | | | | |
| model vs own base rate | 303 | 0.792 [0.784, 0.800] | 0.466 [0.456, 0.478] | 0.316 [0.306, 0.324] | 0.307 [0.295, 0.318] | 301/2/0 | <0.001 | <0.001 | survives |
| model vs popularity | 303 | 0.792 [0.784, 0.800] | 0.618 [0.606, 0.631] | 0.170 [0.159, 0.180] | 0.161 [0.149, 0.172] | 283/20/0 | <0.001 | <0.001 | survives |
| model vs the model without ListenBrainz list features | 303 | 0.792 [0.785, 0.800] | 0.793 [0.781, 0.800] | 0.002 [0.001, 0.002] | 0.002 [0.002, 0.003] | 195/108/0 | <0.001 | <0.001 | survives |
| **9 will it stick, after the first session (average precision, training period)** | | | | | | | | | |
| model vs popularity | 303 | 0.208 [0.190, 0.232] | 0.063 [0.055, 0.070] | 0.135 [0.120, 0.151] | 0.147 [0.137, 0.158] | 293/10/0 | <0.001 | <0.001 | survives |
| **9 will it stick, after the first session (Brier, training period)** | | | | | | | | | |
| model vs own base rate (lower is better) | 304 | 0.034 [0.029, 0.040] | 0.037 [0.032, 0.044] | 0.002 [0.002, 0.003] | 0.006 [0.005, 0.008] | 273/31/0 | <0.001 | <0.001 | survives |
| **9 will it stick, after the first session (AUC, held-out 8 weeks)** | | | | | | | | | |
| model vs own base rate | 112 | 0.811 [0.783, 0.836] | 0.500 [0.467, 0.516] | 0.315 [0.256, 0.344] | 0.274 [0.226, 0.320] | 97/15/0 | <0.001 | <0.001 | survives |
| model vs popularity | 112 | 0.811 [0.783, 0.838] | 0.607 [0.573, 0.652] | 0.167 [0.137, 0.201] | 0.184 [0.156, 0.211] | 104/8/0 | <0.001 | <0.001 | survives |
| **10 will it stick, after 7 days (AUC, training period)** | | | | | | | | | |
| model vs first-week play counts | 303 | 0.884 [0.879, 0.890] | 0.860 [0.853, 0.869] | 0.019 [0.017, 0.023] | 0.022 [0.019, 0.026] | 257/46/0 | <0.001 | <0.001 | survives |
| model vs popularity | 303 | 0.884 [0.879, 0.889] | 0.623 [0.613, 0.634] | 0.255 [0.245, 0.272] | 0.259 [0.247, 0.270] | 301/2/0 | <0.001 | <0.001 | survives |
| **10 will it stick, after 7 days (AUC, held-out 8 weeks)** | | | | | | | | | |
| model vs first-week play counts | 76 | 0.901 [0.877, 0.918] | 0.878 [0.855, 0.902] | 0.005 [-0.001, 0.013] | 0.021 [0.008, 0.036] | 41/30/5 | 0.235 | 0.470 | weak |

### Will a new artist stick, pooled

Rows: every artist a person first heard after their first 12 weeks, at most 4,000 per person in the training period (a seeded sample) and all of them in the held-out 8 weeks. The leakage re-read checked 15382 sampled rows against the history cut at the horizon; 0 differed. The model uses the columns this data can fill: the ListenBrainz list facts, favourites in the session, map nearness, popularity, the first session's plays, tracks, repeats and returns, its length, the own base rate, and after 7 days the week's plays, days, tracks and whether it came back after the first session.

| horizon | rows | model | pooled AUC | pooled AP | pooled Brier | pooled ECE | stuck / rows |
|---|---|---|---|---|---|---|---|
| first session | training period | model | 0.825 | 0.228 | 0.034 | 0.004 | 28421 / 729005 |
| first session | training period | model without ListenBrainz lists | 0.824 | 0.226 | 0.034 | 0.004 | 28421 / 729005 |
| first session | training period | own base rate | 0.687 | 0.086 | 0.037 | 0.006 | 28421 / 729005 |
| first session | training period | popularity | 0.612 | 0.059 | 0.037 | 0.011 | 28421 / 729005 |
| first session | training period | labels shuffled (null) | 0.467 | 0.043 | 0.038 | 0.013 | 28421 / 729005 |
| first session | held-out 8 weeks | model | 0.863 | 0.238 | 0.026 | 0.006 | 1158 / 38434 |
| first session | held-out 8 weeks | model without ListenBrainz lists | 0.863 | 0.238 | 0.026 | 0.006 | 1158 / 38434 |
| first session | held-out 8 weeks | own base rate | 0.735 | 0.100 | 0.028 | 0.011 | 1158 / 38434 |
| first session | held-out 8 weeks | popularity | 0.626 | 0.048 | 0.029 | 0.014 | 1158 / 38434 |
| first session | held-out 8 weeks | labels shuffled (null) | 0.507 | 0.041 | 0.030 | 0.021 | 1158 / 38434 |
| first 7 days | training period | model | 0.902 | 0.331 | 0.020 | 0.002 | 18740 / 719324 |
| first 7 days | training period | model without ListenBrainz lists | 0.902 | 0.331 | 0.021 | 0.002 | 18740 / 719324 |
| first 7 days | training period | own base rate | 0.666 | 0.053 | 0.025 | 0.003 | 18740 / 719324 |
| first 7 days | training period | popularity | 0.612 | 0.040 | 0.025 | 0.007 | 18740 / 719324 |
| first 7 days | training period | first-week play counts | 0.871 | 0.296 | 0.021 | 0.005 | 18740 / 719324 |
| first 7 days | training period | labels shuffled (null) | 0.510 | 0.051 | 0.025 | 0.007 | 18740 / 719324 |
| first 7 days | held-out 8 weeks | model | 0.916 | 0.378 | 0.015 | 0.003 | 758 / 38034 |
| first 7 days | held-out 8 weeks | model without ListenBrainz lists | 0.916 | 0.377 | 0.015 | 0.003 | 758 / 38034 |
| first 7 days | held-out 8 weeks | own base rate | 0.713 | 0.066 | 0.019 | 0.008 | 758 / 38034 |
| first 7 days | held-out 8 weeks | popularity | 0.614 | 0.030 | 0.020 | 0.009 | 758 / 38034 |
| first 7 days | held-out 8 weeks | first-week play counts | 0.906 | 0.339 | 0.016 | 0.006 | 758 / 38034 |
| first 7 days | held-out 8 weeks | labels shuffled (null) | 0.565 | 0.065 | 0.020 | 0.013 | 758 / 38034 |

Session-horizon weights (fold 0, standardised, reading only): bias -3.57, prevLists -0.01, prevRankScore 0.00, favLists 0.06, favRankScore 0.07, onLb 0.11, favShare 0.18, mapNear 0.31, onMap -0.35, logPeople 0.42, logListed -0.25, sessPlays 0.43, sessProper 0.43, sessTracks -0.31, sessRepeat 0.10, sessReturn 0.15, sessLen -0.39, baseLogit 0.50.

### New loves: the two-stage ranker, pooled

| windows | new loves | candidate coverage | ranker @20 | lists @20 | popular @20 | ranker @50 | lists @50 | popular @50 |
|---|---|---|---|---|---|---|---|---|
| rolling, inside training | 8892 | 0.183 | 0.024 | 0.017 | 0.011 | 0.047 | 0.031 | 0.024 |
| held-out 8 weeks | 812 | 0.158 | 0.017 | 0.016 | 0.006 | 0.036 | 0.030 | 0.015 |

### Doors: list facts at the first play

Start reasons do not exist here, so only the list facts are read: whether the artist played just before (same session) lists the new artist, and whether a recent favourite does. New loves are compared with up to 5 artists the same person tried and dropped (at most 2 listens ever), first heard nearest in time, and the favourite fact also with never-played ListenBrainz artists named on as many lists.

| fact | people | new loves | controls | ratio [95%] | wins/losses/ties | sign p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|
| came on straight after an artist that lists it | 299 | 0.156 | 0.110 | 1.42 [1.289, 1.543] | 180/103/6 | <0.001 | <0.001 | more common for new loves, but under the 1.5 bar |
| a recent favourite lists it | 302 | 0.417 | 0.224 | 1.87 [1.743, 2.011] | 255/39/4 | <0.001 | <0.001 | more common for new loves, survives |
| a recent favourite lists it in its top 25 | 302 | 0.222 | 0.091 | 2.45 [2.211, 2.730] | 256/33/9 | <0.001 | <0.001 | more common for new loves, survives |
| the artist just before was a recent favourite | 302 | 0.312 | 0.224 | 1.39 [1.338, 1.460] | 257/44/0 | <0.001 | <0.001 | more common for new loves, but under the 1.5 bar |
| a recent favourite lists it (vs never-played artists) | 303 | 0.420 | 0.080 | 5.27 [4.939, 5.643] | 302/0/0 | <0.001 | <0.001 | more common for new loves, survives |

The list-leak check, a recent favourite lists it vs never-played artists, by listening volume:

| people | n | new loves | controls | ratio [95%] | wins/losses |
|---|---|---|---|---|---|
| lightest third | 101 | 0.514 | 0.114 | 4.49 [4.001, 5.112] | 101/0 |
| heaviest third | 100 | 0.392 | 0.070 | 5.58 [4.942, 6.261] | 99/0 |

33396 new loves and 615797 tried-and-dropped artists. Median days from first listen to the 10th: 7.001.

### Traits: stability

Each trait on the two halves of every person's training weeks, correlated across people, with bootstrap intervals over people. Traits that need skips, play lengths or a time zone cannot be measured here.

| group | trait | n | Pearson r [95%] | Spearman rho [95%] | reading |
|---|---|---|---|---|---|
| adventurousness | newShare | 297 | 0.754 [0.680, 0.819] | 0.718 [0.647, 0.780] | stable, interval above 0.5 |
| adventurousness | spread | 298 | 0.876 [0.841, 0.908] | 0.869 [0.828, 0.901] | stable, interval above 0.5 |
| loyalty | oldFriends | 276 | 0.811 [0.748, 0.863] | 0.811 [0.750, 0.862] | stable, interval above 0.5 |
| patience | finished | 0 | - [-, -] | - [-, -] | not measurable here |
| patience | skipSeconds | 0 | - [-, -] | - [-, -] | not measurable here |
| clock | clockHour | 0 | - [-, -] | - [-, -] | not measurable here |
| clock | nightShare | 0 | - [-, -] | - [-, -] | not measurable here |
| clock | clockSpread | 298 | 0.720 [0.644, 0.783] | 0.704 [0.628, 0.779] | stable, interval above 0.5 |
| clock | clockRegular | 297 | 0.725 [0.652, 0.787] | 0.704 [0.634, 0.770] | stable, interval above 0.5 |
| era turnover | turnover | 260 | 0.802 [0.755, 0.847] | 0.781 [0.721, 0.830] | stable, interval above 0.5 |

### Traits: how many dials

301 people with every measurable trait. Each trait is turned into normal scores, the correlation matrix is decomposed, and Horn's parallel analysis (1,000 column shuffles, 95th percentile) decides how many components carry more than noise. Loadings are varimax-rotated, with 95% intervals from 500 bootstrap resamples of people (components matched by congruence).

| component | observed eigenvalue | 95th percentile under shuffling | kept |
|---|---|---|---|
| 1 | 3.100 | 1.272 | yes |
| 2 | 1.863 | 1.150 | yes |
| 3 | 0.475 | 1.070 | no |
| 4 | 0.271 | 1.006 | no |
| 5 | 0.173 | 0.944 | no |
| 6 | 0.117 | 0.877 | no |

Number of components kept in 200 bootstrap resamples: 2: 100%.

| trait | component 1 [95%] | component 2 [95%] |
|---|---|---|
| newShare | -0.881 [-0.903, -0.851] | -0.077 [-0.170, 0.005] |
| spread | -0.794 [-0.831, -0.746] | 0.016 [-0.090, 0.126] |
| oldFriends | 0.940 [0.921, 0.954] | 0.015 [-0.061, 0.091] |
| clockSpread | 0.032 [-0.035, 0.101] | -0.963 [-0.973, -0.952] |
| clockRegular | 0.048 [-0.021, 0.116] | 0.964 [0.951, 0.973] |
| turnover | -0.898 [-0.919, -0.874] | 0.043 [-0.025, 0.112] |

Correlations between traits (normal scores):

| | newShare | spread | oldFriends | clockSpread | clockRegular | turnover |
|---|---|---|---|---|---|---|
| newShare | 1.000 | 0.564 | -0.798 | 0.021 | -0.120 | 0.724 |
| spread | 0.564 | 1.000 | -0.671 | -0.032 | -0.026 | 0.595 |
| oldFriends | -0.798 | -0.671 | 1.000 | -0.001 | 0.041 | -0.819 |
| clockSpread | 0.021 | -0.032 | -0.001 | 1.000 | -0.858 | -0.066 |
| clockRegular | -0.120 | -0.026 | 0.041 | -0.858 | 1.000 | -0.008 |
| turnover | 0.724 | 0.595 | -0.819 | -0.066 | -0.008 | 1.000 |

### What survives in round 3

Survive the adjustment: 2 next 8 weeks' artists (NDCG@10): decay vs all-time; 2 next 8 weeks' artists (NDCG@10): decay vs last 4 weeks; 4 when you listen (bits per play, UTC hour of the week): tuned clock vs uniform; 4 when you listen (bits per play, UTC hour of the week): tuned clock vs all-time clock; 6 new loves, rolling windows (recall@50): two-stage ranker vs popular; 6 new loves, rolling windows (recall@50): two-stage ranker vs plain lists; 6 new loves, rolling windows (recall@50): plain lists vs popular; 6 new loves, rolling windows (recall@20): two-stage ranker vs popular; 9 will it stick, after the first session (AUC, training period): model vs own base rate; 9 will it stick, after the first session (AUC, training period): model vs popularity; 9 will it stick, after the first session (AUC, training period): model vs the model without ListenBrainz list features; 9 will it stick, after the first session (average precision, training period): model vs popularity; 9 will it stick, after the first session (Brier, training period): model vs own base rate (lower is better); 9 will it stick, after the first session (AUC, held-out 8 weeks): model vs own base rate; 9 will it stick, after the first session (AUC, held-out 8 weeks): model vs popularity; 10 will it stick, after 7 days (AUC, training period): model vs first-week play counts; 10 will it stick, after 7 days (AUC, training period): model vs popularity.

Significant only before adjustment: none.

Weak (one test passes, the other does not): 6 new loves, held-out 8 weeks (recall@50): two-stage ranker vs popular; 10 will it stick, after 7 days (AUC, held-out 8 weeks): model vs first-week play counts.

No difference or worse: 8 eras (NDCG@10): era-aware forecast vs decay (no difference); 8 eras (NDCG@10): current era only vs decay (worse than baseline).

Doors: came on straight after an artist that lists it: more common for new loves, but under the 1.5 bar; a recent favourite lists it: more common for new loves, survives; a recent favourite lists it in its top 25: more common for new loves, survives; the artist just before was a recent favourite: more common for new loves, but under the 1.5 bar; a recent favourite lists it (vs never-played artists): more common for new loves, survives.
