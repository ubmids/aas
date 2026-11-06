# Related work: identity from listening, and identity in a few sentences

Literature check, 5 Nov 2026. Every paper below was looked up on the web on that date and the link was checked to
resolve to the paper (or its publisher, DOI or repository page). Numbers are quoted from the paper or its abstract.
Where a number could not be confirmed it is left out rather than recalled from memory.

## What we have, for comparison

From ListenBrainz histories (333 fresh listeners in the confirmation round, 303 in the exploratory round), using only
behavioural measures and no artist or track identities:

1. A percentile fingerprint over 57 behavioural families (discovery rate, loyalty to old favourites, concentration,
   repetition, runs, clock spread, weekly rhythm, sessions, streaks) picks the second half of a person's history out
   of a lineup of 5 with the first half 89.5% of the time [87.1, 91.7]. Chance is 20%.
2. Three human readable threshold statements ("you meet 400+ new artists a month"), each true of at most 20% of
   others and chosen on the first half, do it 67.5% of the time [63.8, 71.0], permutation p = 0.001. A Barnum
   control (statements that are true of most people) reaches 31.7%.
3. 74.8% of people have 3 or more statements that hold on the other half and stay rare.

Two properties set this apart from most prior work, and both matter below. The features carry no item identity (no
artist, no song, no genre), and the test is across time (first half against second half), not uniqueness inside one
snapshot.

A note on scale. A lineup of 5 can show at most log2 5 = 2.3 bits of identity. Singling one person out among 333
needs about 8.4 bits, among 1 million about 20 bits, and among everyone alive about 33 bits. A threshold statement
true of 20% of people carries at most 2.3 bits, one true of 10% at most 3.3 bits, and only if it is independent of
the others and stays true over time. So three statements at the 20% line carry at most about 7 bits even in the
best case. Our current numbers say "much better than chance in a small lineup"; they do not yet say anything about
singling someone out in a crowd.

## 1. Re-identification from music listening and music taste

### Matrosova, Moussallam, Louail and Bodini (2024)

Depict or Discern? Fingerprinting Musical Taste from Explicit Preferences. Transactions of the International Society
for Music Information Retrieval 7(1), 15 to 29. https://doi.org/10.5334/tismir.158

Deezer data: 1 million active users with their explicit likes, and 50,000 users with a year of streaming logs. Two
kinds of fingerprint: a distinctive one (the smallest set of items a user has that no one else has as a subset, a
set cover style problem) and a representative one (k medoids summary for recommendation). Results: 90% of users are
unique by their liked songs and 60% by liked artists; 45% are singled out by one song; the average distinctive
fingerprint is 2.3 artists or 1.4 songs (median 1 song); 97% are unique by the artists they streamed in a year, and
one month of logs makes 95% unique.

How we compare. This is the closest prior work and it already establishes that listening is a fingerprint. Their
uniqueness is far stronger than ours (95% unique among 50,000 against our 67.5% to 89.5% in lineups of 5) because
item identities are enormously sparse: almost nobody else has liked your exact obscure song. Their fingerprints are
item based and measured inside one period; they do not match one period of a person against another. Ours are item
free and cross time. Their "distinctive fingerprint" is also the closest existing analogue of "how many statements
does it take", but the statements are items ("liked song X"), not readable claims about behaviour, and nothing checks
that they stay true later.

### Hirschprung and Leshman (2021)

Privacy disclosure by de-anonymization using music preferences and selections. Telematics and Informatics 59,
101564. https://doi.org/10.1016/j.tele.2021.101564 (press summary:
https://techxplore.com/news/2021-04-playlist-ill.html)

About 150 undergraduates in 4 groups of about 35. Each person submitted three songs from their favourite playlist and
group members, who did not know each other well, had to match people to songs. Matching was 80% to 100% accurate.
Their argument: music selection is a quasi-identifier and should be covered by privacy law.

How we compare. A human matcher, three items, groups of about 35. Like ours it is about a handful of readable clues,
but the clues are song titles (item identity) and there is no time split. Ours uses item free statements and a
machine matcher; a human matcher on our statements would be a direct counterpart to their design.

### Calandrino, Kilzer, Narayanan, Felten and Shmatikov (2011)

"You Might Also Like:" Privacy Risks of Collaborative Filtering. IEEE Symposium on Security and Privacy 2011.
https://www.ieee-security.org/TC/SP2011/PAPERS/2011/paper015.pdf

Infers individual transactions from the public outputs of recommenders (related item lists) given some auxiliary
knowledge of a target, evaluated on Hunch, Last.fm, LibraryThing and Amazon. Relevant because it shows Last.fm data
was already treated as privacy sensitive; it is an inference attack on items, not a fingerprint of behaviour.

### Datasets that make this possible

Vigliensoni and Fujinaga (2017), The Music Listening Histories Dataset, ISMIR 2017.
https://archives.ismir.net/ismir2017/paper/000180.pdf. More than 27 billion time stamped Last.fm logs, with user
profiling features describing listening behaviour. The LFM-1b and LFM-2b sets (Schedl and colleagues) are the other
large scrobble collections; LFM-1b covers about 120,000 users and 1.1 billion events over nine years (as described in
https://transactions.ismir.net/articles/10.5334/tismir.39). None of these dataset papers runs a re-identification
test; they are where a 1 of N test at scale could be run.

## 2. Uniqueness and re-identification of behaviour in general

### Narayanan and Shmatikov (2008)

Robust De-anonymization of Large Sparse Datasets. IEEE Symposium on Security and Privacy 2008, 111 to 125.
https://doi.org/10.1109/SP.2008.33 (preprint https://arxiv.org/abs/cs/0610105)

The Netflix Prize attack. With 8 movie ratings (2 of which may be wrong) and dates known to within 14 days, 99% of
the 500,000 records are uniquely identified; 2 ratings and dates within 3 days suffice for 68%. Matching uses a
weighted similarity score with an "eccentricity" test (the best match must beat the second best by 1.5 standard
deviations). They linked records to IMDb profiles.

How we compare. The canonical result that sparse preference data is identifying. Item based, single snapshot. The
eccentricity test is a useful idea for us: it turns lineup accuracy into a confidence that a match is right.

### de Montjoye, Hidalgo, Verleysen and Blondel (2013)

Unique in the Crowd: The privacy bounds of human mobility. Scientific Reports 3, 1376.
https://doi.org/10.1038/srep01376

15 months of phone location data for about 1.5 million people. Four spatio temporal points make 95% unique; two make
more than half unique. Uniqueness falls only slowly as data are coarsened (roughly the tenth power of resolution).

### de Montjoye, Radaelli, Singh and Pentland (2015)

Unique in the shopping mall: On the reidentifiability of credit card metadata. Science 347(6221), 536 to 539.
https://doi.org/10.1126/science.1256297

Three months of card records for 1.1 million people. Four spatio temporal points re-identify 90%; knowing the price
raises risk by 22% on average.

How we compare to both. The "unique in the crowd" method counts how many random points a person needs before nobody
else matches. Nobody we found has run that method on media consumption with item free behavioural measures. Our
statements are the readable cousin of their points, but we have not yet measured uniqueness against a crowd.

### Creţu, Monti, Marrone, Dong, Bronstein and de Montjoye (2022)

Interaction data are identifiable even across long periods of time. Nature Communications 13, 313.
https://doi.org/10.1038/s41467-021-27714-6 (open copy https://pmc.ncbi.nlm.nih.gov/articles/PMC8789822/)

Learns a behavioural profile from who people call and message (a 2 hop interaction graph, geometric deep learning
with triplet loss) and matches one week to a later week. Among more than 40,000 people it picks the right person 52%
of the time; 24% are still identifiable after 20 weeks. They argue that re-pseudonymised behavioural data is
therefore personal data under GDPR.

How we compare. This is the closest in design: behaviour rather than items, a profile from one period matched to a
later period, top 1 accuracy reported. They do it at 1 of 40,000 with a learned, unreadable embedding; we do it at 1
of 5 with readable measures and sentences. Our numbers are much weaker in scale. Their time decay curve (accuracy
against the gap between periods) is a figure we do not have yet and should.

### Oliveira, Yang, Griffiths, Bonnay and Kulshrestha (2023)

Browsing behavior exposes identities on the Web. arXiv:2312.15489. https://arxiv.org/abs/2312.15489

The four most visited web domains identify 95% of people; 80% can be re-identified across separate time slices.
Item based (domains) but cross time, which makes it a good template for the 1 of N, cross period test we want.

### Olejnik, Castelluccia and Janc (2012)

Why Johnny Can't Browse in Peace: On the Uniqueness of Web Browsing History Patterns. HotPETs 2012.
https://www.petsymposium.org/2012/papers/hotpets12-4-johnny.pdf

368,284 users. 69% have a unique history; with 4 or more detected sites, 97% are unique. Testing 50 pages
fingerprints 42% of users, 500 pages 70%. Category level (indirect) history also fingerprints. That last point is
the nearest analogue to ours: coarse descriptions of behaviour, not the items themselves, still identify.

### Sekara, Alessandretti, Mones and Jonsson (2021)

Temporal and cultural limits of privacy in smartphone app usage. Scientific Reports 11, 3861.
https://doi.org/10.1038/s41598-021-82294-1

About 3.5 million people over a year. Four apps chosen by a popularity based strategy re-identify 91.2%; four at
random 21.8%. Uniqueness rises in summer, and app fingerprints drift slowly over time. The drift result is the
analogue of our stability question.

### Rocher, Hendrickx and de Montjoye (2019)

Estimating the success of re-identifications in incomplete datasets using generative models. Nature Communications
10, 3069. https://doi.org/10.1038/s41467-019-10933-3

A copula model estimates the chance that a specific person is correctly re-identified even from a small sample;
99.98% of Americans would be re-identified by 15 demographic attributes. This is the tool for going from our 333
people to statements about a population of millions without having the population.

### Sweeney (2000) and Golle (2006)

Sweeney, L. Simple Demographics Often Identify People Uniquely. Carnegie Mellon University, Data Privacy Working
Paper 3, 2000. https://dataprivacylab.org/projects/identifiability/paper1.pdf. ZIP code, sex and date of birth make
87% of the 1990 US population likely unique.

Golle, P. Revisiting the Uniqueness of Simple Demographics in the US Population. ACM Workshop on Privacy in the
Electronic Society (WPES) 2006. https://crypto.stanford.edu/~pgolle/papers/census.html. The same three facts make
63% unique on 2000 census data.

How we compare. These are the original "three plain facts" results, and they are the strongest rhetorical precedent
for "identity in k sentences". Their three facts are demographic and permanent; ours are behavioural and must be
re-earned on new data, which is why stability, not just rarity, has to be in our measure.

### Eckersley (2010)

How Unique Is Your Web Browser? Privacy Enhancing Technologies Symposium 2010, LNCS 6205.
https://link.springer.com/chapter/10.1007/978-3-642-14527-8_1 (EFF copy via
https://coveryourtracks.eff.org). About a million browsers; the fingerprint carries at least 18.1 bits of entropy,
18.8 bits with Flash or Java; 83.6% are unique, 94.2% with Flash or Java. He also reports bits per attribute
(surprisal of each component) and measures how fingerprints change between visits.

How we compare. This is the template for "bits of identity per sentence": he reports identifying information in bits
per attribute and in total. Nobody we found has done the same with readable behavioural sentences, or charged each
sentence for the chance it stops being true.

### Zhang, Fawaz, Ioannidis and Montanari (2012)

Guess Who Rated This Movie: Identifying Users Through Subspace Clustering. UAI 2012. https://arxiv.org/abs/1208.1544

Separates the people sharing one Netflix account from ratings alone. Relevant as evidence that behaviour inside a
single account is person specific; it is not a re-identification across datasets.

## 3. Stability of listening behaviour over time

Mok, Way, Maystre and Anderson (2022). The Dynamics of Exploration on Spotify. ICWSM 2022.
https://www.cs.toronto.edu/~ashton/pubs/spotify-exploration-dynamics-icwsm2022.pdf. 100,000 US Spotify listeners
followed over time. Younger listeners explore less and exploit more; people explore in bursts with seasonal cycles and
exploratory phases. Exploration and diversity are different traits.

Kim, Askin and Evans (2024). Disrupted routines anticipate musical exploration. PNAS 121(6), e2306549121.
https://doi.org/10.1073/pnas.2306549121. Over a hundred million streams: breaks in personal routine go with bursts of
musical exploration.

Park, Thom, Mennicken, Cramer and Macy (2019). Global music streaming data reveal diurnal and seasonal patterns of
affective preference. Nature Human Behaviour 3, 230 to 236. https://doi.org/10.1038/s41562-018-0508-z. 765 million
plays by 1 million people in 51 countries; shared diurnal and seasonal patterns.

Anderson, Gil, Gibson, Wolf, Shapiro, Semerci and Greenberg (2021). "Just the Way You Are": Linking Music Listening on
Spotify and Personality. Social Psychological and Personality Science 12(4), 561 to 572.
https://doi.org/10.1177/1948550620923228. Listening behaviour relates to Big Five traits.

Schedl and colleagues on temporal drift of genre preference on LFM-1b:
https://transactions.ismir.net/articles/10.5334/tismir.39.

How we compare. This literature describes listening as having both stable individual differences and strong, partly
shared, temporal variation (age, season, routine breaks). None of it tests whether the individual differences are
stable enough to re-identify a person, which is exactly what our split half lineup measures. It also warns about our
design: a first half against second half split lets age and life stage drift between halves, and exploration comes in
bursts, so discovery statements may be the least stable ones. Kim, Askin and Evans suggest routine breaks are when a
fingerprint would fail.

## 4. Readable, discriminative descriptions of a person

Dale, R. and Reiter, E. (1995). Computational Interpretations of the Gricean Maxims in the Generation of Referring
Expressions. Cognitive Science 19(2), 233 to 263. https://arxiv.org/abs/cmp-lg/9504020. The referring expression
problem: choose attributes that single out one object from a set of distractors. Finding the shortest such
description is NP hard; their incremental algorithm adds attributes greedily and humans over describe in the same
way. This is "identity in k plain sentences" in formal terms, applied to objects in a scene rather than people over
time, with no notion that an attribute may stop being true.

Motwani, R. and Xu, Y. Efficient Algorithms for Masking and Finding Quasi-Identifiers. DIMACS workshop on privacy
and confidentiality (a version also appears in the P3DM 2008 workshop proceedings,
https://userpages.cs.umbc.edu/kunliu1/p3dm08/proceedings/2.pdf). https://crypto.stanford.edu/portia/pubs/articles/MX1363903499.html.
Reduces the smallest attribute set that separates records to set cover over pairs of records, and gives greedy
approximation algorithms with guarantees. Our joint selection of statements is a set cover over (person, other
person) pairs, so greedy selection carries the usual logarithmic approximation guarantee, and this is the citation for
it.

Radlinski, Balog, Diaz, Dixon and Wedin (2022). On Natural Language User Profiles for Transparent and Scrutable
Recommendation. SIGIR 2022. https://arxiv.org/abs/2205.09403. Proposes a length bounded natural language summary of a
user as the representation a recommender works from.

Ramos, Rahmani, Wang, Fu and Lipani (2024). Transparent and Scrutable Recommendations Using Natural Language User
Profiles. ACL 2024, 13971 to 13984. https://doi.org/10.18653/v1/2024.acl-long.753. LLM generated NL profiles used for
recommendation, on par with standard recommenders in the warm start setting.

Sguerra, Epure, Lee and Moussallam (2025). Biases in LLM-Generated Musical Taste Profiles for Recommendation. RecSys
2025. https://arxiv.org/abs/2507.16708. 64 Deezer users rated 12 profiles generated from their own listening and 5
profiles of random other users on a 7 point scale; own profiles were rated clearly higher. Recognition varied with
mainstreamness and specialisation.

Staab, Vero, Balunović and Vechev (2024). Beyond Memorization: Violating Privacy via Inference with Large Language
Models. ICLR 2024. https://arxiv.org/abs/2310.07298. LLMs infer personal attributes from Reddit text with up to 85%
top 1 accuracy.

How we compare. The NL profile line (Radlinski, Ramos, Sguerra) optimises descriptions for recommendation or for
self recognition, not for telling one person from another, and does not report accuracy against the number of
sentences. Sguerra et al. is the nearest thing to a lineup with sentences: their random profiles act as foils. But
the judge is the person themself and the profiles are free text from an LLM. None of these papers asks how many
sentences it takes to single someone out, or measures bits per sentence.

## 5. Barnum controls

Forer, B. R. (1949). The fallacy of personal validation: a classroom demonstration of gullibility. Journal of
Abnormal and Social Psychology 44(1), 118 to 123. https://doi.org/10.1037/h0059240. 39 students rated an identical
sketch, assembled from horoscopes, as an accurate description of themselves.

Dickson, D. H. and Kelly, I. W. (1985). The "Barnum effect" in personality assessment: a review of the literature.
Psychological Reports 57(2), 367 to 382. https://doi.org/10.2466/pr0.1985.57.2.367 (open copy
https://cortecs.org/wp-content/uploads/2014/12/the-barnum-effect-in-personality-assessment-a-review-of-the-literature.pdf)

Suwanaposee, Gutwin, Chen and Cockburn (2023). "Specially For You": Examining the Barnum Effect's Influence on the
Perceived Quality of System Recommendations. CHI 2023. https://doi.org/10.1145/3544548.3580656. n = 492; identical
movie recommendations labelled personalised or not; no significant difference (personalised scored slightly lower).

How we compare. The Barnum literature tests whether people accept a generic description as theirs. Our Barnum
control is different and, we think, stricter: it asks whether generic statements identify a person to a blind
matcher, and they do much worse (31.7% against 67.5%). This is a useful methodological point. A matcher based test
cannot be fooled by the warm feeling Forer measured, so "true of you and few others, and still true later" is an
operational definition of a non Barnum statement. We found no paper that uses identification accuracy as the Barnum
control for personalised feedback.

## Answers

(a) Is a listening fingerprint already known? Yes. Matrosova et al. (2024) show that liked songs and streamed artists
make 90% to 97% of Deezer users unique and that one month of logs is enough for 95%; Hirschprung and Leshman (2021)
show three playlist songs let strangers match people; Narayanan and Shmatikov (2008) and Calandrino et al. (2011)
established the same for ratings and Last.fm style data. What we did not find is a fingerprint built only from item
free behaviour (rates, shares, rhythms, no artist or song identities) and tested across time. The general idea that
behaviour is stable enough to re-identify across periods is known (Creţu et al. 2022 for call graphs, Oliveira et al.
2023 for browsing, Olejnik et al. 2012 for category level browsing), so our 89.5% fingerprint result is best described
as a small scale extension of a known phenomenon to a new kind of feature. At lineups of 5 it is much weaker than the
prior art in scale; its strength is that no item identity is used.

(b) Is identity in k plain sentences new? As far as this search found, yes as a measured quantity. The pieces exist
separately: the minimal set problem (Dale and Reiter 1995 for referring expressions, Motwani and Xu for
quasi-identifiers, Matrosova et al. for minimal item sets), bits per attribute (Eckersley 2010), three plain facts
single out most people (Sweeney 2000, Golle 2006), and readable NL user profiles (Radlinski et al. 2022, Ramos et al.
2024, Sguerra et al. 2025). We found no paper that reports identification accuracy as a function of the number of
human readable behavioural sentences, charges each sentence for whether it stays true on new data, or reports bits of
identity per sentence. That is not proof that none exists; the referring expression and quasi-identifier literatures
are large, and a targeted search in PoPETs and ISMIR proceedings before submission is still worth doing.

(c) What would make it genuinely novel and publishable:

1. Scale. Move from lineups of 5 to 1 of N with N from 5 up to the full ListenBrainz population (tens of thousands
   of listeners in the public dumps), reporting top 1 and rank, and a Rocher style model to extrapolate to millions.
   Without this, reviewers will compare 67.5% of 5 with 95% of 50,000 and stop reading.
2. The k curve with an information account. Accuracy against k for k = 1 to 10 statements, chosen greedily as a set
   cover over (person, foil) pairs with the logarithmic guarantee, against random and best single choices. Report
   stable bits per sentence: the surprisal of the statement among others, discounted by the chance it is still true
   on the later half. Compare the observed curve with the independence upper bound (k times log2 of one over rarity)
   to show how much the sentences overlap.
3. The price of readability and of dropping items. On the same people, run item based fingerprints (top artists, as
   in Matrosova et al.) and the 57 family vector, and show how much identity readable item free sentences keep. A
   headline like "three sentences carry X of the Y bits" is the result nobody has.
4. Time decay. Accuracy against the gap between the two windows (weeks to years), as Creţu et al. did, rather than a
   single half against half split, and check whether routine breaks (Kim, Askin and Evans 2024) are where it fails.
5. Human matchers and self recognition with a Barnum arm. Strangers and the person themself pick from a lineup using
   the sentences, against generic sentences and sentences from another person (Forer style). Pre-registered, this is
   a CHI or CSCW paper on its own, and it directly answers the Sguerra et al. design with a discriminative test.
6. The privacy claim. Aggregate, item free statistics of the kind streaming services publish back to users (yearly
   summaries) are commonly treated as harmless. If a handful of them re-identify across time at 1 of N, that is a
   quasi-identifier finding with GDPR consequences in the line of Creţu et al. and Sekara et al. Venue: PoPETs or
   TISMIR, or ICWSM for the behavioural side.

Of these, 1 and 2 together are the minimum for a paper; 3 makes it distinctive; 5 is a second paper.

## The five most important citations

1. Matrosova, Moussallam, Louail, Bodini (2024), Depict or Discern? Fingerprinting Musical Taste from Explicit
   Preferences, TISMIR 7(1), https://doi.org/10.5334/tismir.158. The listening fingerprint prior art, and minimal
   item sets.
2. Creţu et al. (2022), Interaction data are identifiable even across long periods of time, Nature Communications 13,
   313, https://doi.org/10.1038/s41467-021-27714-6. The cross time behavioural re-identification design to match.
3. Narayanan and Shmatikov (2008), Robust De-anonymization of Large Sparse Datasets, IEEE S&P,
   https://doi.org/10.1109/SP.2008.33. Canonical media consumption de-anonymisation.
4. de Montjoye et al. (2013), Unique in the Crowd, Scientific Reports 3, 1376, https://doi.org/10.1038/srep01376,
   with the 2015 Science credit card paper. The uniqueness method for 1 of N.
5. Eckersley (2010), How Unique Is Your Web Browser?, PETS 2010. The bits per attribute template for bits per
   sentence.

Close behind: Dale and Reiter (1995) for the formal minimal description problem, and Sguerra et al. (2025) for
readable music profiles judged against random foils.
