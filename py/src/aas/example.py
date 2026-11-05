"""The v1 default readers, and one made-up person to run them on.

    from aas import example, profile
    prof = profile.build(example.synthetic_person(), example.default_readers())
    print(profile.dump(prof))

Everything in `synthetic_person` is invented for this file: the writing, the artist and track names and the
listening history. No real person's data is used. `scripts/example_profile.py` writes what v1 returns for this
person to `docs/example_profile.md`, and `tests/test_example.py` runs the same pipeline end to end.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from .person import Person, Play, TextItem


def default_readers() -> list:
    """The v1 defaults: text Big Five (bag of words, Essays), text MBTI (forum), music, and the zero-data fused
    reader over the two text readers. The distilled text reader is not here: it failed its pre-registered test
    (docs/distill.md). Music is described and never fused, because no link from it to the Big Five is tested."""
    from .fusion import FusedReader
    from .readers.music import MusicReader
    from .readers.text import TextBigFiveReader, TextMBTIReader

    bf, mb = TextBigFiveReader(), TextMBTIReader()
    return [bf, mb, MusicReader(), FusedReader([bf, mb])]


# ---------------------------------------------------------------- writing (all invented)

REDDIT = [
    "Honestly the best thing I did for my commute was swapping to wider tyres. 32mm at lower pressure and the canal path stopped rattling my teeth out. Took me two years to try it because I assumed narrow meant fast.",
    "I work nights three times a week so my sleep is a mess regardless, but blackout blinds and a fan for noise made a bigger difference than any supplement I tried. Not medical advice, just what worked for me.",
    "My sourdough was flat for months. What fixed it was not the flour or the starter, it was letting the dough go longer in a warmer spot. My kitchen is cold in winter and I just never accounted for that.",
    "Bought a 1984 steel frame off a neighbour for twenty quid. Bottom bracket was seized solid. Spent a whole Sunday with penetrating oil and a borrowed tool and it finally came out. Weirdly one of my best days this year.",
    "I keep a small notebook of every repair with the date and what parts I used. Sounds fussy but the next time something squeaks I know exactly what I touched last and when.",
    "Hot take maybe but I don't think group rides are for everyone. I like riding alone with nobody to keep up with and nobody waiting for me. I get enough people at work.",
    "Agree with the comment above about checking the chain first. Nine times out of ten the noise people describe is a dry chain or a stiff link, not the gears.",
    "We tried the meal prep thing for a month. It saved money but by Thursday I could not look at another tray of rice and roasted vegetables. Now I cook two bigger meals a week and freeze half.",
    "Not sure this helps but when I moved here I didn't know anyone. I joined the library's repair cafe on Saturdays and that is how I met most of the people I now see regularly.",
    "This thread is making me anxious about my own tyres lol. Just went and checked and the rear has a cut I hadn't noticed. Thanks I guess.",
    "Pharmacy hours are rough. Twelve hour shifts and you still go home thinking about whether you double checked that one dose. Most days are fine. Some days I replay everything on the ride home.",
    "If your bread is gummy inside, let it cool completely before cutting. I know it's torture. It keeps cooking as it cools.",
    "I don't really do resolutions but this year I wanted to ride to the coast and back in a day. Did it in September. 140km, two punctures, one very large slice of cake at the halfway point.",
    "Old frames are lovely but be honest with yourself about sizing. I bought one that was too big because it was pretty and rode it for a year with a sore back before I admitted it.",
    "My partner thinks I spend too long reading reviews before buying anything. I think buying the wrong pump twice is worse than reading for an evening. We are both probably right.",
    "Rain gear advice: get the cheap overshoes, skip the expensive jacket. Wet feet ruin a ride far more than a damp back does.",
    "I like that this sub is mostly people fixing things rather than buying things. Feels rarer than it should.",
    "Had a patient's relative shout at me today over a delay that wasn't anyone's fault. I stayed calm, sorted it, then sat in the car for ten minutes afterwards. Riding home helped.",
    "Tried a rye loaf for the first time. Dense, sour, slightly weird, I loved it. Nobody else in the house did, which means more for me.",
    "You can absolutely learn to true a wheel at home. Zip ties on the frame as a guide and go a quarter turn at a time. Patience matters more than tools.",
    "Moved my riding to early mornings after night shifts. Empty roads, cold air, nobody around. It's the closest thing I have to meditation.",
    "If anyone local wants a spare set of mudguards I have two that don't fit anything I own. Happy to leave them out for collection.",
    "I keep thinking I should try racing just to see. Then I remember how much I like stopping for coffee whenever I want.",
    "Planning ahead is half the fun for me. I had the coast route on paper for weeks before I rode it, with every water stop and two bail out stations marked.",
    "The repair cafe fixed forty three things last Saturday. Toasters, a lamp, a child's scooter, six bikes. People get oddly emotional when something they love works again.",
    "Not to be that person but please use lights in winter even on lit roads. I nearly clipped someone on the path last night who was all in black with nothing on.",
    "Got my first proper bread oven thermometer and realised my oven runs about fifteen degrees cold. Explains a lot.",
    "Worried I'm overthinking this but is it normal for a new chain to skip on an old cassette? Asking before I buy another part I may not need.",
    "Update: it was the cassette. Replaced both, shifting is perfect, and I've added it to the notebook.",
    "I'm not a big talker at work but I do like training the new technicians. Explaining something clearly to someone who wants to learn it is satisfying.",
]

TWEETS = [
    "Night shift done. Sunrise ride home along the river and not a single car. Best part of the week.",
    "Flat number two of the month. Patched it at the roadside in the rain like a professional, then cried a bit like an amateur.",
    "The starter has survived another week. I'm unreasonably proud of a jar of flour and water.",
    "Reminder to myself: check the dose, check it again, then go home and stop checking it in your head.",
    "Library repair cafe this Saturday. Bring your broken lamps and your squeaky bikes.",
    "New tyres, same old legs.",
    "Made rye bread. The house has opinions. I do not care.",
    "Planned the route, packed the spare tube, charged the lights. Tomorrow I find out if I'm as fit as I think I am.",
    "Rode to the coast and back. 140km. Legs are gone, heart is full, cake was excellent.",
    "Why does every bolt on an old bike come out with a fight and every bolt on a new bike come out by itself",
    "Quiet weekend planned. Bread, a short ride, a long book, nobody needing anything from me. Perfect.",
    "Cold mornings are back. Overshoes on, gloves on, still the best way to start a day.",
    "Fixed a neighbour's kid's scooter and got paid in a drawing of me as a dinosaur. Fair rate.",
    "Twelve hours on the ward, home by eight, asleep by nine. Glamour.",
    "Somebody left a perfectly good steel frame by the bins. It's in my hallway now. My partner has questions.",
    "Trued my first wheel. Took two hours. It is very slightly wobbly and I love it.",
    "Oven runs cold, which explains three years of pale crusts. Small victories.",
    "Stood at the top of the hill for ten minutes just looking at the fog in the valley. Late for nothing.",
    "Spent the evening reading tyre reviews instead of going to bed. Regret nothing. Will regret it at 6am.",
    "Winter commuting checklist: lights, mudguards, patience.",
    "Good shift. Nobody shouted, every order made sense, and the coffee machine worked. Rare alignment.",
    "Started a notebook for the bread too. It's notebooks all the way down.",
]

BLOG = (
    "Why I fix old bikes. I started fixing bicycles because I could not afford a new one, and I kept doing it long "
    "after I could. A thirty year old steel frame has a history you can read if you look closely: a dent where it "
    "fell against a wall, paint touched up in a slightly wrong colour, a bottle cage bolt that someone replaced with "
    "a screw from a kitchen drawer. Every one of those is a decision somebody made on an ordinary afternoon. When I "
    "take a bike apart I am in conversation with all of them. The work itself suits me. It is quiet, it is slow, and "
    "it rewards doing things in the right order. You clean before you inspect, you inspect before you buy parts, and "
    "you write down what you did so the next person, usually me, does not have to guess. At work I spend twelve hours "
    "making sure nothing is missed, and it is tiring because the stakes are other people. In the hallway with a bike "
    "upside down the stakes are mine. If I get a cable wrong I adjust it. Nobody is hurt and I learn something. "
    "There is also the moment when it works. You spin the cranks and the chain runs silently, the wheel turns true "
    "and the brakes bite cleanly. I have felt that hundreds of times now and it has not got old. Last month I gave a "
    "rebuilt bike to a student who had been walking forty minutes each way to her college. She rode it round the car "
    "park twice and came back grinning. I will probably never see that bike again, and I like that it is out there, "
    "carrying someone, with my notes on it written in pencil under the saddle. I do not think I will ever stop. There "
    "are too many good frames by the bins and too many people who need a way to get around. And I am happiest with "
    "oil on my hands and a problem in front of me that I can actually solve."
)

SEARCHES = ["bottom bracket tool 68mm", "rye starter hydration", "coast road closures saturday",
            "night shift sleep tips", "tubeless vs inner tube commuting"]


def _texts(start: datetime) -> list[TextItem]:
    out = []
    for i, t in enumerate(REDDIT):
        out.append(TextItem(start + timedelta(days=9 * i + 2), "reddit", "comment", t))
    for i, t in enumerate(TWEETS):
        out.append(TextItem(start + timedelta(days=12 * i + 5), "x", "tweet", t))
    out.append(TextItem(start + timedelta(days=200), "blog", "blog", BLOG))
    for i, t in enumerate(SEARCHES):
        out.append(TextItem(start + timedelta(days=30 * i + 7), "google_search", "search", t))
    return out


# ---------------------------------------------------------------- listening (all invented)

_SYL = ["mar", "lo", "ven", "ka", "tide", "ro", "fen", "ash", "mi", "ber", "lune", "dor", "sa", "wick", "el",
        "nor", "pa", "glass", "ri", "hol"]


def _artists(rng: np.random.Generator, n: int) -> list[str]:
    names, seen = [], set()
    while len(names) < n:
        k = int(rng.integers(2, 4))
        w = "".join(rng.choice(_SYL, k)).capitalize()
        name = (f"The {w}s" if rng.random() < 0.3 else w + (f" {rng.choice(_SYL).capitalize()}" if rng.random() < 0.5 else ""))
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _plays(start: datetime, days: int, seed: int) -> list[Play]:
    """A year of plays: a slowly drifting core of favourites, a steady trickle of new artists, listening mostly on
    the ride to and from shifts and in the evening, about one play in six skipped."""
    rng = np.random.default_rng(seed)
    pool = _artists(rng, 400)
    core = list(range(25))                      # current favourites (indices into pool)
    next_new = 25
    plays: list[Play] = []
    for d in range(days):
        day = start + timedelta(days=d)
        if d % 21 == 0 and d:                  # every three weeks two favourites drift out and two artists come in
            for _ in range(2):
                core.pop(int(rng.integers(len(core))))
                core.append(next_new)
                next_new += 1
        night_week = (d // 7) % 2 == 0
        n = int(rng.poisson(26))
        for _ in range(n):
            # listening blocks: commute (around 07:30 or 20:00 after nights), evening, late night
            block = rng.choice(3, p=[0.45, 0.4, 0.15])
            centre = (20.0 if night_week else 7.5) if block == 0 else (19.5 if block == 1 else 23.0)
            hour = float(np.clip(rng.normal(centre, 0.9), 0, 23.98))
            when = day + timedelta(hours=hour)
            if rng.random() < 0.12:            # something new or from the long tail
                a = int(rng.integers(25, len(pool)))
            else:
                a = core[min(int(rng.exponential(7)), len(core) - 1)]
            length = int(rng.integers(150, 330)) * 1000
            skipped = rng.random() < 0.16
            ms = int(rng.integers(2, 40)) * 1000 if skipped else length
            plays.append(Play(when=when, artist=pool[a], track=f"{pool[a]} track {int(rng.integers(1, 15))}",
                              ms_played=ms, skipped=skipped, reason_start="clickrow" if rng.random() < 0.1 else "trackdone",
                              reason_end="fwdbtn" if skipped else "trackdone", shuffle=bool(rng.random() < 0.5),
                              source="spotify", country="GB"))
    plays.sort(key=lambda p: p.when)
    return plays


def synthetic_person(seed: int = 20261105, with_music: bool = True) -> Person:
    """An invented hospital pharmacist who rides, bakes and fixes old bikes: 30 Reddit comments, 22 posts on X, one
    blog post, five searches, and (by default) a year of Spotify-style plays."""
    start = datetime(2025, 11, 1, tzinfo=timezone.utc)
    p = Person(texts=_texts(start), sources=["reddit (synthetic)", "x (synthetic)", "blog (synthetic)",
                                             "google search (synthetic)"],
               notes=["every item in this person is invented for the example; no real person's data"])
    if with_music:
        p.plays = _plays(start, 365, seed)
        p.sources.append("spotify extended history (synthetic)")
    return p
