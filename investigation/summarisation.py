# from bert_score import BERTScorer
import time

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    [
        "rouge1",
        #    "rougeL"
    ],
    use_stemmer=True,
)

summaries = [
    (
        "I still have contact with an old ex's friends but can't stand to see or talk to him. His friends are really nice ,so how do I tell them I possibly want to unfriend them on Facebook because of him?",
        "og",
    ),
    (".", "null"),
    (
        "This is a test sentence to check that the score is not total bullshit. Should it work? Lets see.",
        "irrelevant",
    ),  # irrelevant
    (
        "After a difficult breakup 2 years ago, I am considering cutting contact with my ex's friends to avoid being reminded of him and the pain of that time, but I don't want to insult them. I need advice on whether distancing myself from his friends is the right thing to do in order to move on healthily.",
        "claude",
    ),  # claude summary
    (
        "I'm friend's with my ex's friends, we get on well. But they remind me of him and that causes me pain. I can't decide whether to cut contact with them or not.",
        "my",
    ),  # my summary
    (
        "I went through a tough breakup almost three years ago, which worsened my depression at the time. Now that I've made progress in my life, I'm considering cutting off contact with my ex's friends and avoiding any reminders of him, but I'm unsure how they will react and worried about potential awkwardness.",
        "chatgpt",
    ),  # chatgpt
    (
        "As a woman who has been through a challenging breakup, I am seeking advice on how to handle my current situation where I still have contact with certain people from my ex-partner's life despite trying to move on. Specifically, I want to cut off contact with his friends (both girls) and avoid seeing him in public settings due to the emotional turmoil it causes me. I am concerned about potential awkwardness or insulting them if I do so, but ultimately believe it is necessary for my own well-being.",
        "xwin",
    ),  # xwin
    (
        "I broke up with my ex two years ago, and now I'm thinking of ghosting his friends and avoiding him because they're all just annoying and I don't want to deal with any of them anymore. I'm sure they won't care at all, and there won't be any awkwardness because they're not important to me.",
        "bad chatgpt",
    ),
    (  # bad chatgpt summary
        "I think that cutting off contact with these girls is a bad idea because it's best for your mental health. It's important to take care of yourself and avoid getting hurt again. If they accept it then it will be easier, but if not then it will be awkward. I don't think there will be any insulted because they are friends with you. They might say something mean to me though. But I guess that's just a possibility.",
        "tinyllama",
    ),  # tinyllama (garbage)
    (
        "I’ve been through a lot in the past few years, from a painful breakup to cutting ties with my mother, and now I’m contemplating whether to sever connections with two female friends who are close to my ex. Despite the progress I’ve made in my life, seeing him or his pictures still triggers painful memories, and I’m unsure if distancing myself from these friends is the right move or how they’ll react.",
        "gpt4",
    ),  # gpt4, best
    (
        "I have contact with an old ex's friends but can't stand to see or talk to him. His friends are really nice ,so how do I tell them I possibly want to unfriend them on Facebook because of him?",
        "og 1 word change",
    ),
]
# summaries = [
#     "I still have contact with an old ex's friends but can't stand to see or talk to him. His friends are really nice ,so how do I tell them I possibly want to unfriend them on Facebook because of him?"
# ] * len(references)
references = [
    "SUBREDDIT: r/relationships TITLE: I (f/22) have to figure out if I want to still know these girls or not and would hate to sound insulting POST: Not sure if this belongs here but it's worth a try. Backstory: When I (f/22) went through my first real breakup 2 years ago because he needed space after a year of dating roand it effected me more than I thought. It was a horrible time in my life due to living with my mother and finally having the chance to cut her out of my life. I can admit because of it was an emotional wreck and this guy was stable and didn't know how to deal with me. We ended by him avoiding for a month or so after going to a festival with my friends. When I think back I wish he just ended. So after he ended it added my depression I suffered but my friends helped me through it and I got rid of everything from him along with cutting contact. Now: Its been almost 3 years now and I've gotten better after counselling and mild anti depressants. My mother has been out of my life since then so there's been alot of progress. Being stronger after learning some lessons there been more insight about that time of my life but when I see him or a picture everything comes back. The emotions and memories bring me back down. His friends (both girls) are on my facebook because we get along well which is hard to find and I know they'll always have his back. But seeing him in a picture or talking to him at a convention having a conversation is tough. Crying confront of my current boyfriend is something I want to avoid. So I've been thinking that I have to cut contact with these girls because it's time to move on because it's healthier. It's best to avoid him as well. But will they be insulted? Will they accept it? Is there going to be awkwardness? I'm not sure if it's the right to do and could use some outside opinions. TL;DR:"
    # .split(
    #     "."
    # )
    # "I still have contact with an old ex's friends but can't stand to see or talk to him. His friends are really nice ,so how do I tell them I possibly want to unfriend them on Facebook because of him?"
] * len(summaries)

# s = time.time()
# scorer = BERTScorer(lang="en")
# print(time.time() - s)

s = time.time()
# summ_scores = scorer.score(
#     refs=references, cands=[s[0] for s in summaries], verbose=True
# )
summ_scores = []
for i, summary in enumerate(summaries):
    summ_scores.append(
        (
            scorer.score(target=references[i], prediction=summary[0]),
            summary[1],
        )
    )

print(time.time() - s)
print(summ_scores)
# print(
#     [
#         (b - bert_scores[2][1]) / (bert_scores[2][0] - bert_scores[2][1])
#         for b in bert_scores[2]
#     ]
# )

# rank summaries by bert score
ranks = []
for i, summary in enumerate(summaries):
    # ranks.append((summ_scores[2][i], summary[1]))
    ranks.append((summ_scores[i][0]["rouge1"].fmeasure, summary[1]))
ranks.sort(reverse=True, key=lambda x: x[0])

# print the rankings
for rank in ranks:
    print(rank)
