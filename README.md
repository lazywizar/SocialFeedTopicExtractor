# Social Feed Topic Extractor

![](https://img.shields.io/github/tag/pandao/editor.md.svg) ![](https://img.shields.io/github/release/pandao/editor.md.svg)

A library to annotate social feed with topic. Topics can be used to filter out un-wanted messages from social feed. Library is not limited to social feed and can be used to extract topic of any text. Current implementation is used to gather Twitter feed and attach topic to it. Twitter clients in turn can use these topics to filter out non-desired posts from user's feed.

The code uses a set of pre-determined Topics and a few seed words for each topic.


###Features
- Support pulling large scala data from Twitter
- Support to build and fit custom data to create a GuidedLDA model
- Support to use the above created model to annotate any given text
- Pull latest `n` posts from Twitter, and add topic to each post

### Installing
Library is based off [GuidedLDA](http://github.com/vi3k6i5/GuidedLDA "GuidedLDA"). These are the packages needed to be installed.

    $ pip install guidedlda
    $ pip install nltk
    $ python -m nltk.downloader all
    $ pip install python-twitter

Twitter credentails need to be set to environment variables.
    $ export CONSUMER_KEY=<>
    $ export CONSUMER_SECRET=<>
    $ export ACCESS_TOKEN_KEY=<>
    $ export ACCESS_TOKEN_SECRET=<>

> For setting up Twitter app follow [Twitter API](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html "Twitter API")

## Documentation
Library supports 4 major funtionalities.
1. Download twitter posts for a given set of hashtags. Convert the data to LDAC format which could be used to create a DTM and hence used for fitting to LDA model.
2. Train a model and fit it to the given dataset and a set of topics and its corresponding keywords.
3. Use the above model to annotate topic to all posts in a downloaded file with one post per line.
4. Download lastest `n` twitter posts from the user's feed and add topic to them.

 >  \#4 is what would be most useful for any Twitter client. The output contains topic, twitter_feed_id and text of the post. Client can use the id and topic to select which posts to show and which to omit.

###Quick Start
There are two main command line options `--mode` or `-m` and `-i` or `--inputfile`

Supported modes are:
`Train`:  Create a model and fit it to a default dataset or custom created dataset. 
`Pull`:  Pulls feed from Twitter for a set of given hashtags
`TestOnline`:  Main method, pulls latest 20 feed from Twitter. Annotates each post with the top topic.
`TestOffline`: Annotates a already downloaded feed file with topic. This needs the inputfile to be specified.


The code uses the following default topics and a few seed words for each topic.
`{0: 'Sports', 1: 'Business', 2: 'Arts/Litrature', 3: 'Politics', 4: 'Misc'}` This can be easily changed by the clients as desired.


#####1. Train a default model using NYT (New York Times) dataset.
` $ python3 SocialTopicExtractor.py -m Train`

    Preparing a model by fitting NYT dataset from guidedLDA
    INFO:guidedlda:n_documents: 8447
    INFO:guidedlda:vocab_size: 3012
    INFO:guidedlda:n_words: 1221626
    INFO:guidedlda:n_topics: 5
    INFO:guidedlda:n_iter: 100
    WARNING:guidedlda:all zero column in document-term matrix found
    INFO:guidedlda:<0> log likelihood: -11481773
    INFO:guidedlda:<20> log likelihood: -9736069
    INFO:guidedlda:<40> log likelihood: -9642465
    INFO:guidedlda:<60> log likelihood: -9613654
    INFO:guidedlda:<80> log likelihood: -9597819
    INFO:guidedlda:<99> log likelihood: -9585385
    Topic [Sports]: game play team win season player second start point victory
    Topic [Business]: company percent market business price executive sell pay president plan
    Topic [Arts/Litrature]: life play man write book woman music world young thing
    Topic [Politics]: official state government issue political case states leader member public
    Topic [Misc]: place city home area building small live school house old


#####2. Annotate already download feed file with topics
`tw_latest_raw.txt` is a sample feed file included in the library which can be used to test. There should be one post per line.
`python3 SocialTopicExtractor.py -m TestOffline -i tw_latest_raw.txt`

    Cleaning up the input feed data file
    Extracting topics for these feeds
    (25, 2897)
    76
    Top topic: Misc [Post: '1069484641961226240 #SriLankan President #MaithripalaSirisena is considering dropping an attempt to dissolve the parliament to ease off… https://t.co/rW7XNRYXAl SriLankan MaithripalaSirisena']
    Top topic: Misc [Post: '1069483901838680064 Kamala Harris, the first Indian-origin US Senator, says she will decide whether to consider a presidential bid in 2… https://t.co/J7N84nEeZy']
    Top topic: Arts/Litrature [Post: '1069483856556892160   Qatar is to leave OPEC next month, the Gulf state’s new Energy Minister, Saad al-Kaabi, announced. https://t.co/ZSSU5h18ne']
    Top topic: Politics [Post: '1069483748033388545 RT @kdrajagopal: #ZakiaJafri appeal to be heard in January third week @the_hindu @abaruah64 ZakiaJafri']
    Top topic: Misc [Post: '1069483702705475585 RT @dperi84: Tempo of overseas deployments and exercises at all time high this year. Conducted 20 exercises with friendly foreign navies: @…']
    Top topic: Sports [Post: '1069483374744690689   RT @parthpunter: I spoke to the farmers and workers from Madhya Pradesh, who voted on 28th and made it to the #KisanMuktiMarch in Delhi the… KisanMuktiMarch']
    Top topic: Politics [Post: '1069482821020987392 MEDLINE Data Changes—2019. NLM Tech Bull. 2018 Nov-Dec New 2019 MeSH headings / Data Terms: Batman,Data Aggrega… https://t.co/sAV5M8MFfY']
    Top topic: Politics [Post: '1069482748983812096 Thanks for introducing us. Amit! https://t.co/cOiE9L8mf5']
    Top topic: Business [Post: '1069482040897269761 There is an urgent need for proper debate about when germline gene editing is warranted https://t.co/PG2r7KY1N5']
    Top topic: Misc [Post: '1069482034626670592 RT @FirstpostSports: After blazing start to his Test career, time for Rishabh Pant to put in hard shifts, writes @UjwalKS #IndvsAus  https:… IndvsAus']
    Top topic: Sports [Post: '1069481746876588033   This is how the government has hollowed out the Right to Information Act. Don’t fill vacancies, let petitions pile… https://t.co/eltQaU6IRv']
    Top topic: Business [Post: '1069481367388471296 In a major breakthrough, Gujarat police arrests three persons, including two BJP workers, for their alleged involve… https://t.co/U9Hk1ksgoO']
    ....

#####3. Download latest feed from Twitter and annotate each post with the top topic.
Downlaodes latest 20 posts from the user's twitter feed and adds topic to them.

`python3 SocialTopicExtractor.py -m TestOnline`

See the magic for yourself! 
> You must have your Twitter API credentials in the environment.

### Credits
-------
I would like to sincerely thank my mentor and TA `Bhavya` for guiding me towards GuidedLDA (pun intended!). There were several road blocks in the project and she did help me navigate through it. 
Ofcourse credit to Prof. Zhai for the excelent course on Information Retrieval which inspired and gave me the theoritical understanding to solve this problem.

### License
-------
The library is licensed under Version 2.0 of the Mozilla Public License.
