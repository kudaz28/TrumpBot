# LTSM (Recurrent Neural Network) for Tweet Generation

This is an experiment in using a double stacked LTSM to build a character model that can generate novel tweets.

Here's some examples of this trained on political tweets, and then fine tuned on the tweets of Donald Trump...

       diversity: 0.2
       ----- Generating with seed: "ioofficial i will be"
       ioofficial i will be a great deal with any marcon of the country would be a great president who did the republicans we must be sure to change the right things d

       ----- diversity: 0.5
       ----- Generating with seed: "ioofficial i will be"
       ioofficial i will be a great leader and i am there we are on the race in the us will lead to say what i was in las vegas! #trump2016 #makeamericagreatagain #tru

       ----- diversity: 1.0
       ----- Generating with seed: "ioofficial i will be"
       ioofficial i will be on @foxandfriends! berand that you're best fichert maryn? http://tmitt.com/cardzaro/20/trump_speaks-theners in america's threet turnberry,

       ----- diversity: 1.2
       ----- Generating with seed: "ioofficial i will be"
       ioofficial i will be on the hollyfomplianzelecial
       via @dlougdgitt
       rave bad sovie for syria - will be great to help!
       trump chanking iss tweeting in the out-he is

Obviously, the seeds are pretty bad, and the tweets are almost nonsensical. However, it's pretty cool that it learns to form links, makes hashtags, is all about #makeamericagreatagain, and can form words.

This was trained for about 6 hours on a NVIDIA TI 980Ti. Increasing the length of the seed greatly increases training time, so we can't use full tweets as seeds, nor can we do tweets -> replies without running this for a loooong time.

It also has trouble dealing with novel/unseen seeds, which makes me think its severly overfitting. More investigation needed (and more understanding of LTSMs). 