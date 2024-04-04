# Abbreviation finder

For many people abbreviations are the quick hack to increase writing speed.

How do we find though what the abbreviations are someone should use?

from [this guide](https://vasilishynkarenka.com/how-to-type-3x-faster/):

> Here’s how to integrate shortcuts into your workflow:
> 
> - Go bottoms-up. Don’t just download 300 most popular English words but make shortcuts based on your unique needs. This means,
> - Look for big words. You must become the Sherlock of your own typing. Whenever you catch a big word that you type often, go ahead and set up a shortcut for it. The bigger the word, the more benefit you get. For example, you can take the word “progressive” and turn it into a “pg” shortcut. This reduces the number of symbols you need to type 5x and removes any possibility of typos.
> - Look for complex words. Anything that requires special symbols, capital letters, or apostrophes must be turned into shortcuts. For example, I have “ive” shortcut for “I’ve” and type two symbols less each time I get to write this word.
> - Design shortcuts with ergonomics in mind. You must make them as convenient as possible to get really fast. Thus, for a two-symbol shortcut, you need two symbols in different parts of the keyboard. This enables typing the shortcut with two or three different fingers instead of one.
> - Use the first two letters of the word to make shortcuts easy to remember. The first and last letter will do as well. But if you set up a shortcut that doesn’t make sense to you, you won’t remember it. And shortcuts only help if you use them. For example, I have “ab” for “about” and “rd” for “realized.” For word combos, use first letters of both words; for example, I have a “fe” shortcut that transforms into “for example” and “ac” shortcut that becomes “auto compound” after I hit the space bar.
> - Design different variations of a shortcut for the same word. When you start typing three times faster, you will inevitably face a system not picking up keys in the right order. Therefore, it’s useful to have different variations of the same shortcut for the same word. For example, I have both “dnt” and “dont” expanding into “don’t.”

## Using this tool

1. Install requirements

```bash

pip install nltk

```

2. Prepare your docs as txts

lots of ways to do this. but one way is to use pandoc

```bash

pandoc input.docx -t plain -o output.txt
```

Put all your docs to review in a target folder e.g ``txts``


3. Run it

```bash
python abbreviation-finder.py path/to/txts-directory
```

It then spits out some words/sentences e.g


    speaking -> sp
    Consider -> co
    directly -> di
    preferred -> pr
    practice -> pr
    literacy -> li
    extended -> ex
    languages -> la
    confused -> co
    educational -> ed
    telephone -> te
    important -> im
    language -> la
    situation -> si
    sentences -> se
    development -> de
    conversations -> co
    activity -> ac
    alongside -> al
    more languages -> ml
    can not -> cn
    bilingual children -> bc
    use words -> uw


enjoy! 

## Refs

- https://news.ycombinator.com/item?id=39916366#39926892
- https://vasilishynkarenka.com/how-to-type-3x-faster/
- https://blog.abreevy8.io/you-dont-have-to-type-faster-to-type-faster/