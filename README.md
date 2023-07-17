### Contrib_schema
The following steps have been followed for making an example
 1. Analysis of the possible uses of the class
 2. Hints and conclussions about toxic language generation 


####  Functional Analysis
  * What can be used for ?
    1. Avoiding Toxicity . Decoding-based strategy. Defined by Jigsaw Perspective API defines toxicity [1](https://support.perspectiveapi.com/s/about-the-api-faqs?language=en_US) as a rude, disrespectful, or unreasonable comment that is likely to make you leave a discussion.
    Defined by Pavlopoulus et al [2](https://arxiv.org/pdf/2006.00998.pdf) as an "umbrella term" , compromising several subtypes, such as offensive, abusive, and hateful language.

        Conclusion : Broad term capturing various forms of offernsive, problematic or harmful language.
        Context matters when determining what´s toxic. Therefore it makes sense to have a class that allows us to control bad words with respect to context.
        Limitations : While the use of Bad words seem to indicate toxicity, language without such words can still be toxic.

    2. Blocklisting [3](). Block undesirable words , preventing them from being generated. Not necessarly toxic, but undesirable context dependent.
       Note that Blocklisting technique can come together with others as well 

#### Hints about the impact of the class
    

  * Two mental models for designing examples
      1. Toxicity in Context .Design an example in which a certain word in certain context can be considered a bad word. 
      2. Detoxifying Generations. Take examples from research in REALTOXICITYPROMPTS [4](https://aclanthology.org/2020.findings-emnlp.301.pdf) and detoxify them.


#### Class Analysis
  * Class Description. Find [NoBadWordsLogitsProccesor](https://github.com/huggingface/transformers/blob/f1732e1374a082bf8e43bd0e4aa8a2da21a32a21/src/transformers/generation/logits_process.py#L725)
    
  * Arguments
    * bad_words_ids . List of list of token ids that are not allowed to be generated.
    * eos_token_id . The id of the end-of-sequence token. Optionally, use a list to set multiple end-to-endsequence tokens.
  * History / Evolution . Discussions in [#22168](https://github.com/huggingface/transformers/issues/22168)


#### References 
 Toxicity in AI generation [](https://towardsdatascience.com/toxicity-in-ai-text-generation-9e9d9646e68f)
