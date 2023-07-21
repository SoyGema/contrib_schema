### File docstring and test function

class NoBadWordsLogitsProcessor(SequenceBiasLogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be selected.

    <Tip>

    In order to get the token ids of the words that should not appear in the generated text, make sure to set
    `add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words,
    add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers,
    as fast tokenizers' prefixing behaviours come from `pre tokenizers`. Read more
    [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens
            that comes from the instantiated generation configuration file

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")
    
    >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=5 , pad_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a mess.

    ## Now let´s control generation taking the bad words out. Please note that the tokenizer is initialized differently
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)

    >>> def get_tokens_as_list(word_list):
        "Converts a sequence of words into a list of tokens "
        tokens_list = []
        for word in word_list.split(" "):
            tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
            tokens_list.append(tokenized_word)
        return tokens_list

    >>> word_list = "mess"
    >>> bad_words_ids = get_tokens_as_list(word_list=word_list)

    >>> badwords_ids = model.generate(inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, eos_token_id=tokenizer_with_prefix_space.eos_token_id)
    >>> print(tokenizer.batch_decode(badwords_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a surprise.

    >>> badwords_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=5, bad_words_ids=bad_words_ids)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    In a word, the cake is a great way to start


    # Now let´s try with a sequence of words.
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> inputs = tokenizer(["the cake is a"], return_tensors="pt" )
    >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=12 , pad_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    the cake is a bit of a mess, but it's not a problem.

    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
  
    >>> word_list = "but it's not a problem."
    >>> bad_words_ids = get_tokens_as_list(word_list=word_list)

    >>> badwords_ids = model.generate(inputs["input_ids"], max_new_tokens=12, bad_words_ids=bad_words_ids)
    >>> print(tokenizer.batch_decode(badwords_ids, skip_special_tokens=True)[0])

    the cake is a bit of a mystery, but I think it's a good idea
    ```
    """

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: Union[int, List[int]]):
        self.bad_word_ids = bad_words_ids
        self._validate_arguments()

        # Filter EOS token from bad_words_ids
        if eos_token_id is None:
            eos_token_id = []
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        bad_words_ids = list(
            filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id), bad_words_ids)
        )

        # Forbidding a sequence is equivalent to setting its bias to -inf
        sequence_bias = {tuple(sequence): float("-inf") for sequence in bad_words_ids}
        super().__init__(sequence_bias=sequence_bias)

    def _validate_arguments(self):
        bad_words_ids = self.bad_word_ids
        if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.")
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )
