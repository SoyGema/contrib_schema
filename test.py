### According to the class documentation, 
# the class NoBadWordsLogitProcessor is equivalent to SequenceBiasLogitsProcessor set to -float(inf)
# Therefore the generation should be the same. The test benchmark against that class with -inf generation



### SequenceBirdsLogitsProcessor with -inf generation. 

from transformers import AutoTokenizer, AutoModelForCausalLM



model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer(["________"], return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])


tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)


def get_tokens_as_tuple(word):
    return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


sequence_bias = {get_tokens_as_tuple("________"): -float("inf")}
biased_ids = model.generate(inputs["input_ids"], max_new_tokens=1, sequence_bias=sequence_bias)
print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
