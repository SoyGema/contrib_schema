from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"], max_new_tokens=5 , pad_token_id=tokenizer.eos_token_id)

print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])

tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)

def get_tokens_as_list(word_list):
    """Converts a sequence of words into a list of tokens """
    tokens_list = []
    for word in word_list.split(" "):
        tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list

word_list = "mess"
bad_words_ids = get_tokens_as_list(word_list=word_list)

badwords_ids2 = model.generate(inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, eos_token_id=tokenizer_with_prefix_space.eos_token_id)

print(tokenizer.batch_decode(badwords_ids2, skip_special_tokens=True)[0])

biased_ids = model.generate(inputs["input_ids"], max_new_tokens=5, num_beams=5, bad_words_ids=bad_words_ids)
print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
