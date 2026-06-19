import torch

from simple_transformer.config import small_addition_config
from simple_transformer.data import AdditionTokenizer
from simple_transformer.model import SimpleTransformerLM, count_parameters


def test_small_addition_model_forward_pass():
    config = small_addition_config(max_digits=3)
    model = SimpleTransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len - 1))
    labels = torch.randint(0, config.vocab_size, (2, config.max_seq_len - 1))

    output = model(input_ids, labels=labels)

    assert output["logits"].shape == (2, config.max_seq_len - 1, config.vocab_size)
    assert output["loss"].ndim == 0
    assert 900_000 < count_parameters(model) < 1_100_000


def test_model_generates_for_single_and_variable_length_batches():
    tokenizer = AdditionTokenizer()
    config = small_addition_config(max_digits=3)
    model = SimpleTransformerLM(config)
    single_prompt = torch.tensor([tokenizer.encode("12+3=")], dtype=torch.long)
    prompts = [
        tokenizer.encode("1+2="),
        tokenizer.encode("12+3="),
        tokenizer.encode("123+45="),
    ]

    single_output = model.generate(
        single_prompt,
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
    )
    output_ids = model.generate_batch(
        prompts,
        eos_token_id=tokenizer.eos_token_id,
    )

    assert single_output.shape[0] == 1
    assert single_output.shape[1] > single_prompt.shape[1]
    assert len(output_ids) == len(prompts)
    for prompt, output in zip(prompts, output_ids):
        assert output[: len(prompt)].tolist() == prompt
        assert output.shape[0] == config.max_seq_len


def test_generate_batch_matches_full_length_generate_for_same_length_prompts():
    tokenizer = AdditionTokenizer()
    config = small_addition_config(max_digits=3)
    model = SimpleTransformerLM(config)
    prompts = [
        tokenizer.encode("1+2="),
        tokenizer.encode("3+4="),
    ]
    stacked_prompts = torch.tensor(prompts, dtype=torch.long)

    batch_output_ids = model.generate_batch(
        prompts,
        eos_token_id=tokenizer.eos_token_id,
    )
    generate_output_ids = model.generate(
        stacked_prompts,
        max_new_tokens=config.max_seq_len - stacked_prompts.size(1),
        eos_token_id=tokenizer.eos_token_id,
    )

    assert torch.equal(torch.stack(batch_output_ids), generate_output_ids)


def test_cached_forward_matches_full_forward():
    tokenizer = AdditionTokenizer()
    config = small_addition_config(max_digits=3)
    model = SimpleTransformerLM(config)
    model.eval()
    input_ids = torch.tensor([tokenizer.encode("12+34=46", add_eos=True)])

    with torch.no_grad():
        full_logits = model(input_ids)["logits"]
        cache = model.new_cache(batch_size=1)
        first_logits = model(input_ids[:, :5], cache=cache)["logits"]
        second_logits = model(input_ids[:, 5:], cache=cache)["logits"]

    cached_logits = torch.cat((first_logits, second_logits), dim=1)
    assert torch.allclose(cached_logits, full_logits, atol=1e-5)
