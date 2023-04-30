import random
import string
import textwrap

def random_word(length: int) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_text_with_tokens(num_tokens: int) -> str:
    words = []
    while len(words) < num_tokens:
        word_length = random.randint(1, 10)
        if len(words) + word_length <= num_tokens:
            words.append(random_word(word_length))
        else:
            break

    remaining_tokens = num_tokens - len(words)
    if remaining_tokens > 0:
        words.append(random_word(remaining_tokens))

    return ' '.join(words)

# Generate a text with 2048 tokens
text = generate_text_with_tokens(1024)

# Print the text with a fixed width for readability
print(textwrap.fill(text, width=80))
