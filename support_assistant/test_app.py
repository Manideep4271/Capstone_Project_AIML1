
from main import ask_question


print("=" * 70)
print("POLICY QUESTION")
print("=" * 70)

result1 = ask_question(
    "What is the delivery fee for orders below INR 149?"
)

print(result1.model_dump_json(indent=2))


print()
print("=" * 70)
print("GENERAL QUESTION")
print("=" * 70)

result2 = ask_question(
    "What is the capital of India?"
)

print(result2.model_dump_json(indent=2))
