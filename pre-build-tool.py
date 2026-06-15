from ddgs import DDGS

def search(query: str):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return "\n".join(r["body"] for r in results)

results = search("What is the capital of France?")
print(results)

