from langchain_core.prompts import PromptTemplate
from langchain_core.load import dumps
import json

# ── Prompt Template ───────────────────────────────────────────────────────────
template = PromptTemplate(
    input_variables=['paper_input', 'style_input', 'length_input'],
    validate_template=True,
    template="""
You are an expert AI research assistant specializing in explaining academic papers clearly and accurately.

Your task is to summarize the research paper titled "{paper_input}" according to the following specifications:

-----------------------------------------------------------------
SUMMARY PREFERENCES
-----------------------------------------------------------------
- Explanation Style  : {style_input}
- Explanation Length : {length_input}

-----------------------------------------------------------------
SUMMARY STRUCTURE
-----------------------------------------------------------------

1. Overview
   - What is the paper about?
   - What problem does it solve?
   - Why is it significant in the field?

2. Core Idea and Approach
   - Explain the main idea or methodology in simple terms.
   - Use relatable analogies to simplify complex concepts.

3. Mathematical Details (if applicable)
   - Include key equations or formulas from the paper.
   - Briefly explain what each equation represents.
   - Where helpful, illustrate with a short intuitive code snippet.

4. Results and Findings
   - What were the key results or benchmarks?
   - How did it compare to previous approaches?

5. Impact and Applications
   - What real-world problems can this paper help solve?
   - What future research directions does it open up?

-----------------------------------------------------------------
IMPORTANT GUIDELINES
-----------------------------------------------------------------
- Strictly follow the requested explanation style ({style_input}) throughout.
- Keep the response within the requested length ({length_input}).
- If any specific information is not available in the paper, respond with:
  "Insufficient information available" — do NOT guess or hallucinate.
- Ensure the summary is clear, accurate, and well-structured.
"""
)

# ── Save method ────────────────────────────────────────
with open("template.json", "w") as f:
    json.dump(json.loads(dumps(template)), f, indent=2)

print("template.json saved successfully!")