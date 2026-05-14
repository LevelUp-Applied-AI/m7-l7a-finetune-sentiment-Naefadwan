# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation | 3 | 5 | 60.0% |
| lexical_trigger | 2 | 5 | 40.0% |
| domain_shift | 2 | 5 | 40.0% |
| length_extreme | 3 | 5 | 60.0% |
| sarcasm | 0 | 5 | 0.0% |
| other | 2 | 5 | 40.0% |
| **Total** | **12** | **30** | **40.0%** |

## Confirmed hypotheses

The model failed significantly on **sarcasm** (0/5 correct). For example, in Row IDs 17, 18, and 19, the model predicted "positive" for sentences like "Oh great, another update that breaks everything" (0.82 prob) and "I love how it crashes every time I open a menu" (0.69 prob). This confirms that the model is heavily reliant on surface-level sentiment words ("great", "love") and fails to understand the ironic context.

It also struggled with **lexical_triggers** (2/5 correct). In Row ID 8 ("I am happy to delete this") and Row ID 9 ("great example of how not to design a UI"), the model predicted "positive" (0.98+ prob) despite the clear negative intent. This confirms the hypothesis that strong cue words override context.

## Refuted hypotheses

The model handled **negation** better than expected (3/5 correct). While it missed "I cannot say that I enjoyed this" (Row 4), it correctly identified "did not improve" (Row 1), "fails to provide" (Row 5), and "would not recommend" (Row 24) as negative. This suggests it has learned simple negation patterns fairly well during fine-tuning.

Surprisingly, it correctly handled the scientific fact "The mitochondria is the powerhouse of the cell" (Row 26) as **neutral**, even though it failed on other domain shifts.

## What the results reveal about the decision boundary

The results reveal that the model's decision boundary is highly sensitive to specific "anchor" words. If a sentence contains "love", "great", or "happy", the model has a very high bias towards the "positive" class, regardless of whether those words are negated (sarcastic) or part of a negative phrase ("happy to delete"). 

Furthermore, the model's **neutral** class seems to be a "catch-all" for uncertainty rather than a well-defined sentiment. When faced with domain shifts like recipe instructions (Row 10), it defaulted to "negative" with low-to-mid confidence rather than recognizing the lack of sentiment. This suggests the decision boundary for "neutral" is quite narrow and potentially over-fitted to specific app-review phrasing.
