# Calibration Analysis

## Reliability diagram interpretation

The reliability diagram for the fine-tuned DistilBERT model shows significant **over-confidence** across almost all confidence buckets. While the model often assigns high probabilities to its predictions, the empirical accuracy consistently lags behind those confidence levels. For example:
- In the **0.9–1.0** confidence bucket, the model has an accuracy of only **0.687**, which is a massive gap (~30%) from the expected accuracy.
- In the **0.7–0.8** confidence bucket, the accuracy is only **0.488**, indicating that when the model is fairly sure, it is effectively wrong half the time.
- The model rarely makes low-confidence predictions, with the vast majority of samples (1,147 out of 1,495) being pushed into the highest confidence bucket despite the high error rate there.

## Expected Calibration Error

The reported **ECE is 0.2881**. In the context of sentiment analysis for app reviews, this number indicates a low level of trustworthiness for production use. An ECE of nearly 0.29 means that, on average, the model's "confidence" is off by 29 percentage points. For a production system, this makes it difficult to rely on the model's probability scores for downstream logic (like flagging only highly negative reviews for human intervention).

## A specific calibration pattern

A prominent pattern is the extreme **over-confidence in the majority class** and a lack of middle-ground confidence. The model has been "pushed" to make very high-confidence predictions (0.9+) for most samples. This often arises because the cross-entropy loss used during fine-tuning encourages the model to increase the logit of the correct class indefinitely, leading to over-confident "peakiness" in the softmax distribution. Without regularization techniques like label smoothing or post-hoc calibration, the model learns to be over-confident even on difficult samples near class boundaries.

## A proposed engineering action

Based on these findings, I would propose implementing **Temperature Scaling** as a post-hoc calibration step. By learning a single scalar parameter `T` on a validation set to soften the softmax probabilities, we can bring the confidence levels closer to the empirical accuracy without changing the model's accuracy. Additionally, for a production rollout, I would recommend a **threshold-based abstention** policy where the model's predictions are only shown to users or acted upon if the (calibrated) probability exceeds a high bar (e.g., 0.9 after scaling), while everything else is sent for human review.
