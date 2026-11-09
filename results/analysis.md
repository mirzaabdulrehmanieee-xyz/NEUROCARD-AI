# Task 6 – Model Analysis

## 6.1 A Model That Always Predicts "N"

The test dataset contains 18,118 samples belonging to class N out of a total of 21,892 test samples.

If a model always predicts class N, its test accuracy would be:

\[
Accuracy = \frac{18118}{21892} \times 100
\]

\[
Accuracy = 82.76\%
\]

Our trained neural network achieves a test accuracy of **97.80%**, which is approximately 15.04 percentage points higher than the majority-class baseline.

This comparison shows that accuracy alone can be misleading when dealing with an imbalanced dataset. A model that always predicts the majority class can achieve a relatively high accuracy without correctly identifying minority classes. Therefore, per-class accuracy and balanced accuracy are also important evaluation metrics.

## 6.2 Class with the Lowest Per-Class Accuracy

According to the evaluation results, class S has the lowest per-class accuracy of **64.57%**.

The confusion matrix shows that out of 556 actual S-class samples:

- 359 samples were correctly classified as S.
- 186 samples were incorrectly classified as N.
- 9 samples were classified as V.
- 2 samples were classified as Q.

Therefore, class S is most frequently mistaken for class N.

One possible reason is that some S and N heartbeat waveforms may have similar morphological characteristics, making them difficult to distinguish. Additionally, class S has fewer training samples than class N, which may limit the network's ability to learn its patterns effectively.

## 6.3 Training and Validation Loss

The training and validation loss curves both decrease as the number of epochs increases.

The training loss decreases from approximately 0.176 to 0.050, while the validation loss decreases from approximately 0.177 to 0.072.

The validation loss remains slightly higher than the training loss, creating a small gap between the two curves. However, both curves follow a similar decreasing trend, indicating that the model is learning effectively.

The model shows slight overfitting because the validation loss remains higher than the training loss. However, there is no strong evidence of severe overfitting or underfitting, as the validation loss also decreases overall. The network demonstrates reasonable generalization performance.

## 6.4 Weight Initialization and He Initialization

Initializing every weight to zero would cause all neurons in the same layer to learn identical features because they would receive identical gradients during training. This problem is known as the symmetry problem.

To avoid this, weights are initialized with random values so that different neurons can learn different features.

He initialization uses a variance of:

\[
Var(W) = \frac{2}{n_{in}}
\]

where \(n_{in}\) represents the number of input neurons in the previous layer.

He initialization is suitable for ReLU activation functions because ReLU sets negative values to zero. It helps maintain a suitable activation variance across layers and reduces the risk of vanishing or exploding activations.

## 6.5 Effect of Momentum and Weight Decay

**Momentum:**

Momentum uses information from previous mini-batch gradients to smooth parameter updates. When two consecutive mini-batch gradients point in opposite directions, their effects partially cancel each other. This reduces sudden changes in the update direction and makes training more stable.

**Weight Decay:**

Weight decay gradually pulls weights toward zero. Even if a weight receives no gradient, weight decay can still reduce its magnitude, provided the weight is nonzero.

This helps control the magnitude of model parameters and can reduce overfitting.

## 6.6 Helping the Network Learn Rare Classes

Class F has only 641 training samples, making it one of the rare classes in the dataset.

One possible improvement is to use **class-weighted cross-entropy loss**.

In this approach, higher weights are assigned to minority classes and lower weights to majority classes. Consequently, incorrect predictions for rare classes receive a larger penalty during training.

This encourages the network to pay more attention to minority classes and learn their patterns more effectively.

Class-weighted cross-entropy could therefore improve the classification performance of rare heartbeat classes. This is a proposed improvement and does not need to be implemented for this task.

---

## Overall Conclusion

The neural network achieved a test accuracy of **97.80%** and a balanced accuracy of **84.65%** on the MIT-BIH heartbeat dataset.

The results show that the model performs very well on majority classes, particularly N and Q. However, the lower accuracy of class S indicates that identifying minority heartbeat classes remains challenging.

The comparison between overall accuracy and balanced accuracy highlights the importance of evaluating models using multiple metrics, especially when working with imbalanced datasets.

Overall, the model demonstrates strong classification performance, while class-weighted loss is a potential improvement for handling rare classes in future work.