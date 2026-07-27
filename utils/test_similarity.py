from similarity import (
    text_similarity,
    redundancy_score,
    diversity_score,
    revision_delta,
    similarity_matrix,
)

texts = [
    "The user might contemplate whether the outcome of the battle, determined by unforeseen environmental factors like fog rather than material advantage, can be viewed through an ethical lens, specifically questioning if the resulting victory aligns with principles of fairness or how the immense suffering involved in the war is adequately addressed, regardless of which side prevailed.",
    "The user likely reasons that despite the strategic imbalance favoring the Blue army in terms of ground forces, the final outcome, including the element of fortune introduced by the heavy fog, cannot be judged solely on material strength. They are considering how concepts of fairness and prudence apply to attributing victory or defeat, wondering whether the physical reality (the fog) serves as a neutral arbiter that overrides pre-existing power imbalances, thus prompting a reflection on whether an outcome is ethically justifiable based on circumstantial rather than inherent military advantage.",
    "The user is likely grappling with the ethical implications of conflict resolution, wondering how situational factors\u2014such as environmental conditions like heavy fog\u2014interact with pre-existing power imbalances and established obligations to determine a perceived victor, seeking a moral justification for the outcome beyond mere military success.",
]

print("Similar moral claims:")
print(text_similarity(texts[0], texts[1]))

print("\nUnrelated claims:")
print(text_similarity(texts[0], texts[2]))

print("\nRedundancy:")
print(redundancy_score(texts))

print("\nDiversity:")
print(diversity_score(texts))

print("\nRevision delta:")
print(revision_delta(texts[0], texts[1]))

print("\nSimilarity matrix:")
print(similarity_matrix(texts))