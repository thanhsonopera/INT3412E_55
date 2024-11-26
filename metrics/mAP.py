def compute_ap(similar, retrieved):
    """
    Compute Average Precision (AP) for a single query.
    """
    relevant_count = len(similar)
    if relevant_count == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, img in enumerate(retrieved, start=1):
        if img in similar:
            hits += 1
            precision_sum += hits / i

    return precision_sum / relevant_count


def compute_map(data, retrieved):
    """
    Compute mAP for all queries.
    """
    aps = []
    for i, entry in enumerate(data.values()):
        query = entry["query"]
        similar = set(entry["similar"])
        ap = compute_ap(similar, retrieved[i])
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0
