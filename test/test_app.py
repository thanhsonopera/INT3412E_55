if int(dataset) == 1:
    import json
    from ultis import get_all_image_paths, extract_sift_features
    image_directory = "datasets/INRIA/images/"

    image_paths = get_all_image_paths(image_directory)
    with open('datasets/INRIA/groundtruth.json', 'r') as file:
        data = json.load(file)
        for key in data:
            new_similar = []
            for img in data[key]['similar']:
                for idx, path in enumerate(image_paths):
                    if img in path:
                        new_similar.append(idx)
            data[key]['similar'] = new_similar

    outputs = []
    for key in data:
        path = 'datasets/INRIA/images/' + data[key]['query']
        key_points, descriptors = extract_sift_features(path)
        if re == 1:
            query_vlad = vlad.transform([descriptors])

            probabilities = adc.predict_proba(query_vlad)
        elif re == 2:
            query_bof = bof.transform([descriptors])
            probabilities = adc.predict_proba(query_bof)
        elif re == 3:
            query_fish = fish.transform([descriptors])
            probabilities = adc.predict_proba(query_fish)
        sorted_indices = np.argsort(probabilities)

        output_per_query = []
        for i, index in enumerate(sorted_indices):
            print(
                f"Original index: {index}, Probability: {probabilities[index]}")
            if i > 0:
                output_per_query.append(index)

                if len(output_per_query) == 30:
                    break

        outputs.append(output_per_query)

    from metrics.mAP import compute_map
    results = compute_map(data, outputs)
    print(results)
else:
    from ultis import get_all_image_paths, extract_sift_features
    image_directory = "datasets/INRIA/images/"

    image_paths = get_all_image_paths(image_directory)
    from metrics.score4 import calculate_gr_score
    print('Running')
    similarity_matrix = []
    for path in image_paths:
        key_points, descriptors = extract_sift_features(path)

        if re == 1:
            query_vlad = vlad.transform([descriptors])
            probabilities = adc.predict_proba(query_vlad)
        elif re == 2:
            query_bof = bof.transform([descriptors])
            probabilities = adc.predict_proba(query_bof)
        elif re == 3:
            query_fish = fish.transform([descriptors])
            probabilities = adc.predict_proba(query_fish)

        similarity_matrix.append(probabilities)

    result = calculate_gr_score(np.array(similarity_matrix))
    print(result)
